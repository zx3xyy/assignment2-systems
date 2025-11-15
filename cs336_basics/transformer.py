import torch
from torch import nn
from einops import rearrange, einsum
import math
import einx
from jaxtyping import Float, Int
from torch import Tensor


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=torch.float
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.rand(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight, mean=0, std=std, a=-3 * std, b=3 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.rand(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.weight = nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        sum_of_squares = einx.sum("... d_model ->... 1", x**2)
        rms = torch.sqrt(sum_of_squares / self.d_model + self.eps)
        # [..., d_model] * [d_model] / [..., 1]
        res = x * self.weight / rms
        return res.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.silu = SiLU()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Float[Tensor, " ... d_model"]):
        gate_activation = self.w1(x)
        gate = self.silu(gate_activation)
        activation = self.w3(x)
        activation = activation * gate
        return self.w2(activation)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        rotation_matrices = torch.zeros([max_seq_len, d_k, d_k], device=device)

        for i in range(max_seq_len):
            for k in range(self.d_k // 2):
                angle = i / (theta ** (2 * k / d_k))
                rotation_matrices[i, 2 * k, 2 * k] = math.cos(angle)
                rotation_matrices[i, 2 * k, 2 * k + 1] = -math.sin(angle)
                rotation_matrices[i, 2 * k + 1, 2 * k] = math.sin(angle)
                rotation_matrices[i, 2 * k + 1, 2 * k + 1] = math.cos(angle)
        self.register_buffer("rotation_matrices", rotation_matrices)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Float[Tensor, " ... seq_len"],
    ) -> Float[Tensor, " ... seq_len d_k"]:
        rotation = einx.get_at(
            "[pos] d_k1 d_k2, ... seq_len -> ... seq_len d_k1 d_k2",
            self.rotation_matrices,
            token_positions.long(),
        )
        x = einsum(
            rotation, x, "... seq_len d_k1 d_k2, ... seq_len d_k2 -> ... seq_len d_k1"
        )
        return x


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int):
        max_val, _ = torch.max(x, dim=dim, keepdim=True)
        stable_x = x - max_val
        exp = torch.exp(stable_x)
        s = torch.sum(exp, dim=dim, keepdim=True)
        return exp / s


# when seq_len = seq_len_2, it's self-attention.
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... seq_len d_k"],
    K: Float[Tensor, " ... seq_len_2 d_k"],
    V: Float[Tensor, " ... seq_len_2 d_v"],
    mask: Float[Tensor, " ... seq_len seq_len_2"] | None = None,
) -> Float[Tensor, "... seq_len d_v"]:
    d_k = Q.shape[-1]
    QTK = einsum(Q, K, " ... seq_len d_k, ... seq_len_2 d_k -> ... seq_len seq_len_2")
    mask = torch.where(mask, 0.0, -torch.inf)
    masked_QTK = QTK + mask
    attn_score = Softmax()(masked_QTK / math.sqrt(d_k), dim=-1)
    return einsum(
        attn_score, V, "... seq_len seq_len_2, ... seq_len_2 d_v -> ... seq_len d_v"
    )


class CausalMultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        if max_seq_len and theta:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        in_features: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_out"]:
        seq_len = in_features.shape[-2]
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)
        Q = rearrange(
            Q,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        K = rearrange(
            K,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        V = rearrange(
            V,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=in_features.device)
        )
        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = rearrange(
            attn_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
        )
        res = self.output_proj(attn_output)
        return res


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttentionWithRope(
            d_model, num_heads, max_seq_len, theta
        )
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]):
        seq_len = x.shape[1]
        token_positions = torch.rand([1, seq_len])
        for i in range(seq_len):
            token_positions[:, i] = i
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        rope_theta: float,
        vocab_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.softmax = Softmax()

    def forward(
        self, x: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(x)  # b, seq, d_model
        for layer in self.layers:
            x = layer(x)  # b, seq, d_model
        x = self.ln_final(x)  # b, seq, d_model
        x = self.lm_head(x)  # b, seq, d_vocab
        return x
