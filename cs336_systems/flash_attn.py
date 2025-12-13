import torch
from torch import Tensor
from jaxtyping import Float, Bool, jaxtyped
from beartype import beartype
import math
import torch.nn.functional as F
import triton
import triton.language as tl


class TorchFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, "batch q_len d_model"],
        K: Float[Tensor, "batch k_len d_model"],
        V: Float[Tensor, "batch v_len d_model"],
        is_casual: bool,
    ):
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        bs, N_QUERIES, d_model = Q.shape
        N_KEYS = K.shape[1]
        T_q = N_QUERIES // Q_TILE_SIZE
        T_k = N_KEYS // K_TILE_SIZE
        O = torch.zeros((bs, N_QUERIES, d_model)).to(Q.device)
        L = torch.zeros((bs, T_q, Q_TILE_SIZE)).to(Q.device)
        mask = (
            torch.tril(torch.ones((N_QUERIES, N_KEYS)))
            if is_casual
            else torch.ones((N_QUERIES, N_KEYS))
        )
        mask = mask.unsqueeze(0).expand(bs, -1, -1).to(Q.device)

        for b in range(bs):
            for i in range(T_q):
                q_tile = Q[b, i * Q_TILE_SIZE : (i + 1) * Q_TILE_SIZE, :]
                o_tile = torch.zeros((Q_TILE_SIZE, d_model)).to(Q.device)
                l = torch.zeros((Q_TILE_SIZE,)).to(Q.device)
                m = torch.full((Q_TILE_SIZE,), -torch.inf).to(Q.device)
                for j in range(T_k):
                    k_tile = K[b, j * K_TILE_SIZE : (j + 1) * K_TILE_SIZE, :]
                    v_tile = V[b, j * K_TILE_SIZE : (j + 1) * K_TILE_SIZE, :]
                    mask_tile = mask[
                        b,
                        i * Q_TILE_SIZE : (i + 1) * Q_TILE_SIZE,
                        j * K_TILE_SIZE : (j + 1) * K_TILE_SIZE,
                    ]
                    s_tile = q_tile @ k_tile.T / math.sqrt(d_model)
                    s_tile = s_tile.masked_fill(mask_tile == 0, -float("inf"))
                    row_max = torch.max(s_tile, dim=-1).values
                    old_m = m
                    m = torch.maximum(row_max, m)
                    scale_factor = torch.exp(old_m - m)
                    p_tile = torch.exp(s_tile - m.unsqueeze(-1))  #  softmax 分子
                    l = scale_factor * l + torch.sum(p_tile, dim=-1)  # softmax 分母
                    o_tile = scale_factor.unsqueeze(-1) * o_tile + p_tile @ v_tile
                L[b, i, :] = m + torch.log(l)
                O[b, i * Q_TILE_SIZE : (i + 1) * Q_TILE_SIZE, :] = o_tile / l.unsqueeze(
                    -1
                )
        ctx.save_for_backward(Q, K, V, O, L.view(bs, -1))
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()


# fmt: off
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, # Inputs
    O_ptr, L_ptr, # Outputs
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
): 
# fmt: on
    query_tile_index = tl.program_id(0) # parallize over queries because seq_len dim is embarassingly parallelized
    batch_index = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    
    hi = N_KEYS
    if IS_CAUSAL:
        hi = min((query_tile_index + 1) * Q_TILE_SIZE, N_KEYS)
    
    offs_m = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)   
    for start_k in range(0, hi, K_TILE_SIZE):
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        s = tl.dot(q, k.T) * scale
        offs_n = start_k + tl.arange(0, K_TILE_SIZE)
        if IS_CAUSAL and (start_k + K_TILE_SIZE > query_tile_index * Q_TILE_SIZE):
                mask = offs_n[None, :] > offs_m[:, None]
                s = tl.where(mask, -1.0e6, s)
        row_max = tl.max(s, axis=-1)
        old_m = m
        m = tl.maximum(row_max, m)
        scale_factor = tl.exp(old_m - m)
        p = tl.exp(s - m[:, None])
        l = scale_factor * l + tl.sum(p, axis=-1)
        o = o * scale_factor[:, None]
        o = tl.dot(p.to(v.dtype), v, acc=o)
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    o = o / l[:, None]
    l = m + tl.log(l)
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, l)

    

    
class TritonFlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        Q: Float[Tensor, "batch q_len d_model"],
        K: Float[Tensor, "batch k_len d_model"],
        V: Float[Tensor, "batch v_len d_model"],
        is_causal: bool = False,
    ):
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 16
        bs, N_QUERIES, d_model = Q.shape
        N_KEYS = K.shape[1]
        T_q = triton.cdiv(N_QUERIES, Q_TILE_SIZE)
        O = torch.zeros((bs, N_QUERIES, d_model), device = Q.device)
        L = torch.zeros((bs, N_QUERIES), device = Q.device)
        grid = (T_q, bs)
        flash_fwd_kernel[grid](
            Q, K, V, 
            O, L,Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale=1.0 / (d_model ** 0.5),
            D=d_model, # type: ignore
            Q_TILE_SIZE=Q_TILE_SIZE, # type: ignore
            K_TILE_SIZE=K_TILE_SIZE, # type: ignore
            IS_CAUSAL=is_causal, # type: ignore
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()
