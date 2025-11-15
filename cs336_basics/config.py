from dataclasses import dataclass
import math

@dataclass
class Config:
    # Models
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    context_length: int = 256
    rope_theta: float = 1e4
    vocab_size: int = 50_257
    num_layers: int = 4

    # Optimizers
    max_lr: float = 1e-3 
    min_lr: float = 3e-5
    t_w: int = 1_000
    t_c: int = 100_000
    grad_clip_cap: float = 5.0

    # System
    exp_name: str = "dev"
    torch_compile: bool = True
    batch_size: int = 256
    device: str = "cuda"
    target_token: int = 600_000_000
    eval_interval: int = 100
    ckpt_interval: int = 1_000
    ckpt_path: str = "/Users/chengze/work/336_assignmen1/ckpt/"
    train_data_path: str = "/Users/chengze/work/336_assignmen1/data/TinyStoriesV2-GPT4-train.npy"
    valid_data_path: str = "/Users/chengze/work/336_assignmen1/data/TinyStoriesV2-GPT4-valid.npy"
    
    @property
    def max_iter(self) -> int:
        return math.ceil(self.target_token / self.context_length / self.batch_size)
