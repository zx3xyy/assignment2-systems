from dataclasses import dataclass
import math

MODEL_PRESETS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
    
@dataclass
class Config:
    # Models
    model_name: str = ""
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    context_length: int = 256
    rope_theta: float = 1e4
    vocab_size: int = 10_000
    num_layers: int = 12

    # Optimizers
    max_lr: float = 1e-3 
    min_lr: float = 3e-5
    t_w: int = 1_000
    t_c: int = 100_000
    grad_clip_cap: float = 5.0

    # System
    exp_name: str = "dev"
    torch_compile: bool = True
    batch_size: int = 4
    device: str = "cuda"
    target_token: int = 600_000_000
    eval_interval: int = 100
    ckpt_interval: int = 1_000
    ckpt_path: str = "/Users/chengze/work/336_assignmen1/ckpt/"
    train_data_path: str = "/Users/chengze/work/336_assignmen1/data/TinyStoriesV2-GPT4-train.npy"
    valid_data_path: str = "/Users/chengze/work/336_assignmen1/data/TinyStoriesV2-GPT4-valid.npy"

    # Benchmark
    benchmark_backward: bool = False
    n_iter_warmup: int = 5
    n_iter_benchmark: int = 5
    
    # Stage
    train: bool = False
    benchmark: bool = True
        
    @property
    def max_iter(self) -> int:
        return math.ceil(self.target_token / self.context_length / self.batch_size)


def process_config(cfg: Config):
    preset = MODEL_PRESETS.get(cfg.model_name)
    if preset is None:
        return
    for k, v in preset.items():
        setattr(cfg, k, v)