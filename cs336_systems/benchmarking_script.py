import logging
import os
import subprocess
import sys
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

from contextlib import nullcontext
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
import cs336_basics.model
import torch.cuda.nvtx as nvtx
from einops import einsum
import math
from cs336_basics.nn_utils import softmax

log = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    context_length: int = 512
    d_model: int = 2560
    d_ff: int = 10240
    num_layers: int = 32
    num_heads: int = 32
    rope_theta: float = 10000.0


@dataclass
class BenchmarkConfig:
    warmup_iters: int = 5
    run_iters: int = 10
    verbose: bool = True
    batch_size: int = 4
    device: str = "cuda"
    mode: str = "forward_only"
    precision: str = "fp32"
    compile: bool = False
    compile_mode: str = "default"
    profile: bool = False
    autocast: bool = False
    memory_snapshot: bool = False


@dataclass
class Config:
    experiment_name: str = "benchmark"
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


cs = ConfigStore.instance()
cs.store(name="config_schema", node=Config)


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attention_scores = einsum(
            Q, K, "... query d_k, ... key d_k -> ... query key"
        ) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with nvtx.range("computing softmax"):
        attention_weights = softmax(
            attention_scores, dim=-1
        )  # Softmax over the key dimension
    with nvtx.range("final matmul"):
        res = einsum(
            attention_weights, V, "... query key, ... key d_v ->  ... query d_v"
        )
    return res


cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def get_git_info() -> dict[str, str]:
    """Get current git information."""
    try:
        # Run git from the original working directory
        cwd = hydra.utils.get_original_cwd()
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=cwd
            )
            .decode("utf-8")
            .strip()
        )
        git_hash_short = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=cwd,
            )
            .decode("utf-8")
            .strip()
        )
        git_branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=cwd,
            )
            .decode("utf-8")
            .strip()
        )
        git_dirty = (
            subprocess.call(
                ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL, cwd=cwd
            )
            != 0
        )
        return {
            "hash": git_hash,
            "hash_short": git_hash_short,
            "branch": git_branch,
            "dirty": git_dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "hash": "unknown",
            "hash_short": "unknown",
            "branch": "unknown",
            "dirty": False,
        }


def get_system_info() -> dict[str, Any]:
    """Get system information for reproducibility."""
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
    }


def sync_device(device: str):
    """Synchronize CUDA device."""
    torch.cuda.synchronize()


def generate_random_data(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a random dataset of token IDs."""
    dataset_size = 100000
    dataset = np.random.randint(
        0, cfg.model.vocab_size, size=(dataset_size,), dtype=np.int64
    )

    x, y = get_batch(
        dataset,
        cfg.benchmark.batch_size,
        cfg.model.context_length,
        cfg.benchmark.device,
    )

    if cfg.benchmark.verbose:
        log.info(f"Input batch shape: {x.shape}")
        log.info(f"Target batch shape: {y.shape}")
        log.info(f"Device: {x.device}")

    return x, y


def init_model(cfg: Config) -> BasicsTransformerLM:
    """Initialize the transformer model."""
    model = BasicsTransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
    )

    # Handle precision
    dtype = torch.float32
    precision = cfg.benchmark.precision
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16

    model = model.to(cfg.benchmark.device, dtype=dtype)

    if cfg.benchmark.compile:
        compile_mode = cfg.benchmark.compile_mode
        log.info(f"Compiling model with torch.compile(mode='{compile_mode}')...")
        model = torch.compile(model, mode=compile_mode)

    if cfg.benchmark.verbose:
        log.info(
            f"Model initialized with {model.get_num_params() / 1e6:.2f}M parameters (precision: {precision})"
        )

    return model


def init_wandb(cfg: Config, git_info: dict[str, str], system_info: dict[str, Any]):
    """Initialize Weights & Biases logging."""
    # Convert OmegaConf to dict for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["git"] = git_info
    config_dict["system"] = system_info

    # Get reproduce command
    reproduce_command = " ".join(sys.argv)
    config_dict["reproduce_command"] = reproduce_command

    wandb.init(
        project="cs336-systems-benchmark",
        name=cfg.experiment_name,
        config=config_dict,
        tags=[cfg.benchmark.mode, git_info["hash_short"]],
    )


def log_to_csv(cfg: Config, results: dict[str, float], git_info: dict[str, str]):
    """Log results to a central CSV file."""
    import csv

    original_cwd = hydra.utils.get_original_cwd()
    # Use a different CSV file for autocast experiments as requested
    csv_filename = "benchmark_autocast_results.csv"
    central_csv_path = Path(original_cwd) / csv_filename

    file_exists = central_csv_path.exists()

    # Get model name from hydra config choice if available
    try:
        model_name = hydra.core.hydra_config.HydraConfig.get().runtime.choices.model
    except Exception:
        model_name = "unknown"

    # Flatten config for CSV
    flat_results = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": cfg.experiment_name,
        "git_hash": git_info["hash_short"],
        "model_name": model_name,
        "vocab_size": cfg.model.vocab_size,
        "context_length": cfg.model.context_length,
        "d_model": cfg.model.d_model,
        "num_layers": cfg.model.num_layers,
        "num_heads": cfg.model.num_heads,
        "d_ff": cfg.model.d_ff,
        "batch_size": cfg.benchmark.batch_size,
        "mode": cfg.benchmark.mode,
        "precision": cfg.benchmark.precision,
        "compile": cfg.benchmark.compile,
        "compile_mode": cfg.benchmark.compile_mode,
        "autocast": cfg.benchmark.autocast,
        **results,
    }

    with open(central_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_results)

    log.info(f"Results appended to {central_csv_path}")


def benchmark_forward(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    cfg: Config,
) -> tuple[float, float]:
    """Benchmark forward pass only. Returns (avg_time_ms, peak_memory_gb)."""
    device = cfg.benchmark.device

    # Warmup
    if cfg.benchmark.verbose:
        log.info(f"[Forward] Running {cfg.benchmark.warmup_iters} warmup iterations...")

    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if cfg.benchmark.autocast
        else nullcontext()
    )

    with torch.cuda.nvtx.range("warmup"):
        for _ in range(cfg.benchmark.warmup_iters):
            with torch.no_grad():
                with ctx:
                    _ = model(x)
    sync_device(device)

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Benchmark
    if cfg.benchmark.verbose:
        log.info(f"[Forward] Running {cfg.benchmark.run_iters} benchmark iterations...")

    sync_device(device)

    if cfg.benchmark.memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    with torch.cuda.nvtx.range("benchmark_loop"):
        start = time.perf_counter()
        for _ in range(cfg.benchmark.run_iters):
            with torch.no_grad():
                with ctx:
                    _ = model(x)
        sync_device(device)
        end = time.perf_counter()

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    if cfg.benchmark.memory_snapshot:
        torch.cuda.memory._dump_snapshot(
            f"{cfg.model.context_length}_forward_memory_snapshot.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)
        torch.cuda.reset_peak_memory_stats()

    avg_time_ms = (end - start) / cfg.benchmark.run_iters * 1000
    return avg_time_ms, peak_memory_gb


def benchmark_forward_backward(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
) -> tuple[float, float]:
    """Benchmark forward + backward pass. Returns (avg_time_ms, peak_memory_gb)."""
    device = cfg.benchmark.device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Warmup
    if cfg.benchmark.verbose:
        log.info(
            f"[Forward+Backward] Running {cfg.benchmark.warmup_iters} warmup iterations..."
        )

    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if cfg.benchmark.autocast
        else nullcontext()
    )

    with torch.cuda.nvtx.range("warmup"):
        for _ in range(cfg.benchmark.warmup_iters):
            with ctx:
                outputs = model(x)
                loss = criterion(outputs.view(-1, cfg.model.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    sync_device(device)

    # Benchmark
    if cfg.benchmark.verbose:
        log.info(
            f"[Forward+Backward] Running {cfg.benchmark.run_iters} benchmark iterations..."
        )

    sync_device(device)

    if cfg.benchmark.memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    with torch.cuda.nvtx.range("benchmark_loop"):
        start = time.perf_counter()
        for _ in range(cfg.benchmark.run_iters):
            with torch.cuda.nvtx.range("forward"):
                with ctx:
                    outputs = model(x)
            with torch.cuda.nvtx.range("backward"):
                with ctx:
                    loss = criterion(outputs.view(-1, cfg.model.vocab_size), y.view(-1))
                loss.backward()
            with torch.cuda.nvtx.range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()
        sync_device(device)
        end = time.perf_counter()

    if cfg.benchmark.memory_snapshot:
        torch.cuda.memory._dump_snapshot(
            f"{cfg.model.context_length}_forward_backward_memory_snapshot.pickle"
        )
        torch.cuda.memory._record_memory_history(enabled=None)
        torch.cuda.reset_peak_memory_stats()

    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    avg_time_ms = (end - start) / cfg.benchmark.run_iters * 1000
    return avg_time_ms, peak_memory_gb


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Config):
    # 1. Initialize WandB
    git_info = get_git_info()
    system_info = get_system_info()
    init_wandb(cfg, git_info, system_info)

    log.info(f"Starting benchmark: {cfg.experiment_name}")
    log.info(f"Git: {git_info['hash_short']} ({git_info['branch']})")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    try:
        # 2. Initialize Model & Data
        with torch.cuda.nvtx.range("init_model"):
            model = init_model(cfg)
        model_params = model.get_num_params()
        log.info(f"Model parameters: {model_params / 1e6:.2f}M")

        with torch.cuda.nvtx.range("generate_data"):
            x, y = generate_random_data(cfg)

        # 3. Run Benchmark
        if cfg.benchmark.mode == "forward_only":
            avg_time_ms, peak_memory_gb = benchmark_forward(model, x, cfg)
        elif cfg.benchmark.mode == "forward_backward":
            avg_time_ms, peak_memory_gb = benchmark_forward_backward(model, x, y, cfg)
        else:
            raise ValueError(f"Unknown benchmark mode: {cfg.benchmark.mode}")

        # 4. Log Results
        throughput = cfg.benchmark.batch_size / (avg_time_ms / 1000)

        results = {
            "avg_time_ms": avg_time_ms,
            "throughput_samples_per_sec": throughput,
            "model_params_millions": model_params / 1e6,
            "peak_memory_gb": peak_memory_gb,
            "status": "success",
            # Log key config parameters as metrics for easier visualization in WandB
            "context_length": cfg.model.context_length,
            "batch_size": cfg.benchmark.batch_size,
            "vocab_size": cfg.model.vocab_size,
            "d_model": cfg.model.d_model,
            "num_layers": cfg.model.num_layers,
            "compile": cfg.benchmark.compile,
            "compile_mode": cfg.benchmark.compile_mode,
        }

        wandb.log(results)
        log.info(f"Benchmark finished. Results: {results}")

        # Save to central CSV
        log_to_csv(cfg, results, git_info)

    except torch.cuda.OutOfMemoryError:
        log.error("OOM: Out of Memory Error!")
        wandb.log({"status": "oom"})

        # Log OOM to CSV
        results = {
            "avg_time_ms": float("nan"),
            "throughput_samples_per_sec": float("nan"),
            "model_params_millions": float("nan"),
            "peak_memory_gb": float("nan"),
            "status": "oom",
        }
        # Try to get model params if model was initialized
        if "model" in locals():
            results["model_params_millions"] = model.get_num_params() / 1e6

        log_to_csv(cfg, results, git_info)

    except Exception as e:
        log.error(f"An error occurred: {e}")
        wandb.log({"status": "failed", "error": str(e)})
        raise e

    finally:
        wandb.finish()

        # Cleanup memory
        if "model" in locals():
            del model
        if "x" in locals():
            del x
        if "y" in locals():
            del y
        if "optimizer" in locals():
            del optimizer

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
