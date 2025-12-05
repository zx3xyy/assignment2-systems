import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM

log = logging.getLogger(__name__)


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
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def sync_device(device: str):
    """Synchronize CUDA device if applicable."""
    if "cuda" in device:
        torch.cuda.synchronize()


def generate_random_data(cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
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


def init_model(cfg: DictConfig) -> BasicsTransformerLM:
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
    model = model.to(cfg.benchmark.device)

    if cfg.benchmark.verbose:
        log.info(
            f"Model initialized with {model.get_num_params() / 1e6:.2f}M parameters"
        )

    return model


def benchmark_forward(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    cfg: DictConfig,
) -> float:
    """Benchmark forward pass only."""
    device = cfg.benchmark.device

    # Warmup
    if cfg.benchmark.verbose:
        log.info(f"[Forward] Running {cfg.benchmark.warmup_iters} warmup iterations...")
    for _ in range(cfg.benchmark.warmup_iters):
        with torch.no_grad():
            _ = model(x)
    sync_device(device)

    # Benchmark
    if cfg.benchmark.verbose:
        log.info(f"[Forward] Running {cfg.benchmark.run_iters} benchmark iterations...")

    sync_device(device)
    start = time.perf_counter()
    for _ in range(cfg.benchmark.run_iters):
        with torch.no_grad():
            _ = model(x)
    sync_device(device)
    end = time.perf_counter()

    avg_time_ms = (end - start) / cfg.benchmark.run_iters * 1000
    return avg_time_ms


def benchmark_forward_backward(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: DictConfig,
) -> float:
    """Benchmark forward + backward pass."""
    device = cfg.benchmark.device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Warmup
    if cfg.benchmark.verbose:
        log.info(
            f"[Forward+Backward] Running {cfg.benchmark.warmup_iters} warmup iterations..."
        )
    for _ in range(cfg.benchmark.warmup_iters):
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
    start = time.perf_counter()
    for _ in range(cfg.benchmark.run_iters):
        outputs = model(x)
        loss = criterion(outputs.view(-1, cfg.model.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    sync_device(device)
    end = time.perf_counter()

    avg_time_ms = (end - start) / cfg.benchmark.run_iters * 1000
    return avg_time_ms


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Initialize WandB
    git_info = get_git_info()
    system_info = get_system_info()

    # Convert OmegaConf to dict for wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["git"] = git_info
    config_dict["system"] = system_info

    # Get reproduce command
    reproduce_command = " ".join(sys.argv)
    config_dict["reproduce_command"] = reproduce_command

    run = wandb.init(
        project="cs336-systems-benchmark",
        name=cfg.experiment_name,
        config=config_dict,
        tags=[cfg.benchmark.mode, git_info["hash_short"]],
    )

    log.info(f"Starting benchmark: {cfg.experiment_name}")
    log.info(f"Git: {git_info['hash_short']} ({git_info['branch']})")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.benchmark.device != "cuda" and torch.cuda.is_available():
        log.warning("CUDA is available but device is set to CPU!")

    # 2. Initialize Model & Data
    model = init_model(cfg)
    model_params = model.get_num_params()
    log.info(f"Model parameters: {model_params / 1e6:.2f}M")

    x, y = generate_random_data(cfg)

    # 3. Run Benchmark
    if cfg.benchmark.mode == "forward_only":
        avg_time_ms = benchmark_forward(model, x, cfg)
    elif cfg.benchmark.mode == "forward_backward":
        avg_time_ms = benchmark_forward_backward(model, x, y, cfg)
    else:
        raise ValueError(f"Unknown benchmark mode: {cfg.benchmark.mode}")

    # 4. Log Results
    throughput = cfg.benchmark.batch_size / (avg_time_ms / 1000)

    results = {
        "avg_time_ms": avg_time_ms,
        "throughput_samples_per_sec": throughput,
        "model_params_millions": model_params / 1e6,
    }

    wandb.log(results)

    # Save to central CSV
    import csv

    original_cwd = hydra.utils.get_original_cwd()
    central_csv_path = Path(original_cwd) / "benchmark_results.csv"

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
        **results,
    }

    with open(central_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_results)

    log.info(f"Benchmark finished. Results: {results}")
    log.info(f"Results appended to {central_csv_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
