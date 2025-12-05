import argparse
from dataclasses import dataclass, field

import numpy as np
import torch
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
import time


@dataclass
class ModelConfig:
    """Configuration for BasicsTransformerLM model."""

    vocab_size: int = 10000
    context_length: int = 128
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    warmup_iters: int = 10
    run_iters: int = 100
    verbose: bool = False
    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark_forward_only: bool = True
    benchmark_forward_backward: bool = False


@dataclass
class Config:
    """Combined configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


def parse_args() -> Config:
    """Parse command line arguments and return Config."""
    parser = argparse.ArgumentParser(
        description="Benchmarking script for BasicsTransformerLM"
    )

    # Model config arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--vocab-size", type=int, default=10000, help="Vocabulary size"
    )
    model_group.add_argument(
        "--context-length", type=int, default=128, help="Context length"
    )
    model_group.add_argument("--d-model", type=int, default=768, help="Model dimension")
    model_group.add_argument(
        "--num-layers", type=int, default=12, help="Number of transformer layers"
    )
    model_group.add_argument(
        "--num-heads", type=int, default=12, help="Number of attention heads"
    )
    model_group.add_argument(
        "--d-ff", type=int, default=3072, help="Feed-forward dimension"
    )
    model_group.add_argument(
        "--rope-theta", type=float, default=10000.0, help="RoPE theta value"
    )

    # Benchmark config arguments
    bench_group = parser.add_argument_group("Benchmark Configuration")
    bench_group.add_argument(
        "--warmup-iters", type=int, default=10, help="Number of warmup iterations"
    )
    bench_group.add_argument(
        "--run-iters", type=int, default=100, help="Number of benchmark iterations"
    )
    bench_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )
    bench_group.add_argument("--batch-size", type=int, default=4, help="Batch size")
    bench_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    # Mutually exclusive benchmark mode
    bench_mode_group = parser.add_mutually_exclusive_group(required=True)
    bench_mode_group.add_argument(
        "--forward-only",
        action="store_true",
        dest="benchmark_forward_only",
        help="Benchmark forward pass only",
    )
    bench_mode_group.add_argument(
        "--forward-backward",
        action="store_true",
        dest="benchmark_forward_backward",
        help="Benchmark forward + backward pass (full training step)",
    )

    args = parser.parse_args()

    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    benchmark_config = BenchmarkConfig(
        warmup_iters=args.warmup_iters,
        run_iters=args.run_iters,
        verbose=args.verbose,
        batch_size=args.batch_size,
        device=args.device,
        benchmark_forward_only=args.benchmark_forward_only,
        benchmark_forward_backward=args.benchmark_forward_backward,
    )

    return Config(model=model_config, benchmark=benchmark_config)


def generate_random_data(config: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a random dataset of token IDs."""
    dataset_size = 100000
    dataset = np.random.randint(
        0, config.model.vocab_size, size=(dataset_size,), dtype=np.int64
    )

    x, y = get_batch(
        dataset,
        config.benchmark.batch_size,
        config.model.context_length,
        config.benchmark.device,
    )

    if config.benchmark.verbose:
        print(f"Input batch shape: {x.shape}")
        print(f"Target batch shape: {y.shape}")
        print(f"Device: {x.device}")

    return x, y


def init_model(config: Config) -> BasicsTransformerLM:
    """Initialize the transformer model."""
    model = BasicsTransformerLM(
        vocab_size=config.model.vocab_size,
        context_length=config.model.context_length,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        rope_theta=config.model.rope_theta,
    )
    model = model.to(config.benchmark.device)

    if config.benchmark.verbose:
        print(f"Model initialized with {model.get_num_params() / 1e6:.2f}M parameters")

    return model


def sync_device(device: str):
    """Synchronize CUDA device if applicable."""
    if "cuda" in device:
        torch.cuda.synchronize()


def benchmark_forward(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    config: Config,
) -> float:
    """Benchmark forward pass only.

    Returns:
        Average time per iteration in milliseconds.
    """
    device = config.benchmark.device

    # Warmup
    if config.benchmark.verbose:
        print(f"[Forward] Running {config.benchmark.warmup_iters} warmup iterations...")
    for _ in range(config.benchmark.warmup_iters):
        with torch.no_grad():
            _ = model(x)
    sync_device(device)

    # Benchmark
    if config.benchmark.verbose:
        print(f"[Forward] Running {config.benchmark.run_iters} benchmark iterations...")

    sync_device(device)
    start = time.perf_counter()
    for _ in range(config.benchmark.run_iters):
        with torch.no_grad():
            _ = model(x)
    sync_device(device)
    end = time.perf_counter()

    avg_time_ms = (end - start) / config.benchmark.run_iters * 1000
    return avg_time_ms


def benchmark_forward_backward(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    y: torch.Tensor,
    config: Config,
) -> float:
    """Benchmark forward + backward pass (full training step).

    Returns:
        Average time per iteration in milliseconds.
    """
    device = config.benchmark.device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Warmup
    if config.benchmark.verbose:
        print(
            f"[Forward+Backward] Running {config.benchmark.warmup_iters} warmup iterations..."
        )
    for _ in range(config.benchmark.warmup_iters):
        outputs = model(x)
        loss = criterion(outputs.view(-1, config.model.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    sync_device(device)

    # Benchmark
    if config.benchmark.verbose:
        print(
            f"[Forward+Backward] Running {config.benchmark.run_iters} benchmark iterations..."
        )

    sync_device(device)
    start = time.perf_counter()
    for _ in range(config.benchmark.run_iters):
        outputs = model(x)
        loss = criterion(outputs.view(-1, config.model.vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    sync_device(device)
    end = time.perf_counter()

    avg_time_ms = (end - start) / config.benchmark.run_iters * 1000
    return avg_time_ms


def main():
    config = parse_args()
    if config.benchmark.device != "cuda":
        raise ValueError("Benchmarking script requires CUDA device.")

    if config.benchmark.verbose:
        print(f"Config: {config}")

    # Initialize model and data
    model = init_model(config)
    x, y = generate_random_data(config)

    # Run benchmark based on mode
    if config.benchmark.benchmark_forward_only:
        forward_time = benchmark_forward(model, x, config)
        print(f"Average forward pass time: {forward_time:.3f} ms")
    elif config.benchmark.benchmark_forward_backward:
        forward_backward_time = benchmark_forward_backward(model, x, y, config)
        print(f"Average forward+backward pass time: {forward_backward_time:.3f} ms")


if __name__ == "__main__":
    main()
