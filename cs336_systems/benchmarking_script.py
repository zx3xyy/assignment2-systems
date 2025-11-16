import torch
import numpy as np
import random
from torch.amp import autocast 
from cs336_basics.transformer import Transformer
from cs336_basics.train import AdamW, get_batch, gradient_clipping, get_lr_schedule,CrossEntropyLossWithLogits, save_checkpoint
import wandb  
from cs336_basics.config import Config, MODEL_PRESETS
from pathlib import Path
import tyro
from rich.console import Console
from rich.table import Table
from dataclasses import asdict
import time
import traceback
import sys
from timeit import default_timer as timer

console = Console()
def print_config(cfg: Config) -> None:
    table = Table(title="[bold cyan]Experiment Configuration[/bold cyan]", show_lines=False)
    table.add_column("Field", style="bold yellow")
    table.add_column("Value", style="bold white")
    for k, v in asdict(cfg).items():
        table.add_row(k, str(v))
    console.print(table)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def process_config(cfg: Config):
    if cfg.device == "cuda"  and not torch.cuda.is_available():
        console.print("[red]CUDA not avaiable! fallback to cpu[/red]")
        cfg.device = "cpu"
    preset = MODEL_PRESETS.get(cfg.model_name)
    if preset is None:
        return
    for k, v in preset.items():
        setattr(cfg, k, v)
            
def benchmark_one_step(model, data, target, cfg, loss_module, optimizer):
    logits = model(data)
    if loss_module and optimizer:
        loss = loss_module(logits, target)
        loss.backward()
        optimizer.step()

        
def benchmark(model, data, target, cfg, loss, optimizer):
    # Warmup
    console.print("warming up...")
    for _ in range(cfg.n_iter_warmup):
        benchmark_one_step(model, data, target, cfg, loss, optimizer)
    
    torch.cuda.synchronize()  
    console.print("starting benchmark...")  
    start = timer()
    for _ in range(cfg.n_iter_benchmark):
        benchmark_one_step(model, data, target, cfg, loss, optimizer)
        torch.cuda.synchronize()
    end = timer()
    
    console.print(f"[green]✅ Benchmark Finished: {cfg.n_iter_benchmark / (end - start)} it/s [/green]")

def main():
    set_seed(42)
    torch.set_float32_matmul_precision('high')

    cfg: Config = tyro.cli(Config)
    process_config(cfg)
    print_config(cfg)
    model = Transformer(
            cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.context_length, cfg.rope_theta, cfg.vocab_size, cfg.num_layers
        ).to(cfg.device)
    seq = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=cfg.device)
    target = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=cfg.device)
    
    if cfg.benchmark:
        loss_module = CrossEntropyLossWithLogits() if cfg.benchmark_backward else None
        optimizer = AdamW(model.parameters()) if cfg.benchmark_backward else None
        benchmark(model, seq, target, cfg, loss_module, optimizer)

    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print("⚠️ Training crashed with exception:", repr(e))
        console.print("------ Traceback ------")
        traceback.print_exc()
        console.print("-----------------------")
        sys.exit(1)