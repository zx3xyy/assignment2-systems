import torch
import numpy as np
import random
from torch.amp import autocast 
from cs336_basics.transformer import Transformer
from cs336_basics.train import AdamW, get_batch, gradient_clipping, get_lr_schedule,CrossEntropyLossWithLogits, save_checkpoint
import wandb  
from config import Config
from pathlib import Path
import tyro
from rich.console import Console
from rich.table import Table
from dataclasses import asdict
import time
import traceback
import sys

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
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.use_deterministic_algorithms(True)
    # if torch.cuda.is_available():
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    torch.set_float32_matmul_precision('high')

    cfg: Config = tyro.cli(Config)
    print_config(cfg)

    exp_name = cfg.exp_name
    data = np.load(cfg.train_data_path, mmap_mode="r")
    val_data = np.load(cfg.valid_data_path, mmap_mode="r")

    console.print(f"[green]✅ Loaded train data shape:[/green] {data.shape}")
    console.print(f"[green]✅ Loaded valid data shape:[/green] {val_data.shape}")

    wandb.init(
        project="cs336-assignment1",
        name=exp_name,
        config=cfg, 
    )

    model = Transformer(
            cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.context_length, cfg.rope_theta, cfg.vocab_size, cfg.num_layers
        ).to(cfg.device)

    if cfg.torch_compile:
        console.print(f"[green]✅ Compiling Model...")
        start = time.time()
        model = torch.compile(model)
        console.print(f"[green]✅ Finished model compilation.[/green] Used {(time.time() - start):.2f} seconds")

    loss_module = CrossEntropyLossWithLogits()
    optimizer = AdamW(model.parameters())

    t = 0
    start = time.time()
    
    while t < cfg.max_iter:
        optimizer.zero_grad(set_to_none=True)
        seq, target = get_batch(x=data, batch_size=cfg.batch_size, context_length=cfg.context_length, device=cfg.device)
        with autocast(device_type=cfg.device, dtype=torch.bfloat16):
            logits = model(seq) # B, context_len, vocab_size

        target = target.long()
        loss = loss_module(logits, target)
        loss.backward()
        gradient_clipping(model.parameters(), cfg.grad_clip_cap)
        lr = get_lr_schedule(t, cfg.max_lr, cfg.min_lr, cfg.t_w, cfg.t_c)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        if t % cfg.eval_interval == 0:
            with torch.no_grad():
                seq, target = get_batch(x=val_data, batch_size=cfg.batch_size, context_length=cfg.context_length, device=cfg.device)
                logits = model(seq)
                target = target.long()
                eval_loss = loss_module(logits, target)
            elapsed = time.time() - start
            start = time.time()
            qps = (cfg.eval_interval * cfg.context_length * cfg.batch_size) / elapsed
            wandb.log({"eval_loss": eval_loss, "train_loss": loss, 'qps': qps}, step=t * cfg.context_length * cfg.batch_size)

        if cfg.ckpt_interval > 0 and t % cfg.ckpt_interval == 0:
            path = Path(cfg.ckpt_path)
            path = path / f"{exp_name}_{(t//cfg.ckpt_interval):04d}.pt"
            save_checkpoint(model, optimizer, t, path)
        t += 1
    wandb.finish()
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print("⚠️ Training crashed with exception:", repr(e))
        console.print("------ Traceback ------")
        traceback.print_exc()
        console.print("-----------------------")
        sys.exit(1)