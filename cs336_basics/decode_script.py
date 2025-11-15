from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
import torch
from torch import nn
import tiktoken
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from cs336_basics.transformer import Transformer
from cs336_basics.train import load_checkpoint
from config import Config

console = Console()

@torch.no_grad()
def decode(model: nn.Module, input_ids: torch.Tensor, max_new_tokens: int, temperature: float, p: float) -> torch.Tensor:
    model.eval()
    token_generated = 0
    while token_generated < max_new_tokens:
        logits = model(input_ids)[:, -1, :]
        logits = logits / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=1)

        if 0.0 < p < 1.0:
            sorted_probs, indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=1)
            trimmed = cumsum > p
            trimmed[:, 0] = False
            sorted_probs[trimmed] = 0
            next_token_id = torch.multinomial(sorted_probs, num_samples=1)
            next_token = indices.gather(1, next_token_id)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == 50256:  # gpt2 EOS
            break

        input_ids = torch.cat((input_ids, next_token), dim=1)
        token_generated += 1
    return input_ids

@dataclass
class Args:
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    p: float = 0.9
    ckpt_name: str = "dev_0099.pt"
    seed: Optional[int] = 42

@dataclass
class CLI:
    cfg: Config
    args: Args

def main() -> int:
    cli = tyro.cli(CLI)         # 解析成一个对象
    cfg, args = cli.cfg, cli.args

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = tiktoken.get_encoding("gpt2")
    model = Transformer(
        cfg.d_model, cfg.num_heads, cfg.d_ff, cfg.context_length,
        cfg.rope_theta, cfg.vocab_size, cfg.num_layers
    ).to(cfg.device)
    load_checkpoint(Path(cfg.ckpt_path) / args.ckpt_name, model, optimizer=None)

    # rich 打印 Config
    table = Table(title="Model Configuration", title_style="bold cyan")
    for k, v in vars(cfg).items():
        table.add_row(k, str(v))
    console.print(table)

    console.print(Panel.fit(f"[bold yellow]Prompt:[/bold yellow]\n{args.prompt}", border_style="green"))

    input_ids = torch.as_tensor(tokenizer.encode(args.prompt), dtype=torch.int64).unsqueeze(0).to(cfg.device)
    output_ids = decode(model, input_ids, args.max_new_tokens, args.temperature, args.p).squeeze().tolist()
    decoded = tokenizer.decode(output_ids)

    console.print(Panel(decoded, title="Model Output", title_align="left", border_style="blue"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
