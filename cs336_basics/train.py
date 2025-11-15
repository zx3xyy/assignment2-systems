import torch
from torch import nn
from jaxtyping import Float, Int
from torch import Tensor
from collections.abc import Callable, Iterable
from typing import Optional, Tuple
import math
from typing import IO, Any, BinaryIO
import os
import numpy as np

class CrossEntropyLossWithLogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: Float[Tensor, "... d"], labels: Int[Tensor, "..."]):
        # Optional for Numerical Stability
        max_val, _idx = torch.max(logits, dim=-1, keepdim=True)  # [..., 1]
        logits = logits - max_val

        exp_logits = torch.exp(logits)  # [... d]
        exp_logits_sum = torch.sum(exp_logits, dim=-1, keepdim=True)  # [... 1]
        loss = -torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)) + torch.log(
            exp_logits_sum
        )
        return torch.mean(loss)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        defaults = {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)
                lr = group["lr"]
                grad = p.grad.data
                # Moving average of exp_avg and exp_avg_sq
                m = group["beta_1"] * m + (1 - group["beta_1"]) * grad
                v = group["beta_2"] * v + (1 - group["beta_2"]) * grad**2

                lr *= math.sqrt(1 - group["beta_2"] ** t) / (1 - group["beta_1"] ** t)
                p.data = p.data - lr * m / (torch.sqrt(v) + group["eps"])
                p.data = p.data - p.data * group["lr"] * group["weight_decay"]
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
                self.state[p] = state

        return loss


def get_lr_schedule(t, max_lr, min_lr, t_w, t_c):
    if t < t_w:
        return t / t_w * max_lr
    elif t > t_c:
        return min_lr
    return min_lr + 0.5 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (
        max_lr - min_lr
    )


def gradient_clipping(params, cap, eps=1e-6):
    grad_square_sum = 0
    for param in params:
        if param.grad is None:
            continue
        grad_square_sum += torch.norm(param.grad.data) ** 2

    grad_l2_norm = math.sqrt(grad_square_sum)
    if grad_l2_norm > cap:
        scaling_factor = cap / (grad_l2_norm + eps)
        for param in params:
            if param.grad is None:
                continue
            param.grad.data = param.grad.data * scaling_factor


def get_batch(
    x: torch.Tensor, batch_size: int, context_length: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq = []
    target = []

    lower_bound = 1
    upper_bound = len(x) - context_length
    import random

    for _ in range(batch_size):
        first_target = random.randint(lower_bound, upper_bound)
        seq.append(
            torch.as_tensor(np.copy(x[first_target - 1 : first_target + context_length - 1]), dtype=torch.int64)
        )
        target.append(torch.as_tensor(np.copy(x[first_target : first_target + context_length]), dtype=torch.int64))

    return torch.stack(seq, dim=0).to(device), torch.stack(target, dim=0).to(device)

def save_checkpoint(    
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    ckpt = {}
    ckpt["model"] = model.state_dict()
    ckpt["optimizer"] = optimizer.state_dict()
    ckpt["iteration"] = iteration
    torch.save(ckpt, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]