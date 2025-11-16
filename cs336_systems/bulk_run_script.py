# local_submitit_launcher.py
from __future__ import annotations
from typing import List
from copy import deepcopy
import submitit
import time
from cs336_basics.config import Config, MODEL_PRESETS, process_config
from cs336_systems.benchmarking_script import benchmark
import torch
import io
import os
import json
import contextlib

def make_sweep() -> List[Config]:
    base = Config()
    model_names = ["small", "medium", "large", "xl", "2.7B"]
    configs: List[Config] = []
    for i, model_name in enumerate(model_names):
        cfg = deepcopy(base)
        cfg.model_name = model_name
        process_config(cfg)
        # cfg.n_iter_benchmark = 100
        cfg.exp_name = f"model_name:{model_name}_local_{i}"
        configs.append(cfg)
    return configs


def run_job(cfg: Config, idx: int):
    os.makedirs("local_logs", exist_ok=True)
    job_name = f"job_{idx}_{cfg.model_name}"
    out_path = f"local_logs/{job_name}.out"
    err_path = f"local_logs/{job_name}.err"
    cfg_path = f"local_logs/{job_name}.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    # Capture stdout/stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer), \
             contextlib.redirect_stderr(stderr_buffer):
            benchmark(cfg)

        success = True

    except Exception as e:
        print(f"[launcher] job {idx} raised: {e}")
        success = False

    # Write logs
    with open(out_path, "w") as f:
        f.write(stdout_buffer.getvalue())
    with open(err_path, "w") as f:
        f.write(stderr_buffer.getvalue())

    return success

def main() -> None:
    configs = make_sweep()
    for i, cfg in enumerate(configs):
        run_job(cfg, i)
        
if __name__ == "__main__":
    main()