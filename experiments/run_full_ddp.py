#!/usr/bin/env python3
"""Run one full-split GCM cell with data parallelism."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import paper_a_grid as grid


FULL_TRAIN = {
    "hotpot_qa": 90447,
    "squad_v2": 86821,
    "narrativeqa": 32747,
    "bfcl_live_multiple": 737,
    "musr_mm": 160,
}
FULL_VAL = {
    "hotpot_qa": 7405,
    "squad_v2": 5928,
    "narrativeqa": 3461,
    "bfcl_live_multiple": 316,
    "musr_mm": 90,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-path",
        default="",
        help="local checkpoint path or Hugging Face model id; defaults to paper_a_grid.MODELS",
    )
    parser.add_argument("--bench", required=True)
    parser.add_argument("--stage", default="main")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--recipe", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument(
        "--devices",
        default="",
        help="comma-separated physical CUDA devices; overrides --gpus",
    )
    parser.add_argument("--suffix", default="fulltrain-ddp")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    args = parser.parse_args()

    if args.recipe:
        os.environ["GCM_RECIPE"] = args.recipe
    job = grid.Job(
        args.stage,
        args.model,
        args.bench,
        args.seed,
        "ours",
        args.variant,
    )
    _, job_env, _ = grid.command_for(job)
    job_env.pop("GCM_LOAD_ADAPTER", None)
    n_train = args.train_limit or FULL_TRAIN[args.bench]
    n_val = args.val_limit or FULL_VAL[args.bench]
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        devices = [str(i) for i in range(args.gpus)]
    world_size = len(devices)
    job_env.update(
        {
            "GCM_NTRAIN": str(n_train),
            "GCM_NVAL": str(n_val),
            "GCM_EPOCHS": "1",
            "GCM_MAX_STEPS": "0",
            "GCM_PATIENCE": "0",
            "GCM_ACCUM": "1",
            "GCM_BATCH": "1",
            "GCM_SEED": str(args.seed),
            "GCM_TAG": f"{job.tag}_{args.suffix}",
            "CUDA_VISIBLE_DEVICES": ",".join(devices),
        }
    )

    env = os.environ.copy()
    env.update(job_env)
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if value
    )

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        str(ROOT / "experiments" / "run_recipe.py"),
        args.model_path or os.environ.get("GCM_MODEL_PATH", "") or grid.MODELS[args.model],
    ]
    print(" ".join(cmd), flush=True)
    print(
        f"{args.model} {args.bench}: {n_train} train, "
        f"{n_val} validation, seed {args.seed}",
        flush=True,
    )
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
