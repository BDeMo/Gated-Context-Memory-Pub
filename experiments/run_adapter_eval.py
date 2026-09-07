#!/usr/bin/env python3
"""Evaluate a frozen GCM adapter on one target with the target's paper protocol."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import paper_a_grid as grid

FULL_VAL = {
    "quality": 2086,
    "bfcl_live_multiple": 316,
    "squad_v2": 5928,
    "hotpot_qa": 7405,
    "narrativeqa": 3461,
    "musr_mm": 90,
}


def target_job(model: str, target: str, seed: int) -> grid.Job:
    if target in grid.LONG_TRANSFER:
        return grid.Job(
            "longcontext",
            model,
            target,
            seed,
            "ours",
            f"from-{grid.LONG_TRANSFER[target]}",
        )
    if target == "locomo":
        return grid.Job(
            "agent_memory_corrected",
            model,
            target,
            seed,
            "ours",
            "from-quality-corrected",
        )
    return grid.Job("main", model, target, seed, "ours")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="q3_8b")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nval", type=int, default=0)
    args = parser.parse_args()

    if args.target not in grid.BENCH:
        parser.error(f"unknown target: {args.target}")
    if not args.adapter.is_file():
        parser.error(f"adapter does not exist: {args.adapter}")

    job = target_job(args.model, args.target, args.seed)
    env_overrides = grid._common_env(job) | grid._ours_env(job)
    tag = args.tag or (
        f"crossall_{args.model}_from-{args.source_label}_to-{args.target}_s{args.seed}"
    )
    nval = args.nval or FULL_VAL.get(args.target, 100_000)
    env_overrides.update(
        {
            "GCM_TAG": tag,
            "GCM_LOAD_ADAPTER": str(args.adapter),
            "GCM_NTRAIN": "1",
            "GCM_NVAL": str(nval),
            "GCM_EVAL": args.target,
            "GCM_SEED": str(args.seed),
            "CUDA_VISIBLE_DEVICES": args.device,
        }
    )

    environment = os.environ.copy()
    environment.update(env_overrides)
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT / "src"), environment.get("PYTHONPATH", ""))
        if value
    )
    model_path = (
        args.model_path
        or os.environ.get("GCM_MODEL_PATH", "")
        or grid.MODELS[args.model]
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        str(ROOT / "experiments" / "run_recipe.py"),
        model_path,
    ]
    print(" ".join(command), flush=True)
    print(
        f"tag={tag} source={args.source_label} target={args.target} "
        f"adapter={args.adapter} device={args.device}",
        flush=True,
    )
    return subprocess.call(command, cwd=ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
