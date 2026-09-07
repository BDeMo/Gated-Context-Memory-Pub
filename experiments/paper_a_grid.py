"""Resumable Paper-A experiment grid.

The grid implements the selected paper package:

* fair three-seed main comparison on two primary models/four tasks;
* fixed-K cross-model reproduction;
* five load-bearing mechanism ablations.

Workers use deterministic modulo sharding, one process per GPU. Every job writes
an isolated log, result copy, adapter copy, and status JSON below
``results/runs``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
PAPER_ROOT = Path(os.environ.get("GCM_PAPER_ROOT", ROOT / "results" / "runs"))
BASELINE_PROTOCOL_REVISION = "source-and-budget-v2"
LONG_TARGET_CONTROL_PROTOCOL_REVISION = "target-source-and-realized-reader-v1"
STATE_ABLATION_PROTOCOL_REVISION = "recurrent-state-cap-v1"
AGENT_MEMORY_PROTOCOL_REVISION = "locomo-corrected-transfer-v1"

MODELS = {
    "q3_8b": "Qwen/Qwen3-8B",
    "q35_9b": "Qwen/Qwen3.5-9B",
    "q35_4b": "Qwen/Qwen3.5-4B",
    "ministral_8b": "mistralai/Ministral-8B-Instruct-2410",
    "glm4_9b": "zai-org/GLM-4-9B-0414",
    "xlam_8b": "Salesforce/Llama-xLAM-2-8b-fc-r",
    "toolace_8b": "Team-ACE/ToolACE-2-8B",
}

ALL_BASES = (
    "q3_8b",
    "q35_9b",
    "q35_4b",
    "ministral_8b",
    "glm4_9b",
    "xlam_8b",
    "toolace_8b",
)

HARNESS_BASELINES_Q3 = (
    "icae",
    "aoc",
    "beacon",
    "x500",
    "cartridge",
    "comprexit",
    "lcc",
    "meanpool",
)
HARNESS_BASELINES_LINEAR = (
    "icae",
    "aoc",
    "comprexit",
    "lcc",
    "meanpool",
)

BENCH = {
    "quality": {"maxctx": 8192, "enc": 16384, "ntrain": 2000, "gen": 8, "nchunks": 8},
    "bfcl_live_multiple": {"maxctx": 4096, "enc": 4096, "ntrain": 1000, "gen": 32, "nchunks": 8},
    "squad_v2": {"maxctx": 4096, "enc": 4096, "ntrain": 2000, "gen": 16, "nchunks": 8},
    "hotpot_qa": {"maxctx": 4096, "enc": 4096, "ntrain": 2000, "gen": 16, "nchunks": 8},
    "narrativeqa": {"maxctx": 8192, "enc": 8192, "ntrain": 2000, "gen": 64, "nchunks": 8},
    "musr_mm": {"maxctx": 4096, "enc": 4096, "ntrain": 160, "gen": 8, "nchunks": 8},
    "ruler_niah": {"maxctx": 16384, "enc": 16384, "ntrain": 500, "gen": 16, "nchunks": 88},
    "longbench_v2": {"maxctx": 16384, "enc": 32768, "ntrain": 0, "gen": 8, "nchunks": 12},
    "infbench_choice": {"maxctx": 16384, "enc": 131072, "ntrain": 0, "gen": 8, "nchunks": 16},
    "locomo": {"maxctx": 32768, "enc": 32768, "ntrain": 0, "gen": 64, "nchunks": 40},
    "lb_multifieldqa": {"maxctx": 16384, "enc": 32768, "ntrain": 0, "gen": 32, "nchunks": 12},
    "lb_qasper": {"maxctx": 16384, "enc": 32768, "ntrain": 0, "gen": 32, "nchunks": 12},
    "lb_hotpotqa": {"maxctx": 16384, "enc": 32768, "ntrain": 0, "gen": 32, "nchunks": 12},
    "lb_2wikimqa": {"maxctx": 16384, "enc": 32768, "ntrain": 0, "gen": 32, "nchunks": 12},
    "lb_musique": {"maxctx": 16384, "enc": 32768, "ntrain": 0, "gen": 32, "nchunks": 12},
    "lb_narrativeqa": {"maxctx": 16384, "enc": 32768, "ntrain": 0, "gen": 64, "nchunks": 12},
    "babilong_qa1_16k": {"maxctx": 16384, "enc": 16384, "ntrain": 0, "gen": 16, "nchunks": 88},
    "babilong_qa2_16k": {"maxctx": 16384, "enc": 16384, "ntrain": 0, "gen": 16, "nchunks": 88},
    "babilong_qa3_16k": {"maxctx": 16384, "enc": 16384, "ntrain": 0, "gen": 16, "nchunks": 88},
}

LONG_TRANSFER = {
    "longbench_v2": "quality",
    "infbench_choice": "quality",
    "lb_multifieldqa": "squad_v2",
    "lb_qasper": "squad_v2",
    "lb_hotpotqa": "hotpot_qa",
    "lb_2wikimqa": "hotpot_qa",
    "lb_musique": "hotpot_qa",
    "lb_narrativeqa": "narrativeqa",
    "babilong_qa1_16k": "hotpot_qa",
    "babilong_qa2_16k": "hotpot_qa",
    "babilong_qa3_16k": "hotpot_qa",
}

LONG_MANUSCRIPT_TARGETS = (
    "lb_multifieldqa",
    "lb_qasper",
    "lb_hotpotqa",
    "lb_2wikimqa",
    "lb_musique",
    "lb_narrativeqa",
    "longbench_v2",
    "infbench_choice",
)

SEEDS = (42, 43, 44)


@dataclasses.dataclass(frozen=True)
class Job:
    stage: str
    model: str
    bench: str
    seed: int
    method: str
    variant: str = "default"

    @property
    def tag(self) -> str:
        return f"pa_{self.stage}_{self.model}_{self.bench}_{self.method}_{self.variant}_s{self.seed}"


def build_jobs(stage: str) -> list[Job]:
    jobs: list[Job] = []
    if stage in ("main", "all"):
        for model in ("q3_8b", "q35_9b"):
            for bench in ("quality", "bfcl_live_multiple", "squad_v2", "hotpot_qa"):
                for seed in SEEDS:
                    jobs.extend(
                        Job("main", model, bench, seed, method)
                        for method in ("ours", "sft")
                    )
                jobs.extend(
                    Job("main", model, bench, 42, method, "matched-memory")
                    for method in ("window", "ll2", "longllm", "llorig")
                )
                if bench == "quality":
                    jobs.append(Job("main", model, bench, 42, "rawfull", "truefull"))
                    for seed in SEEDS:
                        jobs.append(Job("main", model, bench, seed, "sft", "truefull"))
    if stage in ("quality_corrected", "all"):
        # Canonical corrected QuALITY replacement. LongLL variants are omitted:
        # their current cache interface silently falls back and is not reportable.
        for model in ("q3_8b", "q35_9b"):
            for seed in SEEDS:
                jobs.append(Job("quality_corrected", model, "quality", seed, "ours", "corrected"))
                jobs.append(Job("quality_corrected", model, "quality", seed, "sft", "corrected"))
                jobs.append(Job("quality_corrected", model, "quality", seed, "sft", "truefull-corrected"))
            jobs.append(Job("quality_corrected", model, "quality", 42, "window", "corrected"))
            jobs.append(Job("quality_corrected", model, "quality", 42, "ll2", "corrected"))
            jobs.append(Job("quality_corrected", model, "quality", 42, "rawfull", "truefull-corrected"))
    if stage in ("main_baselines_corrected", "all"):
        # Additional directly comparable main-table controls. Isolated tags
        # prevent legacy compression cells from being treated as done.
        for model in ("q3_8b", "q35_9b"):
            for bench in ("quality", "bfcl_live_multiple", "hotpot_qa"):
                for method in ("ll2", "longllm", "llorig", "rag", "meanpool"):
                    jobs.append(
                        Job(
                            "main_baselines_corrected",
                            model,
                            bench,
                            42,
                            method,
                            "corrected",
                        )
                    )
    if stage in ("narrativeqa_baselines", "all"):
        # NarrativeQA is the only cell where routing gains accuracy and saves
        # reader state at once, and it has no retrieval control on any base.
        # Isolated tags keep these separate from the main-table baseline cells.
        for model in ("q3_8b", "ministral_8b"):
            for method in ("rag", "window", "ll2", "meanpool"):
                jobs.append(
                    Job("narrativeqa_baselines", model, "narrativeqa", 42, method, "nqa")
                )
            jobs.append(
                Job(
                    "narrativeqa_baselines",
                    model,
                    "narrativeqa",
                    42,
                    "rawfull",
                    "truefull-nqa",
                )
            )
    if stage in ("generality", "all"):
        for model in ALL_BASES:
            for bench in ("quality", "bfcl_live_multiple"):
                for seed in SEEDS:
                    jobs.append(Job("generality", model, bench, seed, "ours"))
    if stage in ("ablation", "all"):
        variants = ("joint0", "distill0", "recon0", "recur0", "k64", "k256")
        for bench in ("quality", "bfcl_live_multiple"):
            for seed in SEEDS:
                jobs.extend(
                    Job("ablation", "q3_8b", bench, seed, "ours", variant)
                    for variant in variants
                )
    if stage in ("adversarial_ablation", "all"):
        # Compare against the no-adversarial main cell. Both variants use the
        # same discriminator, weight, layer, data, and optimizer. These are
        # explicitly per-item schedules: alternating lets G see the new D,
        # while simultaneous computes both losses from the same D snapshot.
        for bench in ("bfcl_live_multiple", "hotpot_qa"):
            for seed in SEEDS:
                jobs.append(
                    Job("adversarial_ablation", "q3_8b", bench, seed, "ours", "adv-alt")
                )
                jobs.append(
                    Job("adversarial_ablation", "q3_8b", bench, seed, "ours", "adv-sim")
                )
    if stage in ("state_ablation", "all"):
        # Claim-closing comparison: identical recurrent chunk encoder, changing
        # only whether its reader-visible state grows as S*K or is capped at K.
        for bench in ("quality", "bfcl_live_multiple"):
            for seed in SEEDS:
                jobs.append(
                    Job("state_ablation", "q3_8b", bench, seed, "ours", "state-concat")
                )
                jobs.append(
                    Job("state_ablation", "q3_8b", bench, seed, "ours", "state-k128")
                )
    if stage in ("budget", "all"):
        for bench in ("quality",):
            for memory in (64, 128, 256, 512):
                jobs.append(Job("budget", "q3_8b", bench, 42, "ours", f"k{memory}"))
            for raw_budget in (256, 512, 1024, 2048, 4096, 8192):
                jobs.append(Job("budget", "q3_8b", bench, 42, "window", f"w{raw_budget}"))
        # RULER is an optional stress test, not a paper-critical benchmark.
        # Its long-generation path exceeds a 96 GiB device in the current
        # implementation, so only schedule it when explicitly requested.
        if os.environ.get("GCM_INCLUDE_RULER") == "1":
            for memory in (64, 128, 256, 512):
                jobs.append(Job("budget", "q3_8b", "ruler_niah", 42, "ours", f"k{memory}"))
            for raw_budget in (256, 512, 1024, 2048, 4096, 8192):
                jobs.append(Job("budget", "q3_8b", "ruler_niah", 42, "window", f"w{raw_budget}"))
            for length in (4096, 8192, 32768):
                jobs.append(Job("budget", "q3_8b", "ruler_niah", 42, "ours", f"len{length}"))
    if stage in ("replicate", "all"):
        for seed in SEEDS:
            jobs.append(Job("replicate", "q3_8b", "quality", seed, "ours", "repeat1"))
    if stage in ("sft_reaudit", "all"):
        for seed in SEEDS:
            jobs.append(Job("sft_reaudit", "q3_8b", "quality", seed, "sft", "default"))
            jobs.append(Job("sft_reaudit", "q3_8b", "quality", seed, "sft", "truefull"))
    if stage in ("transfer_train", "all"):
        for model in ALL_BASES:
            for bench in ("quality", "squad_v2", "hotpot_qa", "narrativeqa"):
                jobs.append(Job("transfer_train", model, bench, 42, "ours"))
                if model in ("q3_8b", "q35_9b"):
                    jobs.append(Job("transfer_train", model, bench, 42, "sft"))
            jobs.append(Job("transfer_train", model, "quality", 42, "ours", "k32"))
    if stage in ("transfer_train_quality_corrected", "all"):
        # The original transfer_train tags are already marked done for adapters
        # trained with corrupted QuALITY labels. Use isolated tags so resumability
        # cannot silently skip the seven paper-critical replacements.
        for model in ALL_BASES:
            jobs.append(
                Job(
                    "transfer_train_quality_corrected",
                    model,
                    "quality",
                    42,
                    "ours",
                    "corrected",
                )
            )
    if stage in ("longcontext", "all"):
        for model in ALL_BASES:
            for bench in LONG_TRANSFER:
                jobs.append(Job("longcontext", model, bench, 42, "ours", f"from-{LONG_TRANSFER[bench]}"))
                if model in ("q3_8b", "q35_9b"):
                    jobs.append(Job("longcontext", model, bench, 42, "sft", f"from-{LONG_TRANSFER[bench]}"))
            jobs.append(Job("longcontext", model, "infbench_choice", 42, "ours", "from-quality-k32"))
    if stage in ("longcontext_quality_corrected", "all"):
        # Only these two targets consume the paper-critical QuALITY source
        # adapter. Their original done statuses point at invalid adapters.
        for model in ALL_BASES:
            for bench in ("longbench_v2", "infbench_choice"):
                jobs.append(
                    Job(
                        "longcontext_quality_corrected",
                        model,
                        bench,
                        42,
                        "ours",
                        "from-quality-corrected",
                    )
                )
    if stage in ("longcontext_controls_corrected", "all"):
        # Target-side, training-free controls for the manuscript LongBench
        # breakdown and the corrected QuALITY-dependent headline targets.
        # Each method sees the same bounded target source and receives the
        # exact reader budget produced by K=128 per 4,096-token GCM chunk.
        for model in ("q3_8b", "q35_9b"):
            for bench in LONG_MANUSCRIPT_TARGETS:
                for method in ("window", "rag", "ll2"):
                    jobs.append(
                        Job(
                            "longcontext_controls_corrected",
                            model,
                            bench,
                            42,
                            method,
                            "target-matched",
                        )
                    )
    if stage in ("agent_memory_corrected", "all"):
        # Real conversational-memory benchmark. Reuse the corrected QuALITY
        # adapter unchanged; target examples are evaluation-only. Controls are
        # training-free and match the realized S*128 reader budget.
        for model in ("q3_8b", "q35_9b"):
            jobs.append(
                Job(
                    "agent_memory_corrected",
                    model,
                    "locomo",
                    42,
                    "ours",
                    "from-quality-corrected",
                )
            )
            for method in ("window", "rag", "ll2", "rawfull"):
                jobs.append(
                    Job(
                        "agent_memory_corrected",
                        model,
                        "locomo",
                        42,
                        method,
                        "truefull" if method == "rawfull" else "target-matched",
                    )
                )
    if stage in ("longcontext_babilong_corrected", "all"):
        # Existing BABILong statuses were produced while examples advertised
        # SQuAD-v2 metadata. Isolate reruns so the official answer-accuracy
        # dispatch cannot be skipped by those already-done tags.
        for model in ALL_BASES:
            for bench in (
                "babilong_qa1_16k",
                "babilong_qa2_16k",
                "babilong_qa3_16k",
            ):
                jobs.append(
                    Job(
                        "longcontext_babilong_corrected",
                        model,
                        bench,
                        42,
                        "ours",
                        "from-hotpot_qa-metric-corrected",
                    )
                )
                if model in ("q3_8b", "q35_9b"):
                    jobs.append(
                        Job(
                            "longcontext_babilong_corrected",
                            model,
                            bench,
                            42,
                            "sft",
                            "from-hotpot_qa-metric-corrected",
                        )
                    )
    if stage in ("compression_baselines", "all"):
        for bench in ("quality", "bfcl_live_multiple", "squad_v2", "hotpot_qa"):
            for method in HARNESS_BASELINES_Q3:
                jobs.append(Job("compression_baselines", "q3_8b", bench, 42, method))
            for method in HARNESS_BASELINES_LINEAR:
                jobs.append(Job("compression_baselines", "q35_9b", bench, 42, method))
    # De-duplicate jobs shared by stages only if future variants add overlap.
    return list(dict.fromkeys(jobs))


def _common_env(job: Job) -> dict[str, str]:
    cfg = BENCH[job.bench]
    is_longcontext = job.stage in (
        "longcontext",
        "longcontext_quality_corrected",
        "longcontext_babilong_corrected",
    )
    is_agent_memory = job.stage == "agent_memory_corrected"
    is_long_target = (
        is_longcontext
        or is_agent_memory
        or job.stage == "longcontext_controls_corrected"
    )
    train_bench = (
        "quality"
        if is_agent_memory and job.method == "ours"
        else LONG_TRANSFER[job.bench]
        if is_longcontext
        else job.bench
    )
    env = {
        "GCM_TAG": job.tag,
        "GCM_TRAIN": train_bench,
        "GCM_EVAL": job.bench,
        "GCM_SEED": str(job.seed),
        "GCM_NTRAIN": str(BENCH[train_bench]["ntrain"]),
        "GCM_NVAL": "100000",
        "GCM_NCHUNKS": str(cfg["nchunks"]),
        "GCM_MAXCTX": str(cfg["maxctx"]),
        "GCM_ENC_MAXCTX": str(cfg["enc"]),
        "GCM_SOURCE_MAXCTX": str(cfg["enc"]),
        "GCM_GOLD_MAX": "32",
        "GCM_GEN_MAX": str(cfg["gen"]),
        "GCM_GEN_BS": "16",
        "GCM_MAX_STEPS": "2000",
        "GCM_PATIENCE": "0",
        "GCM_BATCH": "1",
        "GCM_ACCUM": "8",
        "GCM_LR": "3e-4",
        "GCM_GOLD_FALLBACK": "0",
        "GCM_GEN_NOFALLBACK": "1",
        "GCM_COMPRESS_FALLBACK": "0",
        "GCM_STRICT_EVAL": "1",
        "GCM_GATE_SIGNALS": "conf,targ,margin",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "offline",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if job.model.startswith("q35_"):
        env.update({
            "GCM_GRAD_CKPT": "1",
            "GCM_GEN_BS": "2",
        })
    if is_long_target:
        env["GCM_GEN_BS"] = "1"
    if job.stage in ("transfer_train", "transfer_train_quality_corrected"):
        # This stage exists to create one reusable source adapter. Its source-task
        # score is not a paper cell; use one validation item and save the adapter.
        env["GCM_NVAL"] = "1"
    if job.bench == "ruler_niah" and job.variant.startswith("len"):
        length = int(job.variant[3:])
        env.update({
            "GCM_NCHUNKS": str(round(length / 186)),
            "GCM_MAXCTX": str(length),
            "GCM_ENC_MAXCTX": str(length),
        })
    return env


def _ours_env(job: Job) -> dict[str, str]:
    env = {
        "GCM_K": "128",
        "GCM_DEPTH": "half",
        "GCM_LORA": "64",
        "GCM_PROJ": "2",
        "GCM_NORM": "hard",
        "GCM_DISTILL": "0.5",
        "GCM_RECON": "0.5",
        "GCM_RECON_MAXCTX": "512",
        "GCM_JOINT": "1",
        "GCM_CHUNK": "4096",
        "GCM_RECUR": "1",
        "GCM_STATE_CAP": "0",
        "GCM_ENC_VARLEN": "1",
    }
    if job.variant == "joint0":
        env["GCM_JOINT"] = "0"
    elif job.variant == "distill0":
        env["GCM_DISTILL"] = "0"
    elif job.variant == "recon0":
        env["GCM_RECON"] = "0"
    elif job.variant == "recur0":
        env["GCM_RECUR"] = "0"
    elif job.variant == "k64":
        env["GCM_K"] = "64"
    elif job.variant == "k256":
        env["GCM_K"] = "256"
    elif job.variant == "adv-alt":
        env["GCM_ADV"] = "0.1"
        env["GCM_ADV_MODE"] = "alternating"
    elif job.variant == "adv-sim":
        env["GCM_ADV"] = "0.1"
        env["GCM_ADV_MODE"] = "simultaneous"
    elif job.variant == "k512":
        env["GCM_K"] = "512"
    elif job.variant == "state-concat":
        env["GCM_STATE_CAP"] = "0"
    elif job.variant == "state-k128":
        env["GCM_STATE_CAP"] = "128"
    elif job.variant.endswith("k32"):
        env["GCM_K"] = "32"
    return env


def _completed_training_adapter(job: Job) -> Path | None:
    """Return an adapter saved after training but before a failed/unfinished full-split evaluation."""
    model_name = Path(MODELS[job.model]).name
    candidates = (
        PAPER_ROOT / "artifacts" / job.tag / f"{job.tag}_adapters.pt",
        ROOT / "out" / model_name / f"{job.tag}_adapters.pt",
    )
    return next((path for path in candidates if path.exists()), None)


def _transfer_adapter(job: Job) -> Path:
    source_bench = "quality" if job.stage == "agent_memory_corrected" else LONG_TRANSFER[job.bench]
    if job.stage in ("longcontext_quality_corrected", "agent_memory_corrected"):
        source_job = Job(
            "transfer_train_quality_corrected",
            job.model,
            source_bench,
            42,
            job.method,
            "corrected",
        )
        model_name = Path(MODELS[job.model]).name
        live_path = ROOT / "out" / model_name / f"{source_job.tag}_adapters.pt"
        artifact_path = (
            PAPER_ROOT
            / "artifacts"
            / source_job.tag
            / f"{source_job.tag}_adapters.pt"
        )
        return artifact_path if artifact_path.exists() else live_path
    source_variant = "k32" if job.variant.endswith("k32") else "default"
    source_job = Job(
        "transfer_train",
        job.model,
        source_bench,
        42,
        job.method,
        source_variant,
    )
    model_name = Path(MODELS[job.model]).name
    live_path = ROOT / "out" / model_name / f"{source_job.tag}_adapters.pt"
    artifact_path = (
        PAPER_ROOT
        / "artifacts"
        / source_job.tag
        / f"{source_job.tag}_adapters.pt"
    )
    return artifact_path if artifact_path.exists() else live_path


def command_for(job: Job) -> tuple[list[str], dict[str, str], Path | None]:
    model_path = MODELS[job.model]
    env = _common_env(job)
    if job.method == "ours":
        env.update(_ours_env(job))
        if job.stage in (
            "longcontext",
            "longcontext_quality_corrected",
            "longcontext_babilong_corrected",
            "agent_memory_corrected",
        ):
            env["GCM_LOAD_ADAPTER"] = str(_transfer_adapter(job))
        elif adapter := _completed_training_adapter(job):
            env["GCM_LOAD_ADAPTER"] = str(adapter)
        return [str(PYTHON), str(ROOT / "experiments/run_recipe.py"), model_path], env, None
    if job.method in ("sft", "window", "ll2", "longllm", "llorig", "rag", "rawfull"):
        mode = {
            "sft": "sft",
            "window": "txl",
            "ll2": "llmlingua",
            "longllm": "longllmlingua",
            "llorig": "llmlingua_orig",
            "rag": "rag",
            "rawfull": "txl",
        }[job.method]
        if job.variant.startswith("truefull"):
            env["GCM_MAXCTX"] = str(BENCH[job.bench]["enc"])
        matched_tokens = 256 if job.bench == "quality" else 128
        env.update(
            {
                "GCM_BASELINE": mode,
                "GCM_K": "128",
                "GCM_LORA": "64",
                "GCM_LL_RATE": str(matched_tokens / BENCH[job.bench]["maxctx"]),
                "GCM_LL_TARGET": str(matched_tokens),
                "GCM_TXL_WINDOW": (
                    job.variant[1:] if job.variant.startswith("w")
                    else str(BENCH[job.bench]["enc"]) if job.method == "rawfull"
                    else str(matched_tokens)
                ),
            }
        )
        if job.method == "rag":
            env["GCM_RAG_BUDGET"] = str(matched_tokens)
        if job.stage == "longcontext_controls_corrected":
            env.update(
                {
                    "GCM_MATCH_READER_BUDGET": "1",
                    "GCM_MATCH_CHUNK": "4096",
                    "GCM_MATCH_K": "128",
                }
            )
        if job.stage == "agent_memory_corrected" and job.method in ("window", "rag", "ll2"):
            env.update(
                {
                    "GCM_MATCH_READER_BUDGET": "1",
                    "GCM_MATCH_CHUNK": "4096",
                    "GCM_MATCH_K": "128",
                }
            )
        if job.method == "sft":
            if job.stage in (
                "longcontext",
                "longcontext_quality_corrected",
                "longcontext_babilong_corrected",
            ):
                env["GCM_LOAD_ADAPTER"] = str(_transfer_adapter(job))
            elif adapter := _completed_training_adapter(job):
                env["GCM_LOAD_ADAPTER"] = str(adapter)
        return [str(PYTHON), str(ROOT / "experiments/paper_a_baseline.py"), model_path], env, None
    if job.method == "gist" or job.method in HARNESS_BASELINES_Q3 or job.method in HARNESS_BASELINES_LINEAR:
        out_dir = PAPER_ROOT / "harness" / job.tag
        cfg = BENCH[job.bench]
        cmd = [
            str(PYTHON),
            "-m",
            "mem_embedding.gcm.harness",
            "--base",
            model_path,
            "--device",
            "cuda:0",
            "--methods",
            job.method,
            "--train-dataset",
            job.bench,
            "--eval-benches",
            job.bench,
            "--out",
            str(out_dir),
            "--n-memory",
            "128",
            "--n-items",
            str(cfg["ntrain"]),
            "--n-eval",
            "100000",
            "--max-ctx-tokens",
            str(cfg["enc"]),
            "--max-new-tokens",
            str(cfg["gen"]),
            "--steps",
            "2000",
            "--base-lora-rank",
            "64",
            "--seed",
            str(job.seed),
            "--signals",
        ]
        return cmd, env, out_dir
    raise ValueError(f"unknown method: {job.method}")


def _copy_artifacts(job: Job, gist_dir: Path | None) -> dict[str, str]:
    dest = PAPER_ROOT / "artifacts" / job.tag
    dest.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    if gist_dir is not None:
        for source in gist_dir.glob("*"):
            if source.is_file():
                target = dest / source.name
                shutil.copy2(source, target)
                copied[source.name] = str(target)
        return copied
    model_name = Path(MODELS[job.model]).name
    out = ROOT / "out" / model_name
    for suffix in (".json", "_adapters.pt"):
        source = out / f"{job.tag}{suffix}"
        if source.exists():
            target = dest / source.name
            shutil.copy2(source, target)
            copied[source.name] = str(target)
    return copied


def _artifacts_complete(
    job: Job,
    gist_dir: Path | None,
    copied: dict[str, str],
) -> bool:
    """Require the outputs downstream consumers need before marking a cell done."""
    if gist_dir is not None:
        name = f"records_{job.bench}.jsonl"
        path = Path(copied.get(name, ""))
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            with path.open() as handle:
                first_record = next(
                    line for line in handle if line.strip()
                )
            json.loads(first_record)
        except (OSError, StopIteration, json.JSONDecodeError):
            return False
        return True
    if f"{job.tag}.json" not in copied:
        return False
    if job.stage == "longcontext_controls_corrected":
        try:
            payload = json.loads(Path(copied[f"{job.tag}.json"]).read_text())
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("status") != "ok" or not payload.get("records"):
            return False
    needs_adapter = job.method == "ours" or (
        job.method == "sft"
        and job.stage not in (
            "longcontext",
            "longcontext_quality_corrected",
            "longcontext_babilong_corrected",
        )
    )
    return not needs_adapter or f"{job.tag}_adapters.pt" in copied


def _existing_artifacts_complete(job: Job) -> bool:
    artifact_dir = PAPER_ROOT / "artifacts" / job.tag
    copied = {
        path.name: str(path)
        for path in artifact_dir.glob("*")
        if path.is_file()
    }
    is_harness = (
        job.method == "gist"
        or job.method in HARNESS_BASELINES_Q3
        or job.method in HARNESS_BASELINES_LINEAR
    )
    return _artifacts_complete(
        job,
        artifact_dir if is_harness else None,
        copied,
    )


def _status_is_live(status: dict[str, Any], timeout_hours: float) -> bool:
    if status.get("state") != "running":
        return False
    age = time.time() - float(status.get("started", 0))
    if age >= timeout_hours * 3600 * 1.2:
        return False
    status_pid = status.get("worker_pid")
    status_host = status.get("hostname")
    if not status_pid and not status_host:
        return True  # Compatibility for workers launched before PID leases.
    if status_host != socket.gethostname():
        return False
    try:
        pid = int(status_pid)
        os.kill(pid, 0)
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.exists() and proc_stat.read_text().split()[2] == "Z":
            return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def _protocol_revision(job: Job) -> str | None:
    if job.stage == "longcontext_controls_corrected":
        return LONG_TARGET_CONTROL_PROTOCOL_REVISION
    if job.stage == "agent_memory_corrected":
        return AGENT_MEMORY_PROTOCOL_REVISION
    if job.stage == "state_ablation":
        return STATE_ABLATION_PROTOCOL_REVISION
    if (
        job.stage in {"main_baselines_corrected", "narrativeqa_baselines"}
        and job.method in {"ll2", "longllm", "llorig", "rag"}
    ):
        return BASELINE_PROTOCOL_REVISION
    if job.stage == "quality_corrected" and job.method == "ll2":
        return BASELINE_PROTOCOL_REVISION
    return None


def run_job(job: Job, gpu: str, timeout_hours: float) -> int:
    logs = PAPER_ROOT / "logs"
    status_dir = PAPER_ROOT / "status"
    logs.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / f"{job.tag}.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            is_live = _status_is_live(status, timeout_hours)
            is_complete = (
                status.get("state") == "done"
                and _existing_artifacts_complete(job)
                and (
                    _protocol_revision(job) is None
                    or status.get("protocol_revision") == _protocol_revision(job)
                )
            )
            if is_complete or is_live:
                print(f"SKIP {job.tag} state={status.get('state')}", flush=True)
                return 0
        except Exception:
            pass

    cmd, overrides, gist_dir = command_for(job)
    run_env = os.environ.copy()
    run_env.update(overrides)
    run_env["CUDA_VISIBLE_DEVICES"] = gpu
    started = time.time()
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "job": dataclasses.asdict(job),
                "started": started,
                "worker_pid": os.getpid(),
                "hostname": socket.gethostname(),
                "protocol_revision": _protocol_revision(job),
            },
            indent=2,
        )
    )
    log_path = logs / f"{job.tag}.log"
    print(f"START {job.tag} gpu={gpu}", flush=True)
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=run_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_hours * 3600,
                check=False,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
            log.write(f"\nCELL_TIMEOUT {timeout_hours}h\n")
    copied = _copy_artifacts(job, gist_dir) if code == 0 else {}
    state = "done" if code == 0 and _artifacts_complete(job, gist_dir, copied) else "failed"
    status_path.write_text(
        json.dumps(
            {
                "state": state,
                "exit_code": code,
                "job": dataclasses.asdict(job),
                "started": started,
                "elapsed_seconds": time.time() - started,
                "log": str(log_path),
                "artifacts": copied,
                "protocol_revision": _protocol_revision(job),
            },
            indent=2,
        )
    )
    print(f"END {job.tag} state={state} code={code}", flush=True)
    return 0 if state == "done" else code or 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "main",
            "quality_corrected",
            "main_baselines_corrected",
            "narrativeqa_baselines",
            "generality",
            "ablation",
            "adversarial_ablation",
            "state_ablation",
            "budget",
            "replicate",
            "sft_reaudit",
            "transfer_train",
            "transfer_train_quality_corrected",
            "longcontext",
            "longcontext_quality_corrected",
            "longcontext_controls_corrected",
            "agent_memory_corrected",
            "longcontext_babilong_corrected",
            "compression_baselines",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timeout-hours", type=float, default=24)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--models", default="", help="Comma-separated model slugs to retain from the stage manifest")
    parser.add_argument("--methods", default="", help="Comma-separated methods to retain from the stage manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args.stage)
    if args.models:
        allowed = {value for value in args.models.split(",") if value}
        jobs = [job for job in jobs if job.model in allowed]
    if args.methods:
        allowed = {value for value in args.methods.split(",") if value}
        jobs = [job for job in jobs if job.method in allowed]
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    (PAPER_ROOT / f"manifest-{args.stage}.json").write_text(
        json.dumps([dataclasses.asdict(job) | {"tag": job.tag} for job in jobs], indent=2)
    )
    if args.list:
        for index, job in enumerate(jobs):
            print(index, job.tag)
        return
    selected = [
        job for index, job in enumerate(jobs)
        if index % args.num_shards == args.shard_index
    ]
    failures = 0
    for job in selected:
        code = run_job(job, args.gpu, args.timeout_hours)
        if code:
            failures += 1
            if not args.continue_on_error:
                break
    print(f"WORKER_DONE jobs={len(selected)} failures={failures}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
