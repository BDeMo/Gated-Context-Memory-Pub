#!/usr/bin/env python3
"""Safe launcher and status tool for Paper A's official baseline worktrees.

This module never implements a compression method.  It verifies a pinned
authors' worktree and launches either an authors' native evaluator (LCLM) or
the thin I/O adapter in paper_a_official_eval.py, which imports authors' model
classes.  Commands are printed by default; GPU execution requires --execute.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_ROOT = Path(
    os.environ.get(
        "PAPER_A_OFFICIAL_ROOT",
        "/Users/s1shi/workspace/paper-a-official-baselines",
    )
)
ADAPTER = Path(__file__).with_name("paper_a_official_eval.py")
SCHEMA_VERSION = "paper-a-official-baseline/v1"


@dataclass(frozen=True)
class MethodSpec:
    repo_dir: str
    commit: str
    origin: str
    checkpoints: tuple[str, ...]


METHODS = {
    "lclm": MethodSpec(
        "lclm",
        "e04ceb982f10b05586c77d2d711b60a55d07805a",
        "https://github.com/LeonLixyz/LCLM.git",
        tuple(f"latent-context/0.6b-4b-LCLM-{ratio}x" for ratio in (4, 8, 16)),
    ),
    "semi-dynamic": MethodSpec(
        "semi-dynamic-context-compress",
        "07fae01c5062e57dc8277f7e696ca928cb775ca5",
        "https://github.com/yuyijiong/semi-dynamic-context-compress.git",
        ("yuyijiong/qwen3-semi-dynamic-soft-context-compress",),
    ),
    "autocompressor": MethodSpec(
        "autocompressors",
        "80352a4233c4a70504c75bf95c5f3e6d2533a863",
        "https://github.com/princeton-nlp/AutoCompressors.git",
        (
            "princeton-nlp/AutoCompressor-Llama-2-7b-6k",
            "princeton-nlp/AutoCompressor-2.7b-6k",
            "princeton-nlp/AutoCompressor-2.7b-30k",
            "princeton-nlp/AutoCompressor-1.3b-30k",
        ),
    ),
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def verify_worktree(method: str, official_root: Path = OFFICIAL_ROOT) -> dict[str, Any]:
    spec = METHODS[method]
    repo = official_root / spec.repo_dir
    errors: list[str] = []
    if not repo.is_dir():
        return {"ok": False, "repo": str(repo), "errors": ["worktree missing"]}
    try:
        head = _git(repo, "rev-parse", "HEAD")
        origin = _git(repo, "remote", "get-url", "origin")
        dirty = _git(repo, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"ok": False, "repo": str(repo), "errors": [str(exc)]}
    if head != spec.commit:
        errors.append(f"HEAD {head} != pinned {spec.commit}")
    if origin.rstrip("/") != spec.origin.rstrip("/"):
        errors.append(f"origin {origin!r} != official {spec.origin!r}")
    if dirty:
        errors.append("official worktree has local modifications")
    return {
        "ok": not errors,
        "repo": str(repo),
        "head": head,
        "origin": origin,
        "clean": not bool(dirty),
        "errors": errors,
    }


def _checkpoint_is_official(method: str, checkpoint: str) -> bool:
    if checkpoint in METHODS[method].checkpoints:
        return True
    path = Path(checkpoint).expanduser()
    # Local snapshots are permitted, but provenance remains the supplied
    # official checkpoint ID and must be stated separately in the manifest.
    return path.is_dir()


def build_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    repo = Path(args.official_root) / METHODS[args.method].repo_dir
    python = str(Path(args.python).expanduser())
    native = Path(args.run_dir).expanduser().resolve() / "native"
    if args.method == "lclm":
        command = [
            python,
            "-m",
            "inference.examples.eval_ruler_niah",
            "--checkpoint",
            args.checkpoint,
            "--prompts-dir",
            str(Path(args.input).expanduser().resolve()),
            "--work-dir",
            str(native),
            "--max-tokens",
            str(args.max_new_tokens),
            "--max-encode-batch-size",
            str(args.max_encode_batch_size),
        ]
    else:
        command = [
            python,
            str(ADAPTER),
            "--method",
            args.method,
            "--official-repo",
            str(repo),
            "--checkpoint",
            args.checkpoint,
            "--checkpoint-id",
            args.checkpoint_id or args.checkpoint,
            "--input",
            str(Path(args.input).expanduser().resolve()),
            "--output",
            str(native / "canonical.jsonl"),
            "--variant",
            args.variant,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--segment-size",
            str(args.segment_size),
            "--raw-max-tokens",
            str(args.raw_max_tokens),
        ]
    return command, repo


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    worktree = verify_worktree(args.method, Path(args.official_root))
    errors = list(worktree["errors"])
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        errors.append(f"input missing: {input_path}")
    python = Path(args.python).expanduser()
    if not python.is_file():
        errors.append(f"isolated-environment Python missing: {python}")
    if not _checkpoint_is_official(args.method, args.checkpoint):
        errors.append(
            f"checkpoint must be an allow-listed official ID or local snapshot: {args.checkpoint}"
        )
    if args.method == "semi-dynamic" and not Path(args.checkpoint).expanduser().is_dir():
        errors.append(
            "Semi-Dynamic requires a concrete local checkpoint subdirectory "
            "(pass its official HF identity with --checkpoint-id)"
        )
    if Path(args.run_dir).expanduser().resolve() == input_path.resolve():
        errors.append("run directory must differ from input")
    command, repo = build_command(args)
    return {
        "ok": not errors,
        "schema_version": SCHEMA_VERSION,
        "classification": "official-authors-code-and-checkpoint",
        "method": args.method,
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "checkpoint_id": args.checkpoint_id or args.checkpoint,
        "worktree": worktree,
        "repo": str(repo),
        "command": command,
        "cwd": str(repo),
        "input": str(input_path.resolve()),
        "run_dir": str(Path(args.run_dir).expanduser().resolve()),
        "errors": errors,
    }


def launch(args: argparse.Namespace) -> int:
    plan = preflight(args)
    print(json.dumps(plan, indent=2))
    if not plan["ok"]:
        return 2
    if not args.execute:
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [part for part in visible.split(",") if part.strip()]
    if len(devices) != 1:
        print(
            "refusing GPU launch: set CUDA_VISIBLE_DEVICES to exactly one device",
            file=sys.stderr,
        )
        return 2
    run_dir = Path(plan["run_dir"])
    status_path = run_dir / "status.json"
    if status_path.exists() and not args.resume:
        print(f"refusing to overwrite existing run: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "native").mkdir(exist_ok=True)
    _atomic_json(run_dir / "manifest.json", plan)
    log = open(run_dir / "run.log", "ab", buffering=0)
    process = subprocess.Popen(
        plan["command"],
        cwd=plan["cwd"],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    _atomic_json(
        status_path,
        {
            "state": "running",
            "pid": process.pid,
            "hostname": socket.gethostname(),
            "started_at": time.time(),
            "manifest": str(run_dir / "manifest.json"),
        },
    )
    print(f"started pid={process.pid}; status={status_path}")
    return 0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def status(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return {"state": "not-started", "run_dir": str(run_dir)}
    value = json.loads(status_path.read_text())
    canonical = run_dir / "native" / "canonical.jsonl"
    summary = run_dir / "native" / "summary.json"
    if canonical.is_file() and canonical.stat().st_size:
        value["state"] = "complete"
        value["canonical_output"] = str(canonical)
    elif summary.is_file() and summary.stat().st_size:
        # LCLM's native driver has aggregate output but no same-base references.
        value["state"] = "native-complete-needs-canonicalization"
        value["native_summary"] = str(summary)
    elif (
        value.get("state") == "running"
        and value.get("hostname") == socket.gethostname()
        and not _pid_alive(int(value["pid"]))
    ):
        value["state"] = "exited-without-canonical-output"
    return value


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-id",
        default="",
        help="Official HF ID when --checkpoint is a local snapshot",
    )
    parser.add_argument("--python", required=True, help="Python from the isolated method env")
    parser.add_argument("--official-root", default=str(OFFICIAL_ROOT))
    parser.add_argument("--variant", default="default")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-encode-batch-size", type=int, default=128)
    parser.add_argument("--segment-size", type=int, default=1536)
    parser.add_argument("--raw-max-tokens", type=int, default=6144)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    plan_parser = sub.add_parser("plan")
    _add_common(plan_parser)
    launch_parser = sub.add_parser("launch")
    _add_common(launch_parser)
    launch_parser.add_argument("--execute", action="store_true")
    launch_parser.add_argument("--resume", action="store_true")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--run-dir", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--method", choices=sorted(METHODS), required=True)
    verify_parser.add_argument("--official-root", default=str(OFFICIAL_ROOT))
    args = parser.parse_args()
    if args.action == "verify":
        result = verify_worktree(args.method, Path(args.official_root))
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2
    if args.action == "status":
        print(json.dumps(status(Path(args.run_dir).expanduser().resolve()), indent=2))
        return 0
    if args.action == "plan":
        result = preflight(args)
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 2
    return launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
