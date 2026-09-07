#!/usr/bin/env python3
"""Export Paper A benchmark items to the official-baseline JSONL contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _context(item: Any) -> str:
    return "\n".join(str(part) for part in (getattr(item, "chunks", None) or []))


def _fold(item: Any, dev_fraction: float = 0.2) -> str:
    payload = "\u241f".join(
        (
            _context(item),
            str(getattr(item, "query", "")),
            str(getattr(item, "gold", "")),
        )
    )
    bucket = int(hashlib.md5(payload.encode("utf-8")).hexdigest(), 16) % 10_000
    return "dev" if bucket < int(dev_fraction * 10_000) else "test"


def _answer_list(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("text", [])
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(part).strip() for part in value if str(part).strip()]


def _gold_label(value: Any, options: list[str]) -> str:
    text = str(value).strip()
    if len(text) == 1 and text.upper() in {
        chr(ord("A") + index) for index in range(len(options))
    }:
        return text.upper()
    if text in options:
        return chr(ord("A") + options.index(text))
    try:
        index = int(value)
    except (TypeError, ValueError):
        return text
    return chr(ord("A") + index) if 0 <= index < len(options) else text


def canonical_record(item: Any, bench: str, options: list[str]) -> dict[str, Any]:
    gold = getattr(item, "gold", "")
    record: dict[str, Any] = {
        "task": bench,
        "item_id": str(getattr(item, "item_id", "")),
        "context": _context(item),
        "question": str(getattr(item, "query", "")),
        "answers": _answer_list(gold),
        "source_chunks": len(getattr(item, "chunks", None) or []),
        "evaluation_fold": _fold(item),
    }
    if options:
        record["options"] = [str(option) for option in options]
        record["gold"] = _gold_label(gold, record["options"])
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benches", required=True, help="Comma-separated benchmark IDs")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--fold", choices=("all", "dev", "test"), default="test")
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--n-chunks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from mem_embedding.gcm.data import load_items, options_for

    records: list[dict[str, Any]] = []
    for bench in (part.strip() for part in args.benches.split(",")):
        if not bench:
            continue
        for item in load_items(bench, args.n, args.n_chunks, args.seed, args.split):
            if args.fold != "all" and _fold(item) != args.fold:
                continue
            records.append(canonical_record(item, bench, list(options_for(item) or [])))
    if not records:
        raise RuntimeError("official-baseline export produced no records")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    with open(temporary, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "n_records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
