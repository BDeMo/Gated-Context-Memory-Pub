"""v1.7 dataset registry, transfer taxonomy, and scoring helpers.

All loaders and metrics come from ``llm_infra`` (public resources). This module only wires the bench
names we use, the in/cross-task and in/cross-domain taxonomy, and a couple of thin scoring helpers.
"""
from __future__ import annotations

import inspect as _inspect
from typing import Any, Callable

from llm_infra.benchmark_metrics import score_item
from llm_infra.datasets import (
    generate_apibank,
    generate_babilong,
    generate_bfcl,
    generate_categorical_niah,
    generate_coding_niah,
    generate_glaive,
    generate_hermes,
    generate_hotpot_qa,
    generate_infinitebench,
    generate_locomo,
    generate_longbench_v2,
    generate_msmarco,
    generate_multi_needle_niah,
    generate_musr,
    generate_narrativeqa,
    generate_numerical_niah,
    generate_quality,
    generate_quality_hard,
    generate_rca,
    generate_ruler_niah,
    generate_squad_v2,
    generate_toolace,
    generate_trivia_qa,
)

# Bench name -> dataset generator (reused from llm_infra).
def _bfcl_pooled(n_items: int = 96, n_chunks: int = 5, seed: int = 0, split: str = "train") -> list:
    """Pool BFCL categories into one larger high-headroom tool corpus to break the per-category ~280-item cap
    (data-scaling test: does more tool DATA lift compress past the ~0.72 steps-plateau toward full?)."""
    import random as _r
    items: list = []
    for i, c in enumerate(["simple", "multiple", "live_multiple"]):
        items += generate_bfcl(category=c, n_items=n_items, n_chunks=n_chunks, seed=seed + 1000 * (i + 1), split=split)
    _r.Random(seed).shuffle(items)
    return items[:n_items]


_MIX_BENCHES = ["bfcl_simple", "bfcl_multiple", "bfcl_live_multiple", "toolace", "hermes", "glaive",
                "rca_openrca", "hotpot_qa", "squad_v2", "musr_mm", "narrativeqa", "quality"]


def _mix(n_items: int = 384, n_chunks: int = 1, seed: int = 0, split: str = "train") -> list:
    """MIX / scaling corpus: pool train items across all offline domains, then order by CURRICULUM
    (ascending context length: easy/short -> hard/long). Scaling the data axis = taking a larger easy->hard
    prefix (n_items). Used by the mix-training scaling study (data x depth x FLOPs)."""
    per = max(n_items // len(_MIX_BENCHES) + 6, 10)
    pool: list = []
    for i, b in enumerate(_MIX_BENCHES):
        try:
            pool += load_items(b, per, n_chunks, seed + 131 * i, split)
        except Exception:  # noqa: BLE001
            pass

    def _ctxlen(it: Any) -> int:
        return len("".join(map(str, getattr(it, "chunks", []) or []))) + len(str(getattr(it, "query", "")))

    pool.sort(key=_ctxlen)   # curriculum: short/easy -> long/hard
    return pool[:n_items]


EVAL_GENS: dict[str, Callable[..., list]] = {
    "longbench_v2": generate_longbench_v2,
    "infbench_choice": lambda **k: generate_infinitebench(
        config="longbook_choice_eng", **k
    ),
    "babilong_qa1_4k": lambda **k: generate_babilong(
        task="qa1", config="4k", **k
    ),
    "babilong_qa1_16k": lambda **k: generate_babilong(
        task="qa1", config="16k", **k
    ),
    "babilong_qa2_16k": lambda **k: generate_babilong(
        task="qa2", config="16k", **k
    ),
    "babilong_qa3_16k": lambda **k: generate_babilong(
        task="qa3", config="16k", **k
    ),
    "quality": generate_quality,
    "bfcl_pooled": _bfcl_pooled,
    "mix": _mix,
    "musr_mm": lambda **k: generate_musr(split="murder_mysteries", **k),
    "ruler_niah": generate_ruler_niah,
    "hotpot_qa": generate_hotpot_qa,
    "trivia_qa": generate_trivia_qa,
    "squad_v2": generate_squad_v2,
    "narrativeqa": generate_narrativeqa,
    "ms_marco": generate_msmarco,
    "bfcl_simple": lambda **k: generate_bfcl(category="simple", **k),
    "bfcl_multiple": lambda **k: generate_bfcl(category="multiple", **k),
    "bfcl_irrelevance": lambda **k: generate_bfcl(category="irrelevance", **k),
    "bfcl_live_multiple": lambda **k: generate_bfcl(category="live_multiple", **k),
    "bfcl_live_irrelevance": lambda **k: generate_bfcl(category="live_irrelevance", **k),
    "bfcl_parallel": lambda **k: generate_bfcl(category="parallel", **k),
    "bfcl_parallel_multiple": lambda **k: generate_bfcl(category="parallel_multiple", **k),
    "bfcl_live_simple": lambda **k: generate_bfcl(category="live_simple", **k),
    "bfcl_live_parallel": lambda **k: generate_bfcl(category="live_parallel", **k),
    "bfcl_java": lambda **k: generate_bfcl(category="java", **k),
    "bfcl_javascript": lambda **k: generate_bfcl(category="javascript", **k),
    "bfcl_rest": lambda **k: generate_bfcl(category="rest", **k),
    "bfcl_sql": lambda **k: generate_bfcl(category="sql", **k),
    "rca_openrca": lambda **k: generate_rca(source="openrca", **k),
    "rca_rcaeval": lambda **k: generate_rca(source="rcaeval", **k),
    "apibank": generate_apibank,
    "toolace": generate_toolace,
    "hermes": generate_hermes,
    "glaive": generate_glaive,
    # --- v2.0.0 long-context suite (length via n_chunks / GCM_NCHUNKS; ~182 tok/chunk for the NIAH family) ---
    "multi_needle_niah": generate_multi_needle_niah,   # RULER multi-key retrieval (synthetic, length-controlled)
    "numerical_niah": generate_numerical_niah,         # number needle in haystack
    "categorical_niah": generate_categorical_niah,     # category needle
    "coding_niah": generate_coding_niah,               # code needle
    "locomo": generate_locomo,                         # long multi-session dialogue memory (~7k)
    "quality_hard": generate_quality_hard,             # hard QuALITY MC (long literary)
}

# Benches scored by multiple-choice log-likelihood (not free generation).
MC_BENCHES = {
    "quality",
    "quality_hard",
    "musr_mm",
    "rca_openrca",
    "rca_rcaeval",
    "longbench_v2",
    "infbench_choice",
}

# Primary metric key (from score_item's dict) for the generation benches.
PRIMARY = {
    "ruler_niah": "exact_value_match",
    "hotpot_qa": "squad_f1",
    "squad_v2": "squad_f1",
    "trivia_qa": "squad_f1",
    "narrativeqa": "rouge_l",
    "ms_marco": "rouge_l",
    "bfcl_simple": "tool_acc",
    "bfcl_pooled": "tool_acc",
    "bfcl_multiple": "tool_acc",
    "bfcl_irrelevance": "tool_acc",
    "bfcl_live_multiple": "tool_acc",
    "bfcl_live_irrelevance": "tool_acc",
    "bfcl_parallel": "tool_acc", "bfcl_parallel_multiple": "tool_acc",
    "bfcl_live_simple": "tool_acc", "bfcl_live_parallel": "tool_acc",
    "bfcl_java": "tool_acc", "bfcl_javascript": "tool_acc", "bfcl_rest": "tool_acc", "bfcl_sql": "tool_acc",
    "rca_openrca": "primary_service_match",
    "rca_rcaeval": "primary_service_match",
    "apibank": "tool_acc",
    "toolace": "tool_acc",
    "hermes": "tool_acc",
    "glaive": "tool_acc",
    "multi_needle_niah": "exact_value_match", "numerical_niah": "exact_value_match",
    "categorical_niah": "exact_value_match", "coding_niah": "exact_value_match",
    "locomo": "squad_f1",
    "babilong_qa1_4k": "answer_accuracy",
    "babilong_qa1_16k": "answer_accuracy",
    "babilong_qa2_16k": "answer_accuracy",
    "babilong_qa3_16k": "answer_accuracy",
}

# Domain taxonomy: a Cartridge corpus == one training set. Each (train -> eval) pair is one of three
# relations: in_task (same bench) / cross_task_in_domain (same domain, diff bench) / cross_task_cross_domain.
DOMAIN = {
    "categorical_niah": "synthetic", "ruler_niah": "synthetic", "musr_mm": "synthetic",
    "multi_needle_niah": "synthetic", "numerical_niah": "synthetic", "coding_niah": "synthetic",
    "quality": "literary", "narrativeqa": "literary", "quality_hard": "literary",
    "locomo": "dialogue",
    "hotpot_qa": "wiki", "trivia_qa": "wiki", "squad_v2": "wiki",
    "ms_marco": "web",
    "longbench_v2": "longreal", "infbench_choice": "longreal",
    "babilong_qa1_4k": "synthetic", "babilong_qa1_16k": "synthetic",
    "babilong_qa2_16k": "synthetic", "babilong_qa3_16k": "synthetic",
    "mix": "mixed",
    "bfcl_simple": "tool", "bfcl_pooled": "tool", "bfcl_multiple": "tool", "bfcl_irrelevance": "tool",
    "bfcl_live_multiple": "tool", "bfcl_live_irrelevance": "tool", "apibank": "tool", "toolace": "tool",
    "bfcl_parallel": "tool", "bfcl_parallel_multiple": "tool", "bfcl_live_simple": "tool", "bfcl_live_parallel": "tool",
    "bfcl_java": "tool", "bfcl_javascript": "tool", "bfcl_rest": "tool", "bfcl_sql": "tool",
    "hermes": "tool", "glaive": "tool",
    "rca_openrca": "ops", "rca_rcaeval": "ops",
}


def relation(train_ds: str, eval_bench: str) -> str:
    """3-way transfer label of a (train corpus -> eval bench) pair."""
    if train_ds == eval_bench:
        return "in_task"
    dt, de = DOMAIN.get(train_ds), DOMAIN.get(eval_bench)
    if dt is not None and dt == de:
        return "cross_task_in_domain"
    return "cross_task_cross_domain"


def _call_gen(fn: Callable[..., list], **kw: Any) -> list:
    """Call a dataset generator, passing ``split`` only if it accepts it. Synthetic generators have no
    split (they differ by seed); eval-only datasets may lack the requested split, in which case we fall
    back to the generator's default rather than crashing."""
    try:
        params = _inspect.signature(fn).parameters
        if "split" not in params and not any(
            p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            kw.pop("split", None)
        return fn(**kw)
    except (TypeError, ValueError):
        kw.pop("split", None)
        return fn(**kw)


# Benches whose generator IGNORES the train/val split (single MuSR subset / synthetic seed-only) and would
# therefore leak between a seed-S train load and a seed-(S+1) eval load. We post-partition these by a stable
# content hash so 'train' and 'validation' are disjoint regardless of the seed (belt-and-suspenders; the
# manual-split tool/QA gens are already fixed via llm_infra.datasets._disjoint_split).
_NO_NATIVE_SPLIT = {"musr_mm", "ruler_niah", "categorical_niah", "multi_needle_niah", "numerical_niah", "coding_niah"}


def _hash_bucket(item: Any, frac: float = 0.7) -> str:
    """Bucket an item into train/val by a STABLE content signature (context + gold), NOT the item_id
    (which is enumeration-based and unstable across loads). Same content => same bucket => disjoint splits."""
    import hashlib
    ctx = "".join(map(str, getattr(item, "chunks", []) or []))
    key = ctx + "\u241f" + str(getattr(item, "gold", "")) + "\u241f" + str(getattr(item, "query", ""))
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 1000
    return "train" if h < int(frac * 1000) else "validation"


def load_items(bench: str, n_items: int, n_chunks: int, seed: int, split: str) -> list:
    """Load eval/train items for a bench via its llm_infra generator. For benches that lack a native train/val
    split, post-partition by a stable content hash so train and eval never overlap (leakage guard)."""
    if bench not in EVAL_GENS:
        raise KeyError(f"unknown bench {bench!r}; known: {sorted(EVAL_GENS)}")
    if bench in _NO_NATIVE_SPLIT and str(split) in ("train", "validation", "val", "test"):
        pool = _call_gen(EVAL_GENS[bench], n_items=max(n_items * 4, 400), n_chunks=n_chunks, seed=seed, split=split)
        want = "train" if str(split) == "train" else "validation"
        return [it for it in pool if _hash_bucket(it) == want][:n_items]
    return _call_gen(EVAL_GENS[bench], n_items=n_items, n_chunks=n_chunks, seed=seed, split=split)


def options_for(item: Any) -> list:
    """Multiple-choice options for an MC item (from its meta)."""
    md = item.meta or {}
    return list(md.get("options", md.get("choices", [])))


def _answer_line(prediction: str) -> str:
    """Extract the answer from a raw frozen-base generation. The base has no chat template and rambles after
    the answer (e.g. 'Chad\\n\\nOkay, so the question is asking...'), which deflates token-F1 via precision and
    biases the comparison toward whichever method happens to be terser. We score the FIRST non-empty line,
    uniformly for every method, so accuracy reflects correctness, not verbosity."""
    for line in str(prediction).splitlines():
        line = line.strip()
        if line:
            return line
    return str(prediction).strip()


def score_gen(bench: str, prediction: str, item: Any) -> tuple[float, dict]:
    """Score one generation prediction; returns (primary_scalar, full_metric_dict)."""
    ds = (item.meta or {}).get("dataset", bench)
    # Older BABILong artifacts mislabeled their examples as SQuAD-v2. Dispatch
    # by the benchmark name so reruns use BABILong's official answer accuracy.
    if bench.startswith("babilong"):
        ds = bench
    pred = _answer_line(prediction)
    metrics = score_item(dataset=ds, prediction=pred, gold=item.gold, meta=item.meta)
    key = PRIMARY.get(bench)
    if (getattr(item, "meta", None) or {}).get("gt_call") and "tool_call_acc" in metrics:
        key = "tool_call_acc"  # R3: in args-aware mode the primary is the full-call (name+args) match
    return float(metrics.get(key, 0.0)) if key else 0.0, metrics
