"""Datasets for Plan 08 v0.

Phase 0 ships the synthetic **coding NIAH** generator: each item has `n_chunks`
fake-Python "files" (functions, imports, comments), one of which contains a
planted needle. The probe asks "What value does function `foo_<id>` return?"
and the gold answer is the planted value. Everything is deterministic given a
seed.

Phase 1 will add `RepoBench-C` and `LongBench` adapters; their slots are
stubbed below so the harness contract is fixed early.
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class LongContextItem:
    """One long-context item with chunks, query, gold, plus optional metadata.

    Fields:
        item_id:      Stable identifier for the item.
        chunks:       Ordered list of chunk texts. `n_chunks = len(chunks)`.
        query:        The question (or completion prompt) the model answers.
        gold:         Reference answer string (exact-match or contained-in).
        needle_chunk: Index into `chunks` where the answer was planted.
                      `-1` if the dataset is not needle-style.
        meta:         Free-form metadata for downstream analysis.
    """

    item_id: str
    chunks: list[str]
    query: str
    gold: str
    needle_chunk: int = -1
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _rand_name(rng: random.Random, prefix: str = "foo") -> str:
    suffix = "".join(rng.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{suffix}"


def _rand_value(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=8))


_DISTRACTOR_TEMPLATES = (
    "def {name}({arg}):\n    \"\"\"Stub for {arg} processing.\"\"\"\n    return {arg} * 2\n",
    "class {Name}:\n    \"\"\"Helper {Name}.\"\"\"\n\n    def __init__(self, x):\n        self.x = x\n",
    "import {pkg}\n\n# Configure {pkg} for use in pipeline\n{pkg}.configure(strict=True)\n",
    "TIMEOUT_{const} = {num}\nRETRIES_{const} = {num2}\n# Tunable knobs for module {const}\n",
)


def _distractor_chunk(rng: random.Random) -> str:
    lines: list[str] = []
    for _ in range(rng.randint(4, 9)):
        tpl = rng.choice(_DISTRACTOR_TEMPLATES)
        lines.append(
            tpl.format(
                name=_rand_name(rng, "helper"),
                Name=_rand_name(rng, "Help").title().replace("_", ""),
                arg=_rand_name(rng, "x"),
                pkg=_rand_name(rng, "lib"),
                const=_rand_name(rng, "K").upper(),
                num=rng.randint(1, 99),
                num2=rng.randint(1, 99),
            )
        )
    return "\n".join(lines)


def _needle_chunk(rng: random.Random, target_name: str, target_value: str) -> str:
    distractor = _distractor_chunk(rng)
    needle = (
        f"def {target_name}():\n"
        f"    \"\"\"Returns the secret token planted in this synthetic case.\"\"\"\n"
        f'    return "{target_value}"\n'
    )
    parts = distractor.splitlines()
    insert_at = rng.randint(0, len(parts))
    parts.insert(insert_at, needle)
    return "\n".join(parts)


def generate_coding_niah(
    *,
    n_items: int = 50,
    n_chunks: int = 8,
    seed: int = 12345,
) -> list[LongContextItem]:
    """Generate a deterministic coding-NIAH split.

    Each item has `n_chunks` synthetic Python chunks; one chunk contains a
    `def foo_<id>(): return "<VALUE>"` that the query asks about. Position is
    uniform random in [0, n_chunks).
    """

    rng = random.Random(seed)
    items: list[LongContextItem] = []

    for i in range(n_items):
        target_name = _rand_name(rng, "foo")
        target_value = _rand_value(rng)
        needle_idx = rng.randrange(n_chunks)

        chunks = []
        for j in range(n_chunks):
            if j == needle_idx:
                chunks.append(_needle_chunk(rng, target_name, target_value))
            else:
                chunks.append(_distractor_chunk(rng))

        items.append(
            LongContextItem(
                item_id=f"coding_niah_{i:04d}",
                chunks=chunks,
                query=f"What value does the function {target_name} return?",
                gold=target_value,
                needle_chunk=needle_idx,
                meta={
                    "target_name": target_name,
                    "n_chunks": n_chunks,
                    "dataset": "coding_niah",
                },
            )
        )
    return items


_CATEGORICAL_OPS = (
    ("sum", "the sum of its arguments", "return a + b"),
    ("product", "the product of its arguments", "return a * b"),
    ("max", "the maximum of its arguments", "return max(a, b)"),
    ("concat", "the string concatenation of its arguments", "return str(a) + str(b)"),
)
"""(label, english description, python body) for the categorical NIAH task.

Four classes → 2 bits of answer entropy. Within reach of soft-prompt memory:
a successful wrapper only has to surface which-of-4 the needle chunk
defines, not recall an arbitrary 40-bit string. Use this dataset to
validate the mem-X architecture end-to-end before going back to the harder
exact-string ``coding_niah`` task.
"""


def _categorical_needle_chunk(
    rng: random.Random, target_name: str, op_label: str, op_body: str
) -> str:
    distractor = _distractor_chunk(rng)
    needle = (
        f"def {target_name}(a, b):\n"
        f"    \"\"\"Computes the {op_label} planted in this synthetic case.\"\"\"\n"
        f"    {op_body}\n"
    )
    parts = distractor.splitlines()
    insert_at = rng.randint(0, len(parts))
    parts.insert(insert_at, needle)
    return "\n".join(parts)


def generate_categorical_niah(
    *,
    n_items: int = 50,
    n_chunks: int = 8,
    seed: int = 12345,
) -> list[LongContextItem]:
    """Generate a deterministic categorical-NIAH split.

    Same haystack as ``generate_coding_niah`` but the planted function
    computes one of four operations (sum / product / max / concat). The
    query asks for the operation label; the gold is the label string
    (e.g. ``"sum"``). 2 bits of answer entropy — the soft-memory
    wrappers should be able to surface this with a small K.
    """

    rng = random.Random(seed)
    items: list[LongContextItem] = []

    for i in range(n_items):
        target_name = _rand_name(rng, "foo")
        op_label, op_english, op_body = _CATEGORICAL_OPS[rng.randrange(len(_CATEGORICAL_OPS))]
        needle_idx = rng.randrange(n_chunks)

        chunks = []
        for j in range(n_chunks):
            if j == needle_idx:
                chunks.append(_categorical_needle_chunk(rng, target_name, op_label, op_body))
            else:
                chunks.append(_distractor_chunk(rng))

        items.append(
            LongContextItem(
                item_id=f"categorical_niah_{i:04d}",
                chunks=chunks,
                query=(
                    f"Which operation does the function {target_name} compute? "
                    "Respond with exactly one of these labels and nothing else: "
                    "sum, product, max, concat."
                ),
                gold=op_label,
                needle_chunk=needle_idx,
                meta={
                    "target_name": target_name,
                    "op_english": op_english,
                    "n_chunks": n_chunks,
                    "dataset": "categorical_niah",
                    "answer_choices": [c[0] for c in _CATEGORICAL_OPS],
                },
            )
        )
    return items


def generate_categorical_niah_k2(
    *,
    n_items: int = 50,
    n_chunks: int = 8,
    seed: int = 12345,
) -> list[LongContextItem]:
    """Toy 2-class categorical NIAH: only ``sum`` vs ``product`` planted.

    1 bit of answer entropy — the minimum non-trivial classification task
    on this haystack. Used as a sanity check (Tier-A diagnostic A2): if
    even this fails, the wrapper architecture has a fundamental bug
    that cannot be resolved by hyperparameter sweeps.
    """

    rng = random.Random(seed)
    items: list[LongContextItem] = []
    two_ops = _CATEGORICAL_OPS[:2]  # sum, product

    for i in range(n_items):
        target_name = _rand_name(rng, "foo")
        op_label, op_english, op_body = two_ops[rng.randrange(2)]
        needle_idx = rng.randrange(n_chunks)

        chunks = []
        for j in range(n_chunks):
            if j == needle_idx:
                chunks.append(_categorical_needle_chunk(rng, target_name, op_label, op_body))
            else:
                chunks.append(_distractor_chunk(rng))

        items.append(
            LongContextItem(
                item_id=f"categorical_niah_k2_{i:04d}",
                chunks=chunks,
                query=(
                    f"Which operation does the function {target_name} compute? "
                    "Respond with exactly one of these labels and nothing else: "
                    "sum, product."
                ),
                gold=op_label,
                needle_chunk=needle_idx,
                meta={
                    "target_name": target_name,
                    "op_english": op_english,
                    "n_chunks": n_chunks,
                    "dataset": "categorical_niah_k2",
                    "answer_choices": [c[0] for c in two_ops],
                },
            )
        )
    return items


def _multi_needle_chunk(
    rng: random.Random,
    needles: list[tuple[str, str]],  # list of (name, value)
) -> str:
    """Plant ``len(needles)`` distinct return-value functions in one chunk."""

    distractor = _distractor_chunk(rng)
    parts = distractor.splitlines()
    for name, value in needles:
        body = (
            f"def {name}():\n"
            f"    \"\"\"Returns the planted value for this synthetic case.\"\"\"\n"
            f"    return \"{value}\"\n"
        )
        insert_at = rng.randint(0, len(parts))
        parts.insert(insert_at, body)
    return "\n".join(parts)


def generate_multi_needle_niah(
    *,
    n_items: int = 50,
    n_chunks: int = 6,
    k_needles: int = 3,
    needles_per_chunk: int = 1,
    seed: int = 12345,
) -> list[LongContextItem]:
    """Generate a multi-needle NIAH: ``k_needles`` planted facts, ask for one.

    The wrapper must remember *all* planted (name, value) pairs distinctly,
    then surface the one named in the query. This stresses the
    "memory as index" property that single-needle NIAH does not — a
    wrapper that simply caches "the" value will fail because there are
    multiple candidate values.

    Capacity-curve note: with ``k_needles`` planted 8-char values, the
    *total* information to compress is ~k × 40 bits. The wrapper must
    keep the per-name binding intact, not just the global set.

    Parameters
    ----------
    k_needles : number of distinct facts to plant. Distributed across
        ``needles_per_chunk`` per needle-chunk; remaining chunks are
        plain distractors.
    """

    if k_needles < 2:
        raise ValueError(
            "k_needles must be ≥2; use generate_coding_niah for k=1."
        )
    needle_chunks_needed = (k_needles + needles_per_chunk - 1) // needles_per_chunk
    if needle_chunks_needed > n_chunks:
        raise ValueError(
            f"need {needle_chunks_needed} chunks for {k_needles} needles at "
            f"{needles_per_chunk}/chunk, but n_chunks={n_chunks}"
        )

    rng = random.Random(seed)
    items: list[LongContextItem] = []
    for i in range(n_items):
        # generate k unique (name, value) pairs
        names: list[str] = []
        values: list[str] = []
        used_names: set[str] = set()
        used_values: set[str] = set()
        while len(names) < k_needles:
            n = _rand_name(rng, "foo")
            if n in used_names:
                continue
            v = _rand_value(rng)
            if v in used_values:
                continue
            names.append(n)
            values.append(v)
            used_names.add(n)
            used_values.add(v)

        # assign needles to chunk indices
        all_chunk_idx = list(range(n_chunks))
        rng.shuffle(all_chunk_idx)
        needle_chunk_idx = sorted(all_chunk_idx[:needle_chunks_needed])

        # round-robin assignment of needles to needle chunks
        per_chunk_needles: dict[int, list[tuple[str, str]]] = {
            c: [] for c in needle_chunk_idx
        }
        for idx, (name, val) in enumerate(zip(names, values)):
            tgt = needle_chunk_idx[idx % needle_chunks_needed]
            per_chunk_needles[tgt].append((name, val))

        chunks: list[str] = []

        for j in range(n_chunks):
            if j in per_chunk_needles:
                chunks.append(_multi_needle_chunk(rng, per_chunk_needles[j]))
            else:
                chunks.append(_distractor_chunk(rng))

        # query: ask about one of the k planted names (uniform pick)
        q_idx = rng.randrange(k_needles)
        target_name, target_value = names[q_idx], values[q_idx]
        needle_chunk_of_target = next(
            c for c, ns in per_chunk_needles.items()
            if any(nm == target_name for nm, _ in ns)
        )
        items.append(
            LongContextItem(
                item_id=f"multi_needle_niah_k{k_needles}_{i:04d}",
                chunks=chunks,
                query=f"What value does the function {target_name} return?",
                gold=target_value,
                needle_chunk=needle_chunk_of_target,
                meta={
                    "target_name": target_name,
                    "k_needles": k_needles,
                    "needles_per_chunk": needles_per_chunk,
                    "n_chunks": n_chunks,
                    "planted_names": names,
                    "planted_values": values,
                    "dataset": "multi_needle_niah",
                },
            )
        )
    return items


def _same_form_distractor_chunk(
    rng: random.Random, family_name: str
) -> str:
    """Distractor chunk that visually looks identical to a needle: contains
    a function named ``foo_*`` returning a random 8-char string. The
    wrapper must rely on the *name* to discriminate, not the surface
    template, because every distractor follows the needle template.
    """

    distractor = _distractor_chunk(rng)
    parts = distractor.splitlines()
    name = _rand_name(rng, family_name)
    val = _rand_value(rng)
    body = (
        f"def {name}():\n"
        f"    \"\"\"Returns the planted value for this distractor.\"\"\"\n"
        f"    return \"{val}\"\n"
    )
    insert_at = rng.randint(0, len(parts))
    parts.insert(insert_at, body)
    return "\n".join(parts)


def generate_same_form_distractors(
    *,
    n_items: int = 50,
    n_chunks: int = 6,
    seed: int = 12345,
) -> list[LongContextItem]:
    """Single-needle NIAH where every distractor chunk *also* contains a
    ``foo_*`` return-a-value function. The wrapper cannot rely on
    surface form (template-shape, identifier prefix) to pick the
    needle; it must use the specific name from the query.

    This is the cleanest test of whether the wrapper's read interface
    routes attention by content (good) vs by template (collapses).
    """

    rng = random.Random(seed)
    items: list[LongContextItem] = []
    for i in range(n_items):
        target_name = _rand_name(rng, "foo")
        target_value = _rand_value(rng)
        needle_idx = rng.randrange(n_chunks)

        # ensure no distractor accidentally re-uses target_name
        chunks: list[str] = []
        for j in range(n_chunks):
            if j == needle_idx:
                # plant the real needle
                chunks.append(_multi_needle_chunk(
                    rng, [(target_name, target_value)]
                ))
            else:
                # same-form distractor (different foo_ name)
                while True:
                    chunk = _same_form_distractor_chunk(rng, "foo")
                    if f"def {target_name}(" not in chunk:
                        break
                chunks.append(chunk)

        items.append(
            LongContextItem(
                item_id=f"same_form_distractors_{i:04d}",
                chunks=chunks,
                query=f"What value does the function {target_name} return?",
                gold=target_value,
                needle_chunk=needle_idx,
                meta={
                    "target_name": target_name,
                    "n_chunks": n_chunks,
                    "dataset": "same_form_distractors",
                },
            )
        )
    return items


def _numerical_needle_chunk(
    rng: random.Random, target_name: str, value: str
) -> str:
    distractor = _distractor_chunk(rng)
    needle = (
        f"def {target_name}():\n"
        f"    \"\"\"Returns the planted numerical constant for this synthetic case.\"\"\"\n"
        f"    return {value}\n"
    )
    parts = distractor.splitlines()
    insert_at = rng.randint(0, len(parts))
    parts.insert(insert_at, needle)
    return "\n".join(parts)


def generate_numerical_niah(
    *,
    n_items: int = 50,
    n_chunks: int = 6,
    digits: int = 5,
    seed: int = 12345,
) -> list[LongContextItem]:
    """Generate a deterministic numerical-NIAH split with a knob for answer entropy.

    Each planted function returns a positive integer with exactly ``digits``
    decimal digits (zero-padded if needed, no leading zero in the first
    position). The answer entropy is approximately ``digits × log2(10) ≈ digits × 3.32`` bits
    — this is the dial that lets us trace the wrapper's capacity curve
    between the easy categorical_niah (~2 bits) and the hard
    coding_niah (~40 bits).

    Suggested grid for capacity-curve experiments::

        digits=1  →  ~3.3 bits   (above categorical_niah's 2 bits)
        digits=2  →  ~6.6 bits
        digits=3  → ~10.0 bits
        digits=5  → ~16.6 bits
        digits=8  → ~26.6 bits
        digits=12 → ~39.9 bits   (≈ coding_niah's 40 bits)

    Same haystack distractors as ``generate_coding_niah`` so geometry
    diagnostics are comparable across datasets.
    """

    if digits < 1 or digits > 18:
        raise ValueError(f"digits must be in [1, 18], got {digits}")

    rng = random.Random(seed)
    items: list[LongContextItem] = []
    low = 10 ** (digits - 1) if digits > 1 else 0
    high = 10 ** digits

    for i in range(n_items):
        target_name = _rand_name(rng, "foo")
        value_int = rng.randrange(low, high)
        value = str(value_int).zfill(digits)
        needle_idx = rng.randrange(n_chunks)

        chunks = []
        for j in range(n_chunks):
            if j == needle_idx:
                chunks.append(_numerical_needle_chunk(rng, target_name, value))
            else:
                chunks.append(_distractor_chunk(rng))

        items.append(
            LongContextItem(
                item_id=f"numerical_niah_d{digits}_{i:04d}",
                chunks=chunks,
                query=(
                    f"What value does the function {target_name} return? "
                    "Reply with the number only and nothing else."
                ),
                gold=value,
                needle_chunk=needle_idx,
                meta={
                    "target_name": target_name,
                    "digits": digits,
                    "answer_entropy_bits": round(digits * 3.32193, 2),
                    "n_chunks": n_chunks,
                    "dataset": "numerical_niah",
                },
            )
        )
    return items


# --------------------------------------------------------------------------- #
# Real-text long-context benchmarks (paper main table)                         #
# --------------------------------------------------------------------------- #


def _chunk_text_balanced(text: str, n_chunks: int) -> list[str]:
    """Split `text` into `n_chunks` roughly-balanced pieces respecting
    paragraph boundaries. Each chunk targets `total_chars // n_chunks`
    characters; we greedy-pack paragraphs into bins.

    If the text has fewer paragraphs than `n_chunks` we fall back to a
    character split (rare for QuALITY-sized articles).
    """

    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(paras) < n_chunks:
        # Fall back to a simple character split. Pad with empty strings
        # if needed so downstream code can iterate exactly n_chunks
        # times.
        size = max(1, len(text) // n_chunks)
        chunks = [text[i * size : (i + 1) * size] for i in range(n_chunks - 1)]
        chunks.append(text[(n_chunks - 1) * size :])
        return [c if c else " " for c in chunks]

    target = max(1, len(text) // n_chunks)
    bins: list[list[str]] = [[] for _ in range(n_chunks)]
    bin_lens = [0] * n_chunks
    bin_idx = 0
    for p in paras:
        # Move on if the current bin is at/over target AND we still have
        # downstream bins to fill.
        if bin_lens[bin_idx] >= target and bin_idx < n_chunks - 1:
            bin_idx += 1
        bins[bin_idx].append(p)
        bin_lens[bin_idx] += len(p) + 2

    chunks = ["\n\n".join(b) if b else " " for b in bins]
    # Guard against empty bins (e.g. extremely uneven paragraphs).
    return [c if c.strip() else " " for c in chunks]


def generate_quality(
    *,
    n_items: int = 50,
    n_chunks: int = 8,
    seed: int = 12345,
    split: str = "validation",
    only_hard: bool = False,
) -> list[LongContextItem]:
    """QuALITY (Pang et al. 2022) adapter.

    QuALITY pairs ~5k-token articles with 4-way multiple-choice
    questions. We chunk the article into ``n_chunks`` paragraph-balanced
    pieces, frame the question with the four options labelled A/B/C/D,
    and use the letter of the correct option as ``gold``.

    The output answer space is fixed at 4 (= ``{"A", "B", "C", "D"}``)
    which means the new ``--answer-head-weight`` direct supervision
    works out of the box on this dataset (the train_smoke label
    encoder auto-builds a 4-class head).

    Args:
        n_items: number of items to keep (after the optional hard filter).
        n_chunks: how many chunks to split each article into.
        seed: shuffles the dataset deterministically before taking the
            first ``n_items``.
        split: ``train`` / ``validation`` (2086 examples).
        only_hard: keep only the ``hard=True`` examples (~50% of the
            validation split). Useful for stress testing.

    Returns:
        ``list[LongContextItem]``. ``meta`` carries the source row
        index, the raw option strings, and a copy of the ``hard`` flag.
    """

    # Lazy import so non-real-text runs don't pay the datasets cost.
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "generate_quality requires the `datasets` package; "
            "install with `pip install datasets`."
        ) from exc

    ds = load_dataset("emozilla/quality", split=split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    items: list[LongContextItem] = []
    letters = ["A", "B", "C", "D"]
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = ds[src_idx]
        if only_hard and not row.get("hard", False):
            continue

        article = str(row["article"])
        question = str(row["question"]).strip()
        options = list(row["options"])
        if len(options) != 4:
            # QuALITY is always 4-way; skip the odd malformed row.
            continue
        # emozilla/quality stores answer as a zero-based option index (0..3).
        # Subtracting one silently dropped every A-labelled example and shifted
        # the remaining labels, creating an artificial three-class task.
        gold_idx = int(row["answer"])
        if not (0 <= gold_idx < 4):
            continue
        gold_letter = letters[gold_idx]

        chunks = _chunk_text_balanced(article, n_chunks)

        # The query bundles the question + lettered options; we ask the
        # model for a single-letter answer so the wrapper's
        # answer-head sees a 4-class target.
        opt_lines = "\n".join(
            f"{letters[k]}) {options[k]}" for k in range(4)
        )
        query = (
            f"{question}\n\n"
            f"Choose the single best answer.\n"
            f"{opt_lines}\n\n"
            f"Respond with exactly one letter from {{A, B, C, D}}."
        )

        items.append(
            LongContextItem(
                item_id=f"quality_{src_idx:05d}",
                chunks=chunks,
                query=query,
                gold=gold_letter,
                needle_chunk=-1,  # real-text: no planted needle index
                meta={
                    "src_idx": src_idx,
                    "options": options,
                    "hard": bool(row.get("hard", False)),
                    "dataset": "quality",
                    "n_chunks": n_chunks,
                    "answer_choices": letters,
                },
            )
        )

    if not items:
        raise RuntimeError(
            f"generate_quality produced 0 items "
            f"(split={split!r}, only_hard={only_hard})"
        )
    return items


def generate_quality_hard(
    *, n_items: int = 50, n_chunks: int = 8, seed: int = 12345,
) -> list[LongContextItem]:
    """QuALITY hard subset (`hard=True`). Same interface as
    ``generate_quality(only_hard=True)`` but exposed under a separate
    name so it can be referenced as a dataset key in the train_smoke
    DATASETS registry without per-key parameter plumbing."""

    return generate_quality(
        n_items=n_items, n_chunks=n_chunks, seed=seed,
        split="validation", only_hard=True,
    )


def generate_longbench_v2(
    *,
    config: str = "all",
    n_items: int = 100,
    n_chunks: int = 8,
    seed: int = 12345,
    split: str = "validation",
    max_chars: int = 8_000_000,
) -> list[LongContextItem]:
    """Load the evaluation-only LongBench-v2 multiple-choice benchmark."""
    import json
    from huggingface_hub import hf_hub_download

    del split  # LongBench-v2 publishes one evaluation collection.
    path = hf_hub_download(
        repo_id="THUDM/LongBench-v2",
        filename="data.json",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    letters = ["A", "B", "C", "D"]
    items: list[LongContextItem] = []
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = rows[src_idx]
        if config not in ("all", "") and str(row.get("domain", "")).lower() != config.lower():
            continue
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        options = [str(row.get(f"choice_{letter}", "")).strip() for letter in letters]
        gold = str(row.get("answer", "")).strip().upper()
        if not context or not question or any(not option for option in options) or gold not in letters:
            continue
        context = context[:max_chars]
        option_lines = "\n".join(
            f"{letter}) {option}" for letter, option in zip(letters, options)
        )
        items.append(
            LongContextItem(
                item_id=f"lbv2_{row.get('_id', src_idx)}",
                chunks=_chunk_text_balanced(context, n_chunks),
                query=(
                    f"{question}\n\nChoose the single best answer.\n{option_lines}\n\n"
                    "Respond with exactly one letter from {A, B, C, D}."
                ),
                gold=gold,
                meta={
                    "src_idx": src_idx,
                    "options": options,
                    "dataset": "longbench_v2",
                    "domain": row.get("domain"),
                    "difficulty": row.get("difficulty"),
                    "length_bucket": row.get("length"),
                    "n_chunks": n_chunks,
                    "answer_choices": letters,
                },
            )
        )
    if not items:
        raise RuntimeError(f"generate_longbench_v2 produced 0 items (config={config!r})")
    return items


def generate_infinitebench(
    *,
    config: str = "longbook_choice_eng",
    n_items: int = 100,
    n_chunks: int = 8,
    seed: int = 12345,
    split: str = "validation",
    max_chars: int = 6_000_000,
) -> list[LongContextItem]:
    """Load an evaluation-only InfiniteBench multiple-choice task."""
    import gzip
    import json
    from huggingface_hub import hf_hub_download

    del split  # InfiniteBench task files are evaluation-only.
    path = None
    for filename in (
        f"{config}.jsonl",
        f"data/{config}.jsonl",
        f"{config}.jsonl.gz",
        "longbook_choice_eng.jsonl",
    ):
        try:
            path = hf_hub_download(
                repo_id="xinrongzhang2022/InfiniteBench",
                filename=filename,
                repo_type="dataset",
            )
            break
        except Exception:
            continue
    if path is None:
        raise RuntimeError(f"generate_infinitebench cannot fetch task {config!r}")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    letters = list("ABCDEF")
    items: list[LongContextItem] = []
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = rows[src_idx]
        context = str(row.get("context", "")).strip()
        question = str(row.get("input", "")).strip()
        options = [str(option).strip() for option in (row.get("options") or [])]
        answer = row.get("answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer).strip()
        if not context or not question or not 2 <= len(options) <= len(letters):
            continue
        if answer.upper() in letters[:len(options)]:
            gold = answer.upper()
        else:
            try:
                gold = letters[[option.lower() for option in options].index(answer.lower())]
            except ValueError:
                continue
        context = context[:max_chars]
        labels = letters[:len(options)]
        option_lines = "\n".join(
            f"{label}) {option}" for label, option in zip(labels, options)
        )
        items.append(
            LongContextItem(
                item_id=f"inf_{config}_{src_idx}",
                chunks=_chunk_text_balanced(context, n_chunks),
                query=(
                    f"{question}\n\nChoose the single best answer.\n{option_lines}\n\n"
                    "Respond with exactly one letter."
                ),
                gold=gold,
                meta={
                    "src_idx": src_idx,
                    "options": options,
                    "dataset": "infinitebench",
                    "task": config,
                    "n_chunks": n_chunks,
                    "answer_choices": labels,
                },
            )
        )
    if not items:
        raise RuntimeError(f"generate_infinitebench produced 0 items (task={config!r})")
    return items


def generate_babilong(
    *,
    config: str = "16k",
    task: str = "qa1",
    n_items: int = 100,
    n_chunks: int = 8,
    seed: int = 12345,
    split: str | None = None,
    max_chars: int = 6_000_000,
) -> list[LongContextItem]:
    """Load one BABILong length/task pair using its config-as-length schema."""
    from datasets import load_dataset  # type: ignore

    del split  # BABILong uses task names (qa1, qa2, ...) as HF splits.
    rows = load_dataset("RMT-team/babilong", config, split=task)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    dataset_name = f"babilong_{task}_{config}"
    items: list[LongContextItem] = []
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = rows[src_idx]
        context = str(row.get("input", "")).strip()
        question = str(row.get("question", "")).strip()
        gold = str(row.get("target", row.get("answer", ""))).strip()
        if not context or not question or not gold:
            continue
        items.append(
            LongContextItem(
                item_id=f"babil_{task}_{config}_{src_idx}",
                chunks=_chunk_text_balanced(context[:max_chars], n_chunks),
                query=f"{question}\n\nAnswer with the single fact.",
                gold=gold,
                meta={
                    "src_idx": src_idx,
                    "answers": [gold],
                    "dataset": dataset_name,
                    "babilong_task": task,
                    "babilong_config": config,
                    "n_chunks": n_chunks,
                },
            )
        )
    if not items:
        raise RuntimeError(f"generate_babilong produced 0 items ({config}/{task})")
    return items


def generate_hotpot_qa(
    *,
    n_items: int = 200,
    n_chunks: int = 10,
    seed: int = 12345,
    split: str = "validation",
    level: str | None = None,
) -> list[LongContextItem]:
    """HotpotQA (Yang et al. 2018) distractor adapter.

    Multi-hop QA: each item bundles a question with 10 paragraphs
    (2 supporting + 8 distractors). We treat the 10 paragraphs as
    chunks directly — no balancing needed because HotpotQA is already
    chunked at the paragraph granularity by construction. This matches
    the standard distractor-setting evaluation in Yang et al.

    Args:
        n_items: number of items to keep.
        n_chunks: target number of chunks (default 10 = native HotpotQA);
            if a row has fewer paragraphs we right-pad with " ".
        seed: shuffles deterministically before taking the first n_items.
        split: ``train`` / ``validation``.
        level: optionally filter by question difficulty
            (``easy`` / ``medium`` / ``hard``). ``None`` keeps all.

    Returns:
        ``list[LongContextItem]`` with ``gold`` = the canonical answer
        string and ``meta`` carrying source id, type, level, supporting
        paragraph titles.
    """

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "generate_hotpot_qa requires the `datasets` package; "
            "install with `pip install datasets`."
        ) from exc

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    items: list[LongContextItem] = []
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = ds[src_idx]
        if level is not None and str(row.get("level", "")).lower() != level:
            continue

        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        ctx = row["context"]
        titles = list(ctx.get("title", []))
        sentences = list(ctx.get("sentences", []))
        if not titles or not sentences:
            continue

        paragraphs = []
        for t, sents in zip(titles, sentences):
            body = " ".join(s.strip() for s in sents if s.strip())
            paragraphs.append(f"{t}.\n{body}".strip())

        # Pad / clip to exactly n_chunks chunks.
        if len(paragraphs) >= n_chunks:
            chunks = paragraphs[:n_chunks]
        else:
            chunks = paragraphs + [" "] * (n_chunks - len(paragraphs))

        query = (
            f"{question}\n\n"
            f"Answer with a short span (a few words at most)."
        )

        items.append(
            LongContextItem(
                item_id=f"hotpotqa_{src_idx:06d}",
                chunks=chunks,
                query=query,
                gold=answer,
                needle_chunk=-1,
                meta={
                    "src_idx": src_idx,
                    "type": row.get("type", ""),
                    "level": row.get("level", ""),
                    "support_titles": titles,
                    "dataset": "hotpot_qa",
                    "n_chunks": n_chunks,
                },
            )
        )

    if not items:
        raise RuntimeError(
            f"generate_hotpot_qa produced 0 items (split={split!r}, level={level!r})"
        )
    return items


def generate_musr(
    *,
    n_items: int = 200,
    n_chunks: int = 8,
    seed: int = 12345,
    split: str = "murder_mysteries",
) -> list[LongContextItem]:
    """MuSR (TAUR-Lab/MuSR) adapter — multi-step reasoning over long narratives.

    Three subsets are available as splits of the same dataset:
    ``murder_mysteries`` (250 items, ~5.5k char narrative), ``object_placements``
    (256 items), and ``team_allocation`` (250 items). Each item carries a
    short narrative, a question, a list of candidate ``choices``, and the
    correct ``answer_choice`` (string). The closed-answer-set framing makes
    this benchmark a natural fit for the wrapper's answer head (built as
    ``Linear(d, n_choices)`` automatically by ``train_smoke.py``).
    """

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "generate_musr requires the `datasets` package."
        ) from exc

    ds = load_dataset("TAUR-Lab/MuSR", split=split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    items: list[LongContextItem] = []
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = ds[src_idx]
        narrative = str(row.get("narrative", "")).strip()
        question = str(row.get("question", "")).strip()
        choices_raw = row.get("choices", "[]")
        if isinstance(choices_raw, str):
            try:
                import ast
                choices = ast.literal_eval(choices_raw)
            except Exception:
                choices = []
        else:
            choices = list(choices_raw or [])
        answer = str(row.get("answer_choice", "")).strip()
        if not narrative or not question or not choices or not answer:
            continue

        chunks = _chunk_text_balanced(narrative, n_chunks)
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"][: len(choices)]
        opt_lines = "\n".join(f"{letters[k]}) {choices[k]}" for k in range(len(choices)))
        try:
            gold_letter = letters[choices.index(answer)]
        except ValueError:
            continue
        query = (
            f"{question}\n\n"
            f"Choose the single best answer.\n"
            f"{opt_lines}\n\n"
            f"Respond with exactly one letter from "
            f"{{{', '.join(letters)}}}."
        )

        items.append(
            LongContextItem(
                item_id=f"musr_{split}_{src_idx:05d}",
                chunks=chunks,
                query=query,
                gold=gold_letter,
                needle_chunk=-1,
                meta={
                    "src_idx": src_idx,
                    "split": split,
                    "choices": choices,
                    "answer_text": answer,
                    "dataset": "musr",
                    "n_chunks": n_chunks,
                },
            )
        )

    if not items:
        raise RuntimeError(f"generate_musr produced 0 items (split={split!r})")
    return items


def generate_trivia_qa(
    *,
    n_items: int = 200,
    n_chunks: int = 8,
    seed: int = 12345,
    split: str = "validation",
    require_wiki_context: bool = True,
) -> list[LongContextItem]:
    """TriviaQA-RC (mandarjoshi/trivia_qa, ``rc`` config) adapter.

    Open-domain QA paired with retrieved Wikipedia context (``entity_pages``)
    and web search snippets (``search_results``). We use the Wikipedia
    pages as the long context source. Items without retrieved Wikipedia
    context are skipped when ``require_wiki_context=True``. Gold answer
    is the canonical ``answer.value`` plus all ``answer.aliases`` (any
    alias counts as a hit under the ``contains`` metric).
    """

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("generate_trivia_qa requires `datasets`.") from exc

    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split=split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    items: list[LongContextItem] = []
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = ds[src_idx]
        question = str(row.get("question", "")).strip()
        ans = row.get("answer", {}) or {}
        gold = str(ans.get("value", "")).strip()
        aliases = [str(a).strip() for a in (ans.get("aliases", []) or []) if str(a).strip()]
        if not question or not gold:
            continue

        ep = row.get("entity_pages", {}) or {}
        wiki = ep.get("wiki_context", []) if isinstance(ep, dict) else []
        if isinstance(wiki, list) and wiki:
            big_text = "\n\n".join(str(w) for w in wiki)
        else:
            big_text = ""

        if require_wiki_context and not big_text.strip():
            continue
        if not big_text.strip():
            sr = row.get("search_results", {}) or {}
            scx = sr.get("search_context", []) if isinstance(sr, dict) else []
            big_text = "\n\n".join(str(s) for s in scx) if isinstance(scx, list) else ""
            if not big_text.strip():
                continue

        big_text = big_text[:60_000]
        chunks = _chunk_text_balanced(big_text, n_chunks)
        query = (
            f"{question}\n\n"
            f"Answer in as few words as possible based only on the passages above."
        )

        items.append(
            LongContextItem(
                item_id=f"triviaqa_{src_idx:06d}",
                chunks=chunks,
                query=query,
                gold=gold,
                needle_chunk=-1,
                meta={
                    "src_idx": src_idx,
                    "aliases": aliases,
                    "dataset": "trivia_qa",
                    "n_chunks": n_chunks,
                },
            )
        )

    if not items:
        raise RuntimeError(f"generate_trivia_qa produced 0 items (split={split!r})")
    return items


def generate_msmarco(
    *,
    n_items: int = 200,
    n_chunks: int = 10,
    seed: int = 12345,
    split: str = "validation",
    require_answer: bool = True,
) -> list[LongContextItem]:
    """MS MARCO v2.1 reading-comprehension adapter.

    Each item bundles a query with 10 retrieved passages (one or more
    marked ``is_selected``=1 as the gold supporting passage) and 0+ short
    free-form reference answers. Passages map 1-1 to wrapper chunks
    (default ``n_chunks``=10). When ``require_answer`` is True we skip
    items whose answers are empty or "No Answer Present.".
    """

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("generate_msmarco requires `datasets`.") from exc

    ds = load_dataset("microsoft/ms_marco", "v2.1", split=split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    NO_ANSWER = {"", "no answer present.", "no answer present", "no answer"}
    items: list[LongContextItem] = []
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = ds[src_idx]
        query = str(row.get("query", "")).strip()
        ans_list = list(row.get("answers", []) or [])
        ans_list = [str(a).strip() for a in ans_list if str(a).strip()]
        if not query:
            continue
        if require_answer:
            if not ans_list or all(a.lower() in NO_ANSWER for a in ans_list):
                continue
        gold = ans_list[0] if ans_list else ""

        passages = row.get("passages", {}) or {}
        ptexts = passages.get("passage_text", []) if isinstance(passages, dict) else []
        is_sel = passages.get("is_selected", []) if isinstance(passages, dict) else []
        if not isinstance(ptexts, list) or not ptexts:
            continue
        if len(ptexts) >= n_chunks:
            chunks = [str(p).strip() or " " for p in ptexts[:n_chunks]]
        else:
            chunks = [str(p).strip() or " " for p in ptexts]
            chunks.extend([" "] * (n_chunks - len(chunks)))

        items.append(
            LongContextItem(
                item_id=f"msmarco_{src_idx:06d}",
                chunks=chunks,
                query=f"{query}\n\nAnswer in a short phrase based only on the passages above.",
                gold=gold,
                needle_chunk=-1,
                meta={
                    "src_idx": src_idx,
                    "answers": ans_list,
                    "is_selected": list(is_sel) if isinstance(is_sel, list) else [],
                    "dataset": "ms_marco",
                    "n_chunks": n_chunks,
                },
            )
        )

    if not items:
        raise RuntimeError(f"generate_msmarco produced 0 items (split={split!r})")
    return items


def generate_squad_v2(
    *,
    n_items: int = 200,
    n_chunks: int = 4,
    seed: int = 12345,
    split: str = "validation",
    require_answer: bool = True,
) -> list[LongContextItem]:
    """SQuAD v2 (rajpurkar/squad_v2) adapter.

    SQuAD passages are short (~600-1500 chars), so we set the default
    n_chunks to 4 (~150-400 chars per chunk) — this exercises the
    wrapper at short context, complementing the long-context QuALITY /
    NarrativeQA cells. ``require_answer=True`` skips the v2 "unanswerable"
    items so the held-out cell is comparable to v1 evals.
    """

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("generate_squad_v2 requires `datasets`.") from exc

    ds = load_dataset("rajpurkar/squad_v2", split=split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    items: list[LongContextItem] = []
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = ds[src_idx]
        question = str(row.get("question", "")).strip()
        context = str(row.get("context", "")).strip()
        ans = row.get("answers", {}) or {}
        ans_texts = list(ans.get("text", [])) if isinstance(ans, dict) else []
        ans_texts = [str(a).strip() for a in ans_texts if str(a).strip()]
        if not question or not context:
            continue
        if require_answer and not ans_texts:
            continue
        gold = ans_texts[0] if ans_texts else ""

        chunks = _chunk_text_balanced(context, n_chunks)
        items.append(
            LongContextItem(
                item_id=f"squad2_{src_idx:06d}",
                chunks=chunks,
                query=f"{question}\n\nAnswer with a short span from the passages above.",
                gold=gold,
                needle_chunk=-1,
                meta={
                    "src_idx": src_idx,
                    "answers": ans_texts,
                    "title": row.get("title", ""),
                    "dataset": "squad_v2",
                    "n_chunks": n_chunks,
                },
            )
        )

    if not items:
        raise RuntimeError(f"generate_squad_v2 produced 0 items (split={split!r})")
    return items


def _wikitext_haystack_paragraphs(seed: int, min_chars: int = 400) -> list[str]:
    """Pull a stream of long paragraphs from WikiText-103 to serve as the
    haystack source for procedural NIAH benchmarks (RULER-style).

    Returns a list of cleaned paragraphs (no `<unk>` tokens, no `@-@`
    glue artifacts) each >= ``min_chars`` chars. Stops once we have
    ~5_000 paragraphs (enough for thousands of unique items).
    """

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("WikiText haystack requires `datasets`.") from exc

    import re
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-v1", split="train", streaming=True,
    )
    rng = random.Random(seed)
    paras: list[str] = []
    seen = 0
    for s in ds:
        seen += 1
        t = str(s.get("text", "")).strip()
        if len(t) < min_chars:
            continue
        # Cleanup canonical WikiText artifacts.
        t = t.replace("<unk>", "")
        t = re.sub(r"\s@-@\s", "-", t)
        t = re.sub(r"\s@,@\s", ",", t)
        t = re.sub(r"\s@\.@\s", ".", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) < min_chars:
            continue
        paras.append(t)
        if len(paras) >= 5_000:
            break
        if seen > 200_000:  # bound the stream
            break
    rng.shuffle(paras)
    return paras


def generate_ruler_niah(
    *,
    n_items: int = 200,
    n_chunks: int = 8,
    seed: int = 12345,
    variant: str = "single_value",
    key_pool_size: int = 1024,
) -> list[LongContextItem]:
    """RULER-style NIAH adapter (Hsieh et al. 2024).

    Builds a synthetic ``needle in a haystack`` task following the
    structural template of the public RULER benchmark, but using
    WikiText-103 paragraphs as the haystack (instead of Paul Graham
    essays which are not on HF Hub). The plant has the canonical RULER
    form:

      ``The special magic number for {key} is {value}.``

    Variants:
      * ``single_value`` (default) — plant ONE needle, query asks for
        its value. Closest to our ``categorical_niah`` but with real
        text as filler and a structurally novel needle phrase the
        wrapper has never seen at training. THIS IS THE STRICTEST
        zero-shot transfer test we can run.
      * ``multikey_1`` — plant the same needle template with a unique
        key per item drawn from a closed pool (``key_pool_size``); the
        wrapper must localise on the key.

    Output: ``LongContextItem`` with ``needle_chunk`` set to the chunk
    index containing the planted needle (useful for the planted-needle
    probe), and ``meta`` carrying the planted (key, value) pair, the
    variant tag, and the dataset name ``ruler_niah``.
    """

    haystack = _wikitext_haystack_paragraphs(seed=seed)
    if not haystack:
        raise RuntimeError("ruler_niah: empty haystack from WikiText-103")

    rng = random.Random(seed)
    key_pool = [f"k{rng.randrange(10**6, 10**7)}" for _ in range(key_pool_size)]

    items: list[LongContextItem] = []
    for i in range(n_items):
        key = rng.choice(key_pool)
        value = f"{rng.randrange(10**5, 10**6):06d}"
        needle_chunk_idx = rng.randrange(n_chunks)
        needle = f"The special magic number for {key} is {value}."

        # Fill each chunk with one WikiText paragraph drawn without
        # replacement from the shuffled haystack so the wrapper cannot
        # rely on global frequency statistics.
        chunk_texts: list[str] = []
        for c in range(n_chunks):
            base_idx = (i * n_chunks + c) % len(haystack)
            txt = haystack[base_idx]
            if c == needle_chunk_idx:
                # Inject the needle at a deterministic-but-non-leading
                # position so the wrapper cannot just attend to the
                # chunk prefix.
                sentences = txt.split(". ")
                if len(sentences) >= 2:
                    insert_at = len(sentences) // 2
                    sentences = sentences[:insert_at] + [needle] + sentences[insert_at:]
                    txt = ". ".join(sentences)
                else:
                    txt = txt + " " + needle
            chunk_texts.append(txt)

        query = (
            f"What is the special magic number for {key}? "
            f"Answer with exactly the 6-digit number, nothing else."
        )

        items.append(
            LongContextItem(
                item_id=f"ruler_niah_{variant}_{i:05d}",
                chunks=chunk_texts,
                query=query,
                gold=value,
                needle_chunk=needle_chunk_idx,
                meta={
                    "key": key,
                    "value": value,
                    "variant": variant,
                    "dataset": "ruler_niah",
                    "n_chunks": n_chunks,
                },
            )
        )

    return items


def generate_narrativeqa(
    *,
    n_items: int = 100,
    n_chunks: int = 8,
    seed: int = 12345,
    split: str = "validation",
    max_chars: int = 60_000,
) -> list[LongContextItem]:
    """NarrativeQA (Kočiský et al. 2018) adapter.

    Each item bundles a long narrative document with a free-form
    question. We split the document into ``n_chunks`` paragraph-balanced
    pieces (same routine as QuALITY) and keep the first reference
    answer as ``gold``. The full list of references is preserved in
    ``meta['answers']`` so eval can switch to set-overlap metrics
    (F1 / contains-match against any reference) at scoring time.

    Args:
        n_items: number of items to keep.
        n_chunks: how many chunks to split each document into.
        seed: deterministic shuffle.
        split: ``train`` / ``validation`` / ``test``.
        max_chars: cap on document size (narrativeqa documents can be
            hundreds of pages — we truncate to keep wall-time bounded
            and to stay within the wrapper's training distribution).
    """

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "generate_narrativeqa requires the `datasets` package; "
            "install with `pip install datasets`."
        ) from exc

    ds = load_dataset("deepmind/narrativeqa", split=split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    items: list[LongContextItem] = []
    for src_idx in indices:
        if len(items) >= n_items:
            break
        row = ds[src_idx]
        doc = row.get("document", {}) or {}
        text = str(doc.get("text", "")).strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]

        q = row.get("question", {}) or {}
        question = str(q.get("text", "")).strip()
        if not question:
            continue

        answers_raw = row.get("answers", []) or []
        ref_strings = []
        for a in answers_raw:
            if isinstance(a, dict):
                t = str(a.get("text", "")).strip()
            else:
                t = str(a).strip()
            if t:
                ref_strings.append(t)
        if not ref_strings:
            continue

        chunks = _chunk_text_balanced(text, n_chunks)
        query = (
            f"{question}\n\n"
            f"Answer in a short phrase based only on the passages above."
        )

        items.append(
            LongContextItem(
                item_id=f"narrativeqa_{src_idx:05d}",
                chunks=chunks,
                query=query,
                gold=ref_strings[0],
                needle_chunk=-1,
                meta={
                    "src_idx": src_idx,
                    "answers": ref_strings,
                    "dataset": "narrativeqa",
                    "n_chunks": n_chunks,
                },
            )
        )

    if not items:
        raise RuntimeError(
            f"generate_narrativeqa produced 0 items (split={split!r})"
        )
    return items


def iter_items_jsonl(path: str | Path) -> Iterator[LongContextItem]:
    """Stream a previously-generated dataset from JSONL."""

    import json

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield LongContextItem(**row)


def write_items_jsonl(items: list[LongContextItem], path: str | Path) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_json()) + "\n")


# --- Phase 1 dataset stubs (defined for harness fixity) ----------------------


def load_repobench_c(split: str = "validation", limit: int | None = None):
    """RepoBench-C long-context code-completion loader. Phase 1; stub."""

    raise NotImplementedError(
        "RepoBench-C loader is a Phase 1 deliverable. "
        "For Phase 0 use generate_coding_niah."
    )


_LOCOMO_DEFAULT_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    "main/data/locomo10.json"
)


def _maybe_download_locomo(
    cache_dir: Path | str | None = None,
) -> Path:
    """Return a local Path to the LoCoMo10 JSON, fetching it once if
    necessary. The release lives at snap-research/locomo and is a
    2.7 MB single JSON with 10 conversations.

    We download to ``$LOCOMO_DATA_PATH`` if set, else
    ``$HF_HOME/locomo10.json`` if HF_HOME is set, else
    ``~/.cache/locomo10.json``. The download is idempotent.
    """

    import os
    import urllib.request

    env = os.environ.get("LOCOMO_DATA_PATH")
    if env:
        local = Path(env)
    elif cache_dir is not None:
        local = Path(cache_dir) / "locomo10.json"
    elif os.environ.get("HF_HOME"):
        local = Path(os.environ["HF_HOME"]) / "locomo10.json"
    else:
        local = Path.home() / ".cache" / "locomo10.json"

    if local.exists() and local.stat().st_size > 100_000:
        return local

    local.parent.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("LOCOMO_URL", _LOCOMO_DEFAULT_URL)
    tmp = local.with_suffix(".tmp")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    tmp.rename(local)
    return local


def generate_locomo(
    *,
    n_items: int = 200,
    n_chunks: int = 32,
    seed: int = 12345,
    split: str = "validation",
    train_fraction: float = 0.5,
    max_chars_per_session: int = 1500,
) -> list[LongContextItem]:
    """LoCoMo (Maharana et al. 2024) adapter --- very-long-term
    conversational memory benchmark.

    Each LoCoMo conversation has up to ~35 dated sessions between two
    speakers (avg 300 turns / 9k tokens), plus a list of QA pairs whose
    answers depend on remembering facts from across all sessions. We
    treat sessions as chunks (the natural unit for our wrapper):
    ``chunks = [session_1, session_2, ..., session_K]`` where each
    session string is ``"[<timestamp>] <speaker_a>: ... <speaker_b>: ..."``.

    Because LoCoMo ships as a single 10-conversation file (no native
    train/validation split), we pseudo-split deterministically by
    ``train_fraction`` of the *QA-pairs within each conversation* ---
    so train and validation see disjoint questions but share the same
    session history (which is appropriate: the wrapper is supposed to
    *encode* the history once and answer different questions over it).

    Args:
        n_items: max number of QA pairs to keep.
        n_chunks: target number of session chunks per item (default 32,
            enough to cover the ~35-session LoCoMo conversations). If the
            conversation has fewer sessions we pad with single-space
            placeholder strings; if it has MORE we keep the first n_chunks
            (chronological) and WARN, because later sessions -- and any gold
            evidence they contain -- are dropped. Raise n_chunks to cover the
            full history; `meta["truncated"]` flags items whose history was cut.
        seed: RNG for deterministic QA-pair shuffling.
        split: ``train`` or ``validation`` --- selects disjoint QA
            pairs via ``train_fraction``. (Sessions are shared.)
        train_fraction: fraction of each conversation's QA pairs that
            go to the train split. Default 0.5 = 50/50 train/eval.
        max_chars_per_session: per-session character cap to keep
            chunk lengths bounded (the wrapper uses 256-token chunks
            by default; ~1500 chars ≈ 256-400 tokens with safe headroom).

    Returns:
        list[LongContextItem] with ``gold`` = the canonical short
        answer string, ``query`` = the question, ``chunks`` = the
        session-by-session conversation history.
    """

    local = _maybe_download_locomo()
    import json as _json
    with open(local, "r", encoding="utf-8") as f:
        convos = _json.load(f)

    if split == "train":
        keep_train = True
    elif split in ("validation", "test"):
        keep_train = False
    else:
        raise ValueError(
            f"generate_locomo: unknown split {split!r} "
            "(use 'train' or 'validation')"
        )

    items: list[LongContextItem] = []
    rng = random.Random(seed)

    for conv_idx, sample in enumerate(convos):
        # ---- 1. Build the session-chunk list (chronological) ----
        # LoCoMo10 nests the dialogue under a "conversation" sub-dict
        # (top-level also has "qa", "event_summary", "observation",
        # "session_summary", "sample_id"). Older mirrors put sessions
        # at top level --- fall back to that if "conversation" is absent.
        convo = sample.get("conversation") or sample
        session_keys = sorted(
            [k for k in convo if k.startswith("session_") and not k.endswith("_date_time")
             and isinstance(convo[k], list)],
            key=lambda k: int(k.split("_", 1)[1]) if k.split("_", 1)[1].isdigit() else 0,
        )
        sessions = []
        for sk in session_keys:
            ts = convo.get(f"{sk}_date_time", "")
            turns = convo[sk] or []
            lines = []
            if ts:
                lines.append(f"[{ts}]")
            for t in turns:
                if not isinstance(t, dict):
                    continue
                spk = str(t.get("speaker", "")).strip() or "?"
                txt = str(t.get("text", "")).strip()
                if not txt:
                    continue
                lines.append(f"{spk}: {txt}")
            session_str = "\n".join(lines)
            if len(session_str) > max_chars_per_session:
                session_str = session_str[:max_chars_per_session]
            sessions.append(session_str)

        if not sessions:
            continue

        # pad / truncate to n_chunks
        truncated = len(sessions) > n_chunks
        if truncated:
            import warnings
            warnings.warn(
                f"generate_locomo: conversation {conv_idx} has {len(sessions)} sessions "
                f"> n_chunks={n_chunks}; keeping the first {n_chunks} chronologically. Later "
                f"sessions (and any gold evidence in them) are dropped -- raise n_chunks to "
                f"cover the full history.",
                stacklevel=2,
            )
        if len(sessions) >= n_chunks:
            chunks = sessions[:n_chunks]
        else:
            chunks = sessions + [" "] * (n_chunks - len(sessions))

        # ---- 2. Pull QA pairs and pseudo-split train vs validation ----
        # QA is always top-level (sibling of "conversation").
        qa = sample.get("qa", []) or []
        # filter to QA with both question + answer
        qa_clean = []
        for q in qa:
            if not isinstance(q, dict):
                continue
            question = str(q.get("question", "")).strip()
            # We score only the standard "answer" field. LoCoMo's adversarial
            # category (5) has an empty "answer" by design and is filtered out
            # here (see benchmark_metrics.py: adversarial subset is not scored).
            answer = str(q.get("answer", "")).strip()
            if not question or not answer:
                continue
            qa_clean.append(q)
        if not qa_clean:
            continue

        # deterministic shuffle then split
        idxs = list(range(len(qa_clean)))
        rng_qa = random.Random(seed + conv_idx)
        rng_qa.shuffle(idxs)
        cut = max(1, int(round(len(qa_clean) * train_fraction)))
        sel = idxs[:cut] if keep_train else idxs[cut:]

        for qi in sel:
            q = qa_clean[qi]
            question = str(q["question"]).strip()
            answer = str(q["answer"]).strip()
            category = q.get("category", -1)
            evidence = q.get("evidence", [])
            query = (
                f"{question}\n\n"
                "Answer in a short phrase based only on the conversation "
                "history above."
            )
            items.append(
                LongContextItem(
                    item_id=f"locomo_{conv_idx:02d}_q{qi:03d}",
                    chunks=chunks,
                    query=query,
                    gold=answer,
                    needle_chunk=-1,
                    meta={
                        "conv_idx": conv_idx,
                        "category": category,
                        "evidence": evidence,
                        "n_sessions": len(sessions),
                        "truncated": truncated,
                        "split": split,
                        "dataset": "locomo",
                    },
                )
            )
            if len(items) >= n_items:
                break
        if len(items) >= n_items:
            break

    if not items:
        raise RuntimeError(
            f"generate_locomo produced 0 items (split={split!r})"
        )
    rng.shuffle(items)
    return items


def _char_chunks(text: str, n: int = 1800, ov: int = 150) -> list[str]:
    """Sliding-window char chunker (whitespace-collapsed) shared by the tool-use loaders."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    step = max(1, n - ov)
    out: list[str] = []
    for s in range(0, len(text), step):
        piece = text[s : s + n].strip()
        if piece:
            out.append(piece)
        if s + n >= len(text):
            break
    return out


def _disjoint_split(rows: list, split: str, seed: int, frac: float = 0.7) -> list:
    """Seed-INDEPENDENT train/val partition (prevents train/eval LEAKAGE), then sample within the
    chosen split using the run seed. The 70/30 partition uses a FIXED seed, so the 'train' and
    'validation' index-sets are ALWAYS disjoint regardless of the run seed (the harness uses seed S
    for train and S+1 for eval); the run seed only orders items *within* a split. Non-split callers
    (split not in the known set) just get a seed-shuffled list (synthetic gens legitimately differ by seed)."""
    if str(split) not in ("train", "validation", "val", "test"):
        random.Random(seed).shuffle(rows)
        return rows
    idx = list(range(len(rows)))
    random.Random(20240613).shuffle(idx)           # FIXED partition seed => train/val always disjoint
    k = int(frac * len(idx))
    keep = set(idx[:k] if split == "train" else idx[k:])
    pool = [r for i, r in enumerate(rows) if i in keep]
    random.Random(seed).shuffle(pool)              # run seed orders WITHIN the chosen split
    return pool


def generate_apibank(
    n_items: int = 200,
    n_chunks: int = 6,
    seed: int = 42,
    split: str = "all",
    n_distractors: int = 4,
    **kwargs: Any,
) -> list[LongContextItem]:
    """API-Bank (Li et al. 2023) as a tool-use benchmark. Each item gives API documentation (the
    compressible tool docs) plus an instruction; gold = the target API name from the reference call.
    We bury the relevant API doc among distractor API docs so the model must select the right API
    from a long, noisy tool list. Source: liminghao1630/API-Bank (HF), train + test splits.
    """
    import re as _re
    from datasets import load_dataset  # type: ignore

    # API-Bank fails non-streaming generation (cross-shard schema drift) and its `test` split streams
    # empty of matches, so stream the TRAIN split + a deterministic 70/30 internal split.
    cap = max(n_items * 4, 1500)
    rows = []
    for ex in load_dataset("liminghao1630/API-Bank", split="train", streaming=True):  # streaming: avoids cross-shard CastError
        rows.append(ex)
        if len(rows) >= cap:
            break
    rows = _disjoint_split(rows, split, seed)
    rng = random.Random(seed + 7)  # for distractor sampling, within the chosen split
    docs = [str(r.get("input") or "") for r in rows if r.get("input")]
    items: list[LongContextItem] = []
    for ri, r in enumerate(rows):
        out = str(r.get("output") or "")
        m = _re.search(r"\[\s*([A-Za-z_][\w]*)\s*\(", out)
        doc = str(r.get("input") or "")
        if not m or not doc:
            continue
        gold = m.group(1)
        others = [d for d in docs if d != doc]
        dsel = rng.sample(others, min(n_distractors, len(others))) if others else []
        chunks = [doc] + dsel
        rng.shuffle(chunks)
        instr = str(r.get("instruction") or "")
        query = (instr + "\nWhich API should be called for the request? Answer with the API name only.").strip()
        fn_names = _re.findall(r'"apiCode"\s*:\s*"([^"]+)"', " ".join(chunks)) or [gold]
        items.append(
            LongContextItem(
                item_id=f"apibank_{split}_{ri}",
                chunks=chunks,
                query=query,
                gold=gold,
                needle_chunk=-1,
                meta={"dataset": "apibank", "fn_names": list(dict.fromkeys(fn_names + [gold])), "n_chunks": len(chunks)},
            )
        )
        if len(items) >= n_items:
            break
    if not items:
        raise RuntimeError(f"generate_apibank produced 0 items (split={split!r})")
    return items


def generate_toolace(
    n_items: int = 200,
    n_chunks: int = 6,
    seed: int = 42,
    split: str = "all",
    **kwargs: Any,
) -> list[LongContextItem]:
    """ToolACE (Liu et al. 2024) function-calling as a tool-use benchmark. ``system`` lists the set
    of available functions (the compressible tool docs), the first user turn is the query, and gold
    is the function name in the assistant's tool-call turn (names may contain spaces). Source:
    Team-ACE/ToolACE (HF, single train split -> deterministic 70/30 internal split).
    """
    import re as _re
    from datasets import load_dataset  # type: ignore

    rows = list(load_dataset("Team-ACE/ToolACE", split="train"))
    rows = _disjoint_split(rows, split, seed)
    items: list[LongContextItem] = []
    for ri, r in enumerate(rows):
        convs = r.get("conversations") or []
        user = next((c.get("value") for c in convs if c.get("from") == "user"), None)
        call = next(
            (
                c.get("value")
                for c in convs
                if c.get("from") in ("assistant", "gpt")
                and str(c.get("value", "")).strip().startswith("[")
                and "(" in str(c.get("value", ""))
            ),
            None,
        )
        if not user or not call:
            continue
        gm = _re.match(r"\[\s*([^()\[\]]+?)\s*\(", str(call))
        if not gm:
            continue
        gold = gm.group(1).strip()
        fn_names = [n.strip() for n in _re.findall(r"([A-Za-z][\w ]*?)\s*\(", str(call)) if n.strip()]
        system = str(r.get("system") or "")
        chunks = _char_chunks(system) or [system]
        query = str(user).strip() + "\nWhich function should be called to satisfy the request? Answer with the function name only."
        items.append(
            LongContextItem(
                item_id=f"toolace_{split}_{ri}",
                chunks=chunks,
                query=query,
                gold=gold,
                needle_chunk=-1,
                meta={"dataset": "toolace", "fn_names": fn_names or [gold], "n_chunks": len(chunks)},
            )
        )
        if len(items) >= n_items:
            break
    if not items:
        raise RuntimeError("generate_toolace produced 0 items")
    return items


def generate_bfcl(
    n_items: int = 150,
    n_chunks: int = 6,
    seed: int = 42,
    category: str = "multiple",
    split: str = "all",
    **kwargs: Any,
) -> list[LongContextItem]:
    """Berkeley Function Calling Leaderboard (BFCL) as a tool-use / agentic benchmark.

    Each record provides a user request and a list of available function specs. We treat the
    serialized function docs as the compressible *tool output* (one chunk per function), the user
    request as the query, and the target function name as the gold (tool selection). For the
    ``irrelevance`` categories none of the functions apply, so the gold is ``ABSTAIN`` (the
    do-no-harm answer: name no function). Categories: simple, multiple, irrelevance,
    live_multiple, live_irrelevance. Data: gorilla-llm/Berkeley-Function-Calling-Leaderboard (HF).
    """
    import json as _json
    from huggingface_hub import hf_hub_download  # type: ignore

    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    qf = hf_hub_download(repo, f"BFCL_v3_{category}.json", repo_type="dataset")
    recs = [_json.loads(l) for l in open(qf) if l.strip()]
    gt: dict[str, Any] = {}
    try:
        af = hf_hub_download(repo, f"possible_answer/BFCL_v3_{category}.json", repo_type="dataset")
        for l in open(af):
            if l.strip():
                a = _json.loads(l)
                gt[a["id"]] = a.get("ground_truth", [])
    except Exception:
        gt = {}  # irrelevance/relevance-only categories have no ground-truth calls

    recs = _disjoint_split(recs, split, seed)  # seed-independent 70/30 split (no train/eval leakage)
    items: list[LongContextItem] = []
    for rec in recs[:n_items]:
        funcs = rec.get("function", []) or []
        chunks: list[str] = []
        fn_names: list[str] = []
        for fn in funcs:
            nm = str(fn.get("name", ""))
            if not nm:
                continue
            fn_names.append(nm)
            props = (fn.get("parameters", {}) or {}).get("properties", {}) or {}
            pstr = ", ".join(f"{k} ({(v or {}).get('type','')})" for k, v in props.items())
            chunks.append(f"Function `{nm}`: {fn.get('description','')} Parameters: {pstr}.")
        if not chunks:
            continue
        try:
            q = rec["question"][0][0]["content"]
        except Exception:
            q = str(rec.get("question", ""))
        import os as _os
        args_mode = bool(_os.environ.get("MEM_BFCL_FULL_CALL"))  # R3: opt-in args-aware (name+args) scoring
        ans = gt.get(rec["id"])
        irr = "irrelevance" in category or not ans
        gt_call = None
        if ans and isinstance(ans, list) and ans and isinstance(ans[0], dict):
            gold = next(iter(ans[0].keys()), "")
            gt_call = ans[0]                       # {fn_name: {arg: [acceptable values]}} for args scoring
            irr = False
        else:
            gold = "ABSTAIN"
            irr = True
        if args_mode and not irr:
            query = (
                f"{q}\n\nGiven the available functions, output the single correct function CALL with its "
                f"arguments, as name(arg=value, ...). If no function applies, answer 'none'."
            )
        else:
            query = (
                f"{q}\n\nGiven the available functions, which one should be called to satisfy the "
                f"request? Answer with the function name only, or 'none' if no function applies."
            )
        items.append(
            LongContextItem(
                item_id=str(rec.get("id", f"bfcl_{len(items)}")),
                chunks=chunks,
                query=query,
                gold=gold,
                needle_chunk=-1,
                meta={
                    "dataset": f"bfcl_{category}",
                    "category": category,
                    "fn_names": fn_names,
                    "irrelevance": bool(irr),
                    "n_chunks": len(chunks),
                    "gt_call": gt_call if args_mode else None,  # present only in args mode => triggers args scoring
                },
            )
        )
    if not items:
        raise RuntimeError(f"generate_bfcl produced 0 items (category={category!r})")
    return items


def generate_hermes(
    n_items: int = 200,
    n_chunks: int = 6,
    seed: int = 42,
    split: str = "all",
    **kwargs: Any,
) -> list[LongContextItem]:
    """Nous Research Hermes function-calling (single-turn) as a tool-use benchmark — a DIFFERENT
    team/source from BFCL (Berkeley) / API-Bank (Alibaba) / ToolACE (Huawei). The ``tools`` field
    lists the available function specs (the compressible tool docs, avg ~2.4/item), the ``human``
    turn is the query, and gold = the name in the first ``<tool_call>`` of the ``gpt`` turn.
    Source: NousResearch/hermes-function-calling-v1 (config ``func_calling_singleturn``).
    """
    import json as _json
    import re as _re
    from datasets import load_dataset  # type: ignore

    rows = list(load_dataset("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train"))
    rows = _disjoint_split(rows, split, seed)
    items: list[LongContextItem] = []
    for ri, r in enumerate(rows):
        convs = r.get("conversations") or []
        human = next((c.get("value") for c in convs if c.get("from") == "human"), None)
        gpt = next((c.get("value") for c in convs if c.get("from") in ("gpt", "assistant")), None)
        if not human or not gpt:
            continue
        gm = _re.search(r'\{\s*"name"\s*:\s*"([^"]+)"', str(gpt))
        if not gm:
            continue
        gold = gm.group(1).strip()
        try:
            tools = _json.loads(r.get("tools") or "[]")
        except Exception:
            tools = []
        chunks: list[str] = []
        fn_names: list[str] = []
        for t in tools:
            fn = (t or {}).get("function") or t or {}
            nm = str(fn.get("name", ""))
            if not nm:
                continue
            fn_names.append(nm)
            props = ((fn.get("parameters") or {}).get("properties") or {})
            pstr = ", ".join(f"{k} ({(v or {}).get('type','')})" for k, v in props.items())
            chunks.append(f"Function `{nm}`: {fn.get('description','')} Parameters: {pstr}.")
        if not chunks:
            continue
        query = str(human).strip() + "\nWhich function should be called to satisfy the request? Answer with the function name only."
        items.append(
            LongContextItem(
                item_id=f"hermes_{split}_{ri}",
                chunks=chunks,
                query=query,
                gold=gold,
                needle_chunk=-1,
                meta={"dataset": "hermes", "fn_names": fn_names or [gold], "n_chunks": len(chunks)},
            )
        )
        if len(items) >= n_items:
            break
    if not items:
        raise RuntimeError("generate_hermes produced 0 items")
    return items


def generate_glaive(
    n_items: int = 200,
    n_chunks: int = 6,
    seed: int = 42,
    split: str = "all",
    **kwargs: Any,
) -> list[LongContextItem]:
    """Glaive function-calling v2 as a tool-use benchmark — a DIFFERENT team/source again. The
    ``system`` field declares the available function(s) as JSON (the compressible tool docs;
    single-function-dominant, so lower selection difficulty), the first USER turn is the query,
    and gold = the name in the first ``<functioncall>`` of the assistant turn, else ``ABSTAIN``
    (the assistant declines — a built-in irrelevance signal). Source: glaiveai/glaive-function-calling-v2.
    """
    import json as _json
    import re as _re
    from datasets import load_dataset  # type: ignore

    def _scan_objs(s: str) -> list[dict]:
        dec = _json.JSONDecoder()
        objs: list[dict] = []
        i, n = 0, len(s)
        while i < n:
            c = s.find("{", i)
            if c < 0:
                break
            try:
                o, e = dec.raw_decode(s, c)
                objs.append(o)
                i = e
            except Exception:
                i = c + 1
        return objs

    cap = max(n_items * 8, 4000)
    seen_q: set = set()
    rows = []
    gds = load_dataset("glaiveai/glaive-function-calling-v2", split="train[:20000]")  # non-streaming => cached => offline-safe
    for ex in gds:
        chat = str(ex.get("chat") or "")
        um = _re.search(r"USER:\s*(.*?)(?:\nASSISTANT:|$)", chat, _re.S)
        q = (um.group(1).strip() if um else "")
        if not q or q in seen_q:  # Glaive repeats templated user turns => dedup so train/eval can't share content
            continue
        seen_q.add(q)
        rows.append(ex)
        if len(rows) >= cap:
            break
    rows = _disjoint_split(rows, split, seed)
    items: list[LongContextItem] = []
    for ri, r in enumerate(rows):
        system = str(r.get("system") or "")
        chat = str(r.get("chat") or "")
        chunks: list[str] = []
        fn_names: list[str] = []
        for o in _scan_objs(system):
            nm = str((o or {}).get("name", ""))
            if not nm:
                continue
            fn_names.append(nm)
            props = ((o.get("parameters") or {}).get("properties") or {})
            pstr = ", ".join(f"{k} ({(v or {}).get('type','')})" for k, v in props.items())
            chunks.append(f"Function `{nm}`: {o.get('description','')} Parameters: {pstr}.")
        if not chunks:
            continue
        um = _re.search(r"USER:\s*(.*?)(?:\nASSISTANT:|$)", chat, _re.S)
        query = (um.group(1).strip() if um else "").strip()
        if not query:
            continue
        fm = _re.search(r'<functioncall>\s*\{\s*"name"\s*:\s*"([^"]+)"', chat)
        gold = fm.group(1).strip() if fm else "ABSTAIN"
        irr = gold == "ABSTAIN"
        query = query + "\nWhich function should be called to satisfy the request? Answer with the function name only, or 'none' if no function applies."
        items.append(
            LongContextItem(
                item_id=f"glaive_{split}_{ri}",
                chunks=chunks,
                query=query,
                gold=gold,
                needle_chunk=-1,
                meta={"dataset": "glaive", "fn_names": fn_names or [gold], "irrelevance": bool(irr), "n_chunks": len(chunks)},
            )
        )
        if len(items) >= n_items:
            break
    if not items:
        raise RuntimeError("generate_glaive produced 0 items")
    return items


def generate_rca(
    n_items: int = 300,
    n_chunks: int = 6,
    seed: int = 42,
    source: str = "openrca",
    n_distractors: int = 2,  # 0->full 0.58, 2->0.43, 6->0.07 (drowns signal); 2 keeps signal + some length
    n_options: int = 6,
    split: str = "all",
    **kwargs: Any,
) -> list[LongContextItem]:
    """Root-cause analysis (RCA) as a long-context agentic benchmark.

    Each case is one incident: an SLO/anomaly report plus per-API or per-metric statistics from a
    microservice system. We chunk that telemetry and bury it among distractor incident reports
    (telemetry from *other* incidents), so the model must locate the root-cause service in a long,
    noisy context (the realistic agentic RCA setting). gold = the primary root-cause service; meta
    carries the full structured reference (root_cause_services, root_cause text, evidence
    filenames) so the RCA scorer can compute primary-service match / root_cause F1 / evidence
    overlap. ``source`` is "openrca" (lincyaw/openrca2-v1-500), "rcaeval" (RMIT RCAEval, Zenodo
    14590730), or a direct path to a built cases.jsonl. Cases are produced by the rca-demo
    builders; the path is resolved from MEM_RCA_<SOURCE> or a default under ~/datasets.
    """
    import os
    import json as _json

    defaults = {
        "openrca": os.environ.get("MEM_RCA_OPENRCA", "data/openrca_built/cases.jsonl"),
        "rcaeval": os.environ.get("MEM_RCA_RCAEVAL", "data/rcaeval_built/cases.jsonl"),
    }
    path = defaults.get(source, source)
    if not os.path.exists(path):
        raise FileNotFoundError(f"generate_rca: cases.jsonl not found for source={source!r} at {path}")
    if "MEM_RCA_NDIS" in os.environ:  # experiment knob: vary distractor count without code churn
        n_distractors = int(os.environ["MEM_RCA_NDIS"])
    cases = [_json.loads(l) for l in open(path) if l.strip()]

    def _report(c: dict) -> str:
        return str((c.get("input") or {}).get("ad_report_excerpt") or "")

    def _chunk(text: str, n: int = 1800, ov: int = 150) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        step = max(1, n - ov)
        out: list[str] = []
        for s in range(0, len(text), step):
            piece = text[s : s + n].strip()
            if piece:
                out.append(piece)
            if s + n >= len(text):
                break
        return out

    # Per-system service vocabulary (candidate options), from ALL cases before sampling. RCA as
    # free-form naming is too hard for a frozen base in completion mode (~0 accuracy, no dynamic
    # range), so we frame it as root-cause-service SELECTION (multiple choice over same-system
    # candidate services), scored by mc_loglik like quality/musr. This is also a standard RCA
    # evaluation (root-cause localization as ranking over candidate services).
    sysvocab: dict[str, set] = {}
    for c in cases:
        sysm = str((c.get("metadata") or {}).get("system", "?"))
        for s in (c.get("reference") or {}).get("root_cause_services") or []:
            if s:
                sysvocab.setdefault(sysm, set()).add(str(s))
    allsvc = sorted({s for v in sysvocab.values() for s in v})

    cases = _disjoint_split(cases, split, seed)  # seed-independent 70/30 split (no train/eval leakage)
    rng = random.Random(seed + 7)  # option/distractor sampling, within the chosen split
    pool = [r for r in (_report(c) for c in cases) if r]
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    items: list[LongContextItem] = []
    for ci, c in enumerate(cases[:n_items]):
        rep = _report(c)
        ref = c.get("reference") or {}
        gold_svc = str(ref.get("primary_root_cause") or "").strip()
        if not rep or not gold_svc:
            continue
        # candidate options: gold + same-system distractor services (backfill from all services)
        sysm = str((c.get("metadata") or {}).get("system", "?"))
        cand = [s for s in sysvocab.get(sysm, set()) if s.lower() != gold_svc.lower()]
        if len(cand) < n_options - 1:
            cand = list(dict.fromkeys(cand + [s for s in allsvc if s.lower() != gold_svc.lower()]))
        distract_svc = rng.sample(cand, min(n_options - 1, len(cand))) if cand else []
        options = [gold_svc] + distract_svc
        rng.shuffle(options)
        gi = options.index(gold_svc)
        let = letters[: len(options)]
        # telemetry: target incident (clearly marked) + distractor incidents (labeled noise)
        real = _chunk(rep)
        if not real:
            continue
        real[0] = "INCIDENT UNDER INVESTIGATION:\n" + real[0]
        others = [p for p in pool if p != rep]
        dsel = rng.sample(others, min(n_distractors, len(others))) if others else []
        distract = [
            "UNRELATED TELEMETRY (another incident, context only):\n" + _chunk(o)[0]
            for o in dsel if _chunk(o)
        ]
        all_chunks = real + distract
        optlines = "\n".join(f"{let[i]}) {options[i]}" for i in range(len(options)))
        query = (
            "The report marked 'INCIDENT UNDER INVESTIGATION' describes an incident in a microservice "
            "system; any other reports are unrelated context. Which microservice is the root cause of "
            f"the incident under investigation?\n{optlines}\n\nRespond with exactly one letter from "
            f"{{{', '.join(let)}}}."
        )
        files = [str(n) for n, _ in ((c.get("stats") or {}).get("rca_retrieval") or {}).get("filenames", [])]
        items.append(
            LongContextItem(
                item_id=str(c.get("case_id", f"rca_{ci}")),
                chunks=all_chunks,
                query=query,
                gold=let[gi],
                needle_chunk=-1,
                meta={
                    "dataset": f"rca_{source}",
                    "options": options,
                    "answer_choices": let,
                    "reference": ref,
                    "rca_filenames": files,
                    "n_chunks": len(all_chunks),
                    "source": source,
                },
            )
        )
    if not items:
        raise RuntimeError(f"generate_rca produced 0 items (source={source!r})")
    return items
