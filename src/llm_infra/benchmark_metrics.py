"""Canonical / official metrics for each public benchmark.

The wrapper paper must report the metric the benchmark's authors defined,
not a custom one. Each function below implements the official metric of
its benchmark as faithfully as possible; the function signatures are
uniform so the eval harness can dispatch on ``item.meta["dataset"]`` and
the offline rescorer can be applied to any completed cell's
``predictions.jsonl``.

Coverage (2026-06-03 v0):

  benchmark            | metric(s) returned                | source
  ---------------------|-----------------------------------|---------------------------
  quality              | accuracy_letter                   | Pang et al. 2022 (closed-MC)
  quality_hard         | accuracy_letter                   | same
  musr_mm / op / ta    | accuracy_letter                   | Sprague et al. 2024 (closed-MC)
  hotpot_qa            | squad_em, squad_f1                | Yang et al. 2018 (HotpotQA `hotpot_evaluate_v1.py`)
  babilong_*            | answer_accuracy                   | Kuratov et al. 2024 (`compare_answers`)
  squad_v2             | squad_em, squad_f1                | Rajpurkar et al. 2018 (SQuAD v2 eval script)
  trivia_qa            | squad_em (over aliases),          | Joshi et al. 2017 (TriviaQA eval script;
                       | squad_f1 (over aliases)           |  SQuAD-style normalization)
  narrativeqa          | rouge_l, bleu_4                   | Kočiský et al. 2018 (BLEU-1, BLEU-4,
                       |                                   |  METEOR, ROUGE-L; ROUGE-L is the
                       |                                   |  most-commonly-reported headline.)
  ms_marco             | rouge_l                           | Bajaj et al. 2018 (MS MARCO QA NLG;
                       |                                   |  primary metric is ROUGE-L per
                       |                                   |  https://microsoft.github.io/msmarco/)
  ruler_niah           | exact_value_match                 | Hsieh et al. 2024 (RULER NIAH)
  locomo               | squad_f1, squad_em, rouge_l       | Maharana et al. 2024 (Table 3 reports
                       |                                   |  F1; multi-session conversational QA)

All implementations are pure-stdlib so they can run on any pod or
local machine without `evaluate` / `sacrebleu` / `rouge_score`
installed.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Iterable


# --------------------------------------------------------------------------
# SQuAD-style normalization (used by SQuAD / SQuAD v2 / HotpotQA /
# TriviaQA / many derivatives). Lifted verbatim from the official
# SQuAD evaluation script (Rajpurkar et al. 2016, eval_squad.py).
# --------------------------------------------------------------------------


def _squad_normalize(s: str) -> str:
    """Lower-case, remove punctuation, remove articles, collapse spaces."""

    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _squad_f1(prediction: str, gold: str) -> float:
    pred_toks = _squad_normalize(prediction).split()
    gold_toks = _squad_normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def _squad_em(prediction: str, gold: str) -> float:
    return float(_squad_normalize(prediction) == _squad_normalize(gold))


def _squad_em_over_refs(prediction: str, refs: Iterable[str]) -> float:
    """Max EM over a list of gold references (TriviaQA / NarrativeQA convention)."""

    refs = [r for r in refs if r]
    if not refs:
        return 0.0
    return max(_squad_em(prediction, r) for r in refs)


def _squad_f1_over_refs(prediction: str, refs: Iterable[str]) -> float:
    refs = [r for r in refs if r]
    if not refs:
        return 0.0
    return max(_squad_f1(prediction, r) for r in refs)


# --------------------------------------------------------------------------
# Letter-extraction (closed multiple-choice). Used by QuALITY and MuSR.
# --------------------------------------------------------------------------


_LETTER_AT_START_RE = re.compile(
    r"""^[\s\(\[\"']*           # optional opening junk
        (?:(?:THE\s+)?ANSWER\s+IS\s*|ANSWER\s*[:\-]?\s*|OPTION\s*[:\-]?\s*)?
        \(?\s*([A-Z])\s*[\)\.\,\:\;\]\"'\s]   # letter followed by terminator
    """,
    re.VERBOSE,
)

# Standalone letter anywhere in the text, surrounded by word boundaries
# OR by parens/quotes/period (e.g. ``"X"``, ``(X)``, ``X.``). Used as a
# more-lenient second pass to catch mid-sentence answers like
# ``"I think the answer is C."`` without false-matching ``"Apple"``.
_LETTER_STANDALONE_RE = re.compile(
    r"""(?:^|[\s\(\[\"'])      # word-boundary on the left
        \(?([A-Z])\)?           # the letter, possibly parenthesised
        (?=[\s\.\,\)\]\"':;]|$) # word-boundary on the right
    """,
    re.VERBOSE,
)


def _extract_letter(prediction: str, allowed: str = "ABCD") -> str:
    """Return the first standalone capital letter in ``allowed`` parsed
    from common multiple-choice answer formats. Tries two passes:

    1. **Strict** — the letter is at the START of the response, possibly
       behind ``Answer:`` / ``The answer is`` / ``Option`` prefixes,
       and is followed by a terminator (``.``, ``)``, etc.). This is
       the canonical SQuAD-style multiple-choice parser.
    2. **Lenient** — the letter appears anywhere in the response as a
       standalone token with word boundaries on both sides (e.g.
       ``"I think the answer is C."``, ``"... so I pick (B)."``). This
       rescues frozen-base outputs that don't follow the ``answer-first``
       format perfectly, while still rejecting letters embedded inside
       a content word like ``"Apple"``.

    Returns ``""`` if neither pass finds a letter in ``allowed``.
    """

    if not prediction:
        return ""
    up = (prediction + " ").upper()
    m = _LETTER_AT_START_RE.match(up)
    if m and m.group(1) in allowed:
        return m.group(1)
    for m2 in _LETTER_STANDALONE_RE.finditer(up):
        if m2.group(1) in allowed:
            return m2.group(1)
    return ""


def _letter_accuracy(
    prediction: str,
    gold: str,
    *,
    allowed: str = "ABCD",
    choices: list[str] | None = None,
) -> float:
    """Closed-MC accuracy. Tries three increasingly lenient matchers:

    1.  Parse a standalone letter from common answer formats and
        compare to ``gold`` (the canonical accuracy_letter metric).
    2.  Fall back to choice-text matching: if the model output
        contains the gold choice's text (case-insensitive) but
        contains none of the other choices' text, count as a hit.
        This rescues correct-content / wrong-format answers (e.g.
        ``"Mackenzie"`` for MuSR gold letter ``"A"``).
    3.  If gold is itself raw text (e.g. ``"Mackenzie"`` rather than
        ``"A"``), compare substring presence directly.
    """

    pred_low = prediction.lower()
    g = gold.strip().upper()
    if g in allowed:
        extracted = _extract_letter(prediction, allowed=allowed)
        if extracted == g:
            return 1.0
        # Choice-text fallback: which choice text is present in pred?
        if choices and g in allowed:
            gold_idx = allowed.index(g)
            if 0 <= gold_idx < len(choices):
                gold_text = str(choices[gold_idx]).strip().lower()
                gold_hit = bool(gold_text) and gold_text in pred_low
                # Reject if pred ALSO contains another choice's text
                # (ambiguous — credit only when the model unambiguously
                # named the gold).
                other_hit = any(
                    str(ch).strip().lower() in pred_low
                    for i, ch in enumerate(choices)
                    if i != gold_idx and str(ch).strip()
                )
                if gold_hit and not other_hit:
                    return 1.0
        return 0.0
    # Gold isn't a letter → raw-text substring match (legacy path).
    return float(g.lower() in pred_low)


# --------------------------------------------------------------------------
# ROUGE-L (LCS-based) — pure-Python implementation matching the
# rouge_score package's F-score variant (which is what NarrativeQA and
# MS MARCO scoreboards report).
# --------------------------------------------------------------------------


def _rouge_l_f(prediction: str, gold: str) -> float:
    pred_toks = prediction.lower().split()
    gold_toks = gold.lower().split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    m, n = len(pred_toks), len(gold_toks)
    # LCS table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_toks[i] == gold_toks[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


def _rouge_l_f_over_refs(prediction: str, refs: Iterable[str]) -> float:
    refs = [r for r in refs if r]
    if not refs:
        return 0.0
    return max(_rouge_l_f(prediction, r) for r in refs)


# --------------------------------------------------------------------------
# BLEU-4 sentence-level (n-gram precision with brevity penalty), matching
# the NLTK `corpus_bleu` smoothing-method-1 behaviour used by the
# NarrativeQA original paper.
# --------------------------------------------------------------------------


def _ngrams(toks: list[str], n: int) -> Counter:
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def _bleu_n(prediction: str, golds: list[str], n: int = 4) -> float:
    """Sentence-level BLEU-n over multiple references with add-one
    smoothing (smoothing-method-1 from Chen & Cherry 2014). Returns a
    value in [0, 1].
    """

    pred_toks = prediction.lower().split()
    if not pred_toks:
        return 0.0
    gold_tok_lists = [g.lower().split() for g in golds if g]
    if not gold_tok_lists:
        return 0.0

    # Per-n precision with clipping.
    precisions: list[float] = []
    for k in range(1, n + 1):
        if len(pred_toks) < k:
            precisions.append(0.0)
            continue
        p_ngrams = _ngrams(pred_toks, k)
        # max-count clipping across refs (per BLEU spec)
        max_ref_ngrams: Counter = Counter()
        for g_toks in gold_tok_lists:
            g_ngrams = _ngrams(g_toks, k)
            for ng, c in g_ngrams.items():
                max_ref_ngrams[ng] = max(max_ref_ngrams[ng], c)
        clipped = sum(min(c, max_ref_ngrams[ng]) for ng, c in p_ngrams.items())
        total = sum(p_ngrams.values())
        if total == 0:
            precisions.append(0.0)
        else:
            # add-one smoothing for higher orders to avoid log(0)
            precisions.append((clipped + (1.0 if k > 1 and clipped == 0 else 0.0))
                              / (total + (1.0 if k > 1 and clipped == 0 else 0.0)))

    if any(p == 0.0 for p in precisions):
        return 0.0

    # Geometric mean
    import math
    log_p = sum(math.log(p) for p in precisions) / len(precisions)

    # Brevity penalty
    pred_len = len(pred_toks)
    ref_len = min((len(g) for g in gold_tok_lists),
                  key=lambda x: (abs(x - pred_len), x))
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))
    return bp * math.exp(log_p)


# --------------------------------------------------------------------------
# Per-benchmark dispatch. Returns a dict of metric_name -> value.
# --------------------------------------------------------------------------


def score_item(
    *,
    dataset: str,
    prediction: str,
    gold: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute the canonical metric(s) for one (prediction, gold) pair
    of the given ``dataset``.

    Returns a flat dict mapping metric_name -> value in [0, 1].
    Metric names are stable strings the eval / aggregator code can rely
    on (no abbreviations beyond the standard ones used in the
    benchmark's own paper).
    """

    meta = meta or {}
    refs: list[str]
    if dataset in {
        "quality",
        "quality_hard",
        "longbench_v2",
        "infinitebench",
    }:
        choices = list(meta.get("options") or [])
        allowed = "ABCDEFGH"[:len(choices)] if choices else "ABCD"
        return {"accuracy_letter": _letter_accuracy(
            prediction, gold, allowed=allowed, choices=choices)}
    if dataset in ("musr_mm", "musr_op", "musr_ta", "musr"):
        # MuSR has 2- or 3-way MC; allowed letters depend on the split.
        choices = meta.get("choices") or []
        n = max(2, len(choices)) if choices else 4
        allowed = "ABCDEFGH"[:n]
        return {"accuracy_letter": _letter_accuracy(
            prediction, gold, allowed=allowed, choices=list(choices))}
    if dataset == "hotpot_qa":
        return {
            "squad_em": _squad_em(prediction, gold),
            "squad_f1": _squad_f1(prediction, gold),
        }
    if dataset.startswith("babilong"):
        # Official BABILong evaluation lower-cases the strings, keeps only the
        # first sentence, strips accidental generated examples, and accepts the
        # answer when the target occurs in the remaining text.
        output = (prediction or "").lower().split(".", 1)[0]
        output = output.split("<context>", 1)[0].split("<example>", 1)[0]
        return {"answer_accuracy": float(gold.lower() in output)}
    if dataset == "squad_v2":
        refs = list(meta.get("answers") or [gold])
        return {
            "squad_em": _squad_em_over_refs(prediction, refs),
            "squad_f1": _squad_f1_over_refs(prediction, refs),
        }
    if dataset == "trivia_qa":
        refs = list(meta.get("aliases") or [gold])
        if gold and gold not in refs:
            refs = [gold] + refs
        return {
            "squad_em": _squad_em_over_refs(prediction, refs),
            "squad_f1": _squad_f1_over_refs(prediction, refs),
        }
    if dataset == "narrativeqa":
        refs = list(meta.get("answers") or [gold])
        if gold and gold not in refs:
            refs = [gold] + refs
        return {
            "rouge_l": _rouge_l_f_over_refs(prediction, refs),
            "bleu_4": _bleu_n(prediction, refs, n=4),
        }
    if dataset == "ms_marco":
        refs = list(meta.get("answers") or [gold])
        if gold and gold not in refs:
            refs = [gold] + refs
        return {
            "rouge_l": _rouge_l_f_over_refs(prediction, refs),
            "bleu_1": _bleu_n(prediction, refs, n=1),
        }
    if dataset == "locomo":
        # LoCoMo headline metric in Maharana et al. 2024 (Table 3) is
        # F1 over the canonical short answer; we also report SQuAD-EM
        # and ROUGE-L since LoCoMo answers can be either short tokens
        # ("7 May 2023") or full phrases ("counseling or mental health
        # for Transgender people"). We don't yet score the adversarial
        # subset (category=5, answer field empty): items where gold is
        # empty are filtered upstream by the loader.
        return {
            "squad_em": _squad_em(prediction, gold),
            "squad_f1": _squad_f1(prediction, gold),
            "rouge_l":  _rouge_l_f(prediction, gold),
        }
    if dataset == "ruler_niah":
        # The planted value is a fixed 6-digit numeric string. Official
        # RULER eval is exact string match against the planted value;
        # we strip whitespace and look for the value as a substring to
        # be lenient about decoder formatting around it.
        return {"exact_value_match": float(gold in prediction.strip())}

    if dataset.startswith(("bfcl", "apibank", "toolace", "hermes", "glaive")):
        # Tool selection (BFCL / API-Bank / ToolACE): did the model name the correct tool/API/
        # function? For BFCL irrelevance (gold == "ABSTAIN"), the do-no-harm answer is to name NONE.
        # Match the function name as a whole token (word boundaries) rather than a loose
        # substring, so common-word names (e.g. "search") do not false-positive on prose
        # and a name is not matched as a prefix of a different function (foo vs foo_bar).
        import re as _re
        pred = (prediction or "").lower()
        def _named(nm: str) -> bool:
            nm = str(nm).lower().strip()
            if not nm:
                return False
            return _re.search(r"(?<!\w)" + _re.escape(nm) + r"(?!\w)", pred) is not None
        names = meta.get("fn_names") or []
        if gold == "ABSTAIN":
            return {"tool_acc": float(not any(_named(n) for n in names))}
        out = {"tool_acc": float(_named(gold))}
        gtc = meta.get("gt_call")  # R3 (args-aware PROXY): name correct AND each gold arg-value present in the call
        if gtc and gold in gtc:
            spec = gtc.get(gold) or {}
            sat = []
            for _arg, vals in spec.items():
                vlist = vals if isinstance(vals, list) else [vals]
                nonempty = [str(v).lower().strip() for v in vlist if str(v).strip() not in ("", "none")]
                if not nonempty:  # optional / empty-acceptable arg => trivially satisfied
                    sat.append(True)
                    continue
                sat.append(any(_re.search(r"(?<!\w)" + _re.escape(v) + r"(?!\w)", pred) for v in nonempty))
            arg_frac = (sum(sat) / len(sat)) if sat else 1.0
            out["tool_arg_frac"] = float(out["tool_acc"]) * arg_frac          # name * fraction of args present (soft)
            out["tool_call_acc"] = float(out["tool_acc"] == 1.0 and arg_frac == 1.0)  # strict full-call match
        return out

    if dataset.startswith("rca"):
        # Root-cause analysis. Ported from rca-demo evaluate_predictions.py, adapted to free-text
        # predictions: primary_service_match (does a gold root-cause service appear as a token in
        # the answer; the headline metric, analogous to BFCL tool_acc), service_recall (fraction of
        # gold services named), root_cause_f1 (token-F1 vs the reference root-cause text), and
        # evidence_overlap (fraction of gold evidence filenames named). meta carries the structured
        # reference + the evidence filename list.
        import re as _re
        ref = meta.get("reference") or {}
        predl = (prediction or "").lower()
        gold_services = [str(s).lower().strip() for s in (ref.get("root_cause_services") or []) if s]
        if not gold_services and ref.get("primary_root_cause"):
            gold_services = [str(ref["primary_root_cause"]).lower().strip()]

        def _tokmatch(nm: str) -> bool:
            nm = nm.strip()
            return bool(nm) and _re.search(r"(?<![\w-])" + _re.escape(nm) + r"(?![\w-])", predl) is not None

        primary = float(any(_tokmatch(s) for s in gold_services)) if gold_services else 0.0
        svc_recall = (sum(_tokmatch(s) for s in gold_services) / len(gold_services)) if gold_services else 0.0
        _stop = {"the", "and", "for", "with", "was", "service", "root", "cause", "impact",
                 "cascading", "microservice", "system", "the", "due", "from"}

        def _toks(t: str) -> set:
            t = _re.sub(r"[^a-z0-9_./-]+", " ", str(t).lower())
            return {w for w in t.split() if len(w) > 2 and w not in _stop}

        pt, rt = _toks(prediction or ""), _toks(ref.get("root_cause") or "")
        if not pt and not rt:
            rcf1 = 1.0
        elif not pt or not rt:
            rcf1 = 0.0
        else:
            ov = len(pt & rt)
            rcf1 = 0.0 if ov == 0 else (lambda p, r: 2 * p * r / (p + r))(ov / len(pt), ov / len(rt))
        files = [str(n).lower() for n in (meta.get("rca_filenames") or [])][:10]
        ev = 1.0 if not files else sum(1 for n in files if n and n in predl) / len(files)
        return {"primary_service_match": primary, "service_recall": svc_recall,
                "root_cause_f1": rcf1, "evidence_overlap": ev}

    # Synthetic NIAH variants we built ourselves (categorical_niah,
    # numerical_niah, coding_niah, multi_needle, same_form_distractors) —
    # they are NOT public benchmarks, but we still report a canonical
    # number for them: strict substring match (consistent with the
    # original NIAH eval protocol of Kamradt 2024).
    return {"niah_substring": float(gold.strip() in prediction)}


def aggregate(per_item: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each per-item metric. Missing metrics are skipped (so a
    cell that mixed datasets won't crash, but practically each cell is
    one dataset)."""

    if not per_item:
        return {}
    all_keys: set[str] = set()
    for r in per_item:
        all_keys.update(r.keys())
    out: dict[str, float] = {}
    for k in all_keys:
        vals = [r[k] for r in per_item if k in r and isinstance(r[k], (int, float))]
        if vals:
            out[k] = sum(vals) / len(vals)
    out["n"] = float(len(per_item))
    return out


__all__ = [
    "score_item",
    "aggregate",
    "_squad_normalize",
    "_squad_em",
    "_squad_f1",
    "_letter_accuracy",
    "_rouge_l_f",
    "_bleu_n",
]
