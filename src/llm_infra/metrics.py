"""Metrics for Phase 0 / Phase 1 evaluations.

Designed to be cheap and string-only so the harness can run on CPU without a
verifier model. Code-execution scoring is a Phase 1 add-on (RepoBench-C).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if prediction.strip() == gold.strip() else 0.0


def contains_match(prediction: str, gold: str) -> float:
    """1.0 if gold appears as substring in prediction (case-insensitive).

    This is the right metric for the coding-NIAH probe, where any prefix /
    suffix added by the model around the gold value should not be penalised.
    """

    return 1.0 if gold.strip().lower() in prediction.lower() else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_toks = _tokenize(prediction)
    gold_toks = _tokenize(gold)
    if not pred_toks or not gold_toks:
        return 1.0 if pred_toks == gold_toks else 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_toks)
    recall = num_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


@dataclass
class MetricsSummary:
    n: int
    exact_match: float
    contains_match: float
    token_f1: float
    input_tokens_mean: float
    input_tokens_p95: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "exact_match": self.exact_match,
            "contains_match": self.contains_match,
            "token_f1": self.token_f1,
            "input_tokens_mean": self.input_tokens_mean,
            "input_tokens_p95": self.input_tokens_p95,
        }


def summarise(predictions: list[str], golds: list[str], input_token_counts: list[int]) -> MetricsSummary:
    if not (len(predictions) == len(golds) == len(input_token_counts)):
        raise ValueError("predictions / golds / input_token_counts length mismatch")
    n = len(predictions)
    if n == 0:
        return MetricsSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    em = sum(exact_match(p, g) for p, g in zip(predictions, golds)) / n
    cm = sum(contains_match(p, g) for p, g in zip(predictions, golds)) / n
    f1 = sum(token_f1(p, g) for p, g in zip(predictions, golds)) / n
    sorted_tokens = sorted(input_token_counts)
    mean_tokens = sum(sorted_tokens) / n
    p95_idx = max(0, int(round(0.95 * (n - 1))))
    return MetricsSummary(
        n=n,
        exact_match=em,
        contains_match=cm,
        token_f1=f1,
        input_tokens_mean=mean_tokens,
        input_tokens_p95=float(sorted_tokens[p95_idx]),
    )
