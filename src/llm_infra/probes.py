"""Probes: cheap, repeatable evaluations that target a specific failure mode.

Phase 0 probes:
- PlantedNeedleProbe:   accuracy on the planted-answer task per item.
- EvidenceRetentionProbe: per-bucket accuracy when the needle is in an
  early / middle / late chunk; reveals position-dependent forgetting.

Probes consume the same FourBaselineHarness output (`predictions.jsonl` +
`case_scores.jsonl`) so they can be re-computed from on-disk eval runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from llm_infra.datasets import LongContextItem
from llm_infra.metrics import contains_match


class Probe(Protocol):
    name: str

    def score(
        self,
        items: list[LongContextItem],
        predictions: list[str],
    ) -> dict[str, float]: ...


@dataclass
class PlantedNeedleProbe:
    name: str = "planted_needle"

    def score(self, items, predictions):
        if len(items) != len(predictions):
            raise ValueError("items and predictions length mismatch")
        if not items:
            return {"accuracy": 0.0, "n": 0}
        hits = sum(
            contains_match(p, it.gold) for p, it in zip(predictions, items)
        )
        return {"accuracy": hits / len(items), "n": len(items)}


@dataclass
class EvidenceRetentionProbe:
    """Accuracy split by needle position bucket (early / middle / late)."""

    n_buckets: int = 3
    name: str = "evidence_retention"

    def _bucket(self, item: LongContextItem) -> str:
        n = len(item.chunks)
        if n == 0 or item.needle_chunk < 0:
            return "unknown"
        rel = item.needle_chunk / max(1, n - 1)
        if rel < 1 / self.n_buckets:
            return "early"
        if rel < 2 / self.n_buckets:
            return "middle"
        return "late"

    def score(self, items, predictions):
        if len(items) != len(predictions):
            raise ValueError("items and predictions length mismatch")
        buckets: dict[str, list[float]] = {}
        for item, pred in zip(items, predictions):
            b = self._bucket(item)
            buckets.setdefault(b, []).append(contains_match(pred, item.gold))
        out: dict[str, float] = {}
        for b, vals in buckets.items():
            out[f"{b}_accuracy"] = sum(vals) / max(1, len(vals))
            out[f"{b}_n"] = float(len(vals))
        out["accuracy"] = sum(
            sum(v) for v in buckets.values()
        ) / max(1, sum(len(v) for v in buckets.values()))
        out["n"] = float(sum(len(v) for v in buckets.values()))
        return out


def run_probes(
    probes: Iterable[Probe],
    items: list[LongContextItem],
    predictions: list[str],
) -> dict[str, dict[str, float]]:
    return {p.name: p.score(items, predictions) for p in probes}


def load_predictions_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
