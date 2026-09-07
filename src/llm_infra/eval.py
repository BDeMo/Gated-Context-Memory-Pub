"""The four-baseline evaluation harness.

Runs `full_context`, `summary`, `retrieval`, and `wrapper` strategies on a
dataset of `LongContextItem`s, generates greedily from the frozen base, and
writes a single JSONL of per-strategy metrics. Output layout intentionally
mirrors `rca-demo/scripts/compression/README.md`:

    runs/<run_id>/eval/<model_id>/<strategy>/
      predictions.jsonl
      metrics.json
      case_scores.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from llm_infra.benchmark_metrics import aggregate as _bench_aggregate
from llm_infra.benchmark_metrics import score_item as _bench_score_item
from llm_infra.datasets import LongContextItem
from llm_infra.generation import generate_greedy
from llm_infra.metrics import contains_match, exact_match, summarise, token_f1
from llm_infra.strategies import Strategy


@dataclass
class FourBaselineHarness:
    model: Any
    tokenizer: Any
    strategies: dict[str, Strategy]
    out_dir: Path
    model_id: str = "base"
    max_input_tokens: int = 1024
    max_new_tokens: int = 32

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, items: list[LongContextItem]) -> dict[str, dict[str, Any]]:
        all_metrics: dict[str, dict[str, Any]] = {}
        for strat_name, strategy in self.strategies.items():
            strat_dir = self.out_dir / self.model_id / strat_name
            strat_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[harness] running strategy={strat_name} on {len(items)} items")
            metrics = self._run_strategy(strategy, items, strat_dir)
            all_metrics[strat_name] = metrics
        self._write_summary(all_metrics)
        return all_metrics

    def _run_strategy(self, strategy: Strategy, items: list[LongContextItem], strat_dir: Path):
        preds: list[str] = []
        golds: list[str] = []
        tok_counts: list[int] = []
        case_rows: list[dict[str, Any]] = []
        pred_rows: list[dict[str, Any]] = []
        official_per_item: list[dict[str, float]] = []

        for item in items:
            outs = strategy.prepare(item, self.tokenizer, max_input_tokens=self.max_input_tokens)
            bc = outs.base_call
            bc.assert_consistent()
            cleanup = bc.extra.pop("_cleanup", None)
            try:
                gen = generate_greedy(
                    self.model,
                    self.tokenizer,
                    input_ids=bc.input_ids,
                    attention_mask=bc.attention_mask,
                    inputs_embeds=bc.inputs_embeds,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
            finally:
                if cleanup is not None:
                    cleanup()

            preds.append(gen.text)
            golds.append(item.gold)
            tok_counts.append(outs.n_input_tokens)

            # Official benchmark metric — keyed by the dataset tag the
            # adapter stamped into item.meta["dataset"]. Falls back to a
            # generic NIAH substring metric for items without a known
            # benchmark tag (matches the original Kamradt 2024 NIAH
            # protocol).
            ds_tag = (item.meta or {}).get("dataset", "")
            official = _bench_score_item(
                dataset=ds_tag,
                prediction=gen.text,
                gold=item.gold,
                meta=item.meta or {},
            )
            official_per_item.append(official)

            case_rows.append({
                "item_id": item.item_id,
                "exact_match": exact_match(gen.text, item.gold),
                "contains_match": contains_match(gen.text, item.gold),
                "token_f1": token_f1(gen.text, item.gold),
                "official": official,
                "n_input_tokens": outs.n_input_tokens,
                "n_output_tokens": len(gen.token_ids),
            })
            pred_rows.append({
                "item_id": item.item_id,
                "gold": item.gold,
                "prediction": gen.text,
                "needle_chunk": item.needle_chunk,
                "dataset": ds_tag,
            })

        summary = summarise(preds, golds, tok_counts).to_dict()
        summary["strategy"] = strategy.name
        official_agg = _bench_aggregate(official_per_item)
        if official_agg:
            # Drop the bare 'n' so the official sub-dict is readable as
            # metric_name -> value; n is already in summary['n'].
            official_agg = {k: v for k, v in official_agg.items() if k != "n"}
            summary["official"] = official_agg

        with open(strat_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for row in pred_rows:
                f.write(json.dumps(row) + "\n")
        with open(strat_dir / "case_scores.jsonl", "w", encoding="utf-8") as f:
            for row in case_rows:
                f.write(json.dumps(row) + "\n")
        with open(strat_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if official_agg:
            with open(strat_dir / "official_metrics.json", "w", encoding="utf-8") as f:
                json.dump(official_agg, f, indent=2)

        return summary

    def _write_summary(self, all_metrics: dict[str, dict[str, Any]]) -> None:
        summary_path = self.out_dir / self.model_id / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {"model_id": self.model_id, "strategies": all_metrics},
                f,
                indent=2,
            )
        logger.info(f"[harness] wrote {summary_path}")
