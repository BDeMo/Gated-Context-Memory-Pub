"""Command-line entry points: `llm-infra eval`, `llm-infra probe`, `llm-infra build-dataset`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import yaml
from loguru import logger

from llm_infra.datasets import (
    LongContextItem,
    generate_coding_niah,
    iter_items_jsonl,
    write_items_jsonl,
)
from llm_infra.probes import (
    EvidenceRetentionProbe,
    PlantedNeedleProbe,
    load_predictions_jsonl,
    run_probes,
)


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@click.group()
def main() -> None:
    """llm-infra commands."""


@main.command("build-dataset")
@click.option("--name", default="coding_niah", show_default=True)
@click.option("--n-items", type=int, default=50, show_default=True)
@click.option("--n-chunks", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=12345, show_default=True)
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False))
def cmd_build_dataset(name: str, n_items: int, n_chunks: int, seed: int, out_path: str) -> None:
    """Generate a Phase 0 synthetic dataset and write to JSONL."""

    if name != "coding_niah":
        raise click.ClickException(f"unknown dataset '{name}'; only 'coding_niah' is supported in Phase 0")
    items = generate_coding_niah(n_items=n_items, n_chunks=n_chunks, seed=seed)
    write_items_jsonl(items, out_path)
    click.echo(f"wrote {len(items)} items -> {out_path}")


@main.command("eval")
@click.option("--wrapper", "wrapper_name", required=True, help="Python import name of the wrapper package, e.g. mem_embedding")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
def cmd_eval(wrapper_name: str, config_path: str) -> None:
    """Run the four-baseline harness for the given wrapper + config.

    The config must specify base model, dataset, and per-strategy settings.
    See `configs/phase0_tiny.yaml` in this repo for the schema.
    """

    cfg = _load_config(config_path)
    logger.info(f"[cli.eval] wrapper={wrapper_name} config={config_path}")
    logger.info(f"[cli.eval] config keys: {list(cfg)}")

    raise click.ClickException(
        "cmd_eval requires a wrapper repo to be installed and a base model loader. "
        "Use this CLI from inside a wrapper repo's entrypoint script or call the "
        "harness programmatically (see tests/blackbox in this repo for an example)."
    )


@main.command("probe")
@click.option("--items", "items_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--predictions", "preds_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", default=None, type=click.Path(dir_okay=False))
def cmd_probe(items_path: str, preds_path: str, out_path: str | None) -> None:
    """Re-score an existing predictions.jsonl with all Phase 0 probes."""

    items: list[LongContextItem] = list(iter_items_jsonl(items_path))
    raw_preds = load_predictions_jsonl(preds_path)
    pred_by_id = {row["item_id"]: row.get("prediction", "") for row in raw_preds}
    predictions = [pred_by_id.get(it.item_id, "") for it in items]

    probes = [PlantedNeedleProbe(), EvidenceRetentionProbe()]
    results = run_probes(probes, items, predictions)
    payload = json.dumps(results, indent=2)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(payload)
    click.echo(payload)


if __name__ == "__main__":
    main()
