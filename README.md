# Gated Context Memory

GCM combines a query-conditioned memory writer, an adapted reader, and
first-token confidence for detecting compression failures. Raw-context fallback
is one use of the signal, not a guarantee of improved performance.

This repository contains a clean code snapshot without internal experiment
ledgers, server configuration, raw datasets, credentials, or development history.
The release focuses on the main comparison tables. Auxiliary experimental
launchers are outside the release scope. Runtime logs, event traces, and
per-example records are not distributed; generated results remain local.

## Installation

Use Linux, Python 3.12, and an NVIDIA CUDA environment. Training an 8B model
requires substantial GPU memory; CPU-only training is not supported by the runners.

```bash
git clone https://github.com/BDeMo/Gated-Context-Memory-Pub.git
cd Gated-Context-Memory-Pub
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r deploy/requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e '.[dev]'
export WANDB_MODE=disabled
export GCM_OUT="$PWD/results/outputs"
```

The lock file records the source environment; installation on a fresh GPU
machine and end-to-end paper reproduction have not yet been validated for
this public snapshot. Model and dataset licenses apply separately.

## Train and evaluate

First run a small smoke test; these limits do not produce paper results:

```bash
python experiments/run_full_ddp.py --model q3_8b --model-path Qwen/Qwen3-8B --bench hotpot_qa --devices 0 --train-limit 8 --val-limit 8 --suffix smoke
```

For the source-task HotpotQA training recipe, remove the smoke-test limits:

```bash
python experiments/run_full_ddp.py --model q3_8b --model-path Qwen/Qwen3-8B --bench hotpot_qa --devices 0,1,2,3,4,5,6,7
```

The runner saves result JSON and `*_adapters.pt` under `GCM_OUT`. It trains
the writer queries, projection, and reader adapter while freezing the backbone.
The default recipe uses 128 states per 4,096-token chunk; the reader receives
all chunks' states, not a fixed 128-state document budget.

Evaluate a saved adapter on another task:

```bash
python experiments/run_adapter_eval.py --model q3_8b --model-path Qwen/Qwen3-8B --adapter /path/to/adapter.pt --source-label hotpot_qa --target lb_2wikimqa --device 0 --nval 200
```

`hotpot_qa` is the source-task protocol. `lb_hotpotqa` and `lb_2wikimqa`
are LongBench protocols and their scores must not be substituted for source-task
scores. Some archived LongBench runs score the first generated line, which can
mis-score reasoning markers and summaries. Inspect generated answers before
interpreting these metrics as benchmark performance.

## Checkpoints and result mapping

See [CHECKPOINTS.md](CHECKPOINTS.md). Pretrained checkpoint access and the
checkpoint-to-paper-table mapping remain pending verification; this snapshot
must not yet be treated as a complete reproduction of every paper table.
Do not load untrusted PyTorch pickle checkpoints.

## Tests

```bash
python -m compileall -q src experiments
pytest -q tests
```

See [DATA_SOURCES.md](DATA_SOURCES.md) for dataset provenance. Logging is
disabled in the tutorial; opt in using your own WandB account if desired.
