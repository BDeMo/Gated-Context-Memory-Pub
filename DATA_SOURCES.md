# Data and model sources

The repository does not redistribute datasets or model weights. Loaders in
`src/llm_infra/datasets.py` fetch public resources through Hugging Face on first use and cache
them under `HF_HOME`. Set `HF_TOKEN` only when a model's license requires authentication.

## Paper benchmarks

| Paper name | Loader source | Evaluation split used |
|---|---|---|
| QuALITY | `emozilla/quality` | validation (2,086) |
| BFCL live multiple | `gorilla-llm/Berkeley-Function-Calling-Leaderboard`, `BFCL_v3_live_multiple.json` plus `possible_answer/` | deterministic 30% held-out partition (316) |
| SQuAD v2 | `rajpurkar/squad_v2` | validation (5,928 after protocol filtering) |
| HotpotQA | `hotpotqa/hotpot_qa`, `distractor` | validation (7,405 after protocol filtering) |
| NarrativeQA | `deepmind/narrativeqa` | validation (3,461 after protocol filtering) |
| MuSR murder mysteries | `TAUR-Lab/MuSR` | stable content-hash validation partition (90) |

Additional long-context loaders used by ablations are included in source: LongBench-v2
(`THUDM/LongBench-v2`), InfiniteBench (`xinrongzhang2022/InfiniteBench`), BABILong
(`RMT-team/babilong`), WikiText-103 (`Salesforce/wikitext`), LoCoMo
(`snap-research/locomo`), API-Bank (`liminghao1630/API-Bank`), ToolACE
(`Team-ACE/ToolACE`), Hermes function calling
(`NousResearch/hermes-function-calling-v1`), and Glaive function calling
(`glaiveai/glaive-function-calling-v2`).

To warm the cache without committing data, run any experiment with:

```bash
export HF_HOME="$PWD/.cache/huggingface"
export HF_TOKEN="<only-if-required>"
```

## Model checkpoints

The experiment registry resolves these Hugging Face IDs directly:

- `Qwen/Qwen3-8B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-4B`
- `mistralai/Ministral-8B-Instruct-2410`
- `zai-org/GLM-4-9B-0414`
- `Salesforce/Llama-xLAM-2-8b-fc-r`
- `Team-ACE/ToolACE-2-8B`

Local checkpoint directories can be supplied with `--model-path` (GCM full training) or
`--model` (training-free baselines). Dataset and model licenses remain those of their
original publishers.
