#!/usr/bin/env python3
"""Thin canonical-I/O adapter around authors' released model classes.

This file contains no compressor implementation.  It imports the pinned
Semi-Dynamic or AutoCompressor worktree and evaluates a common JSONL input
with the official checkpoint.  One JSON object is written per input item.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "paper-a-official-baseline/v1"
COMMITS = {
    "semi-dynamic": "07fae01c5062e57dc8277f7e696ca928cb775ca5",
    "autocompressor": "80352a4233c4a70504c75bf95c5f3e6d2533a863",
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"input has no records: {path}")
    required = {"task", "item_id", "context", "question"}
    for index, record in enumerate(records):
        missing = required - record.keys()
        if missing:
            raise ValueError(f"record {index} missing {sorted(missing)}")
        if not record.get("answers") and "gold" not in record:
            raise ValueError(f"record {index} needs answers or gold")
    return records


def _answers(record: dict[str, Any]) -> list[str]:
    value = record.get("answers", record.get("gold", []))
    if isinstance(value, dict):
        value = value.get("text", [])
    if not isinstance(value, list):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _normalized(text: str) -> str:
    return re.sub(r"[^\w]", "", str(text).lower())


def _qa_score(prediction: str, answers: list[str]) -> float:
    prediction = _normalized(prediction)
    normalized_answers = [
        normalized for answer in answers if (normalized := _normalized(answer))
    ]
    return float(any(answer in prediction for answer in normalized_answers))


def _timed(torch: Any, fn: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def _decode_continuation(tokenizer: Any, output: Any, input_length: int = 0) -> str:
    ids = output.sequences[0] if hasattr(output, "sequences") else output[0]
    return tokenizer.decode(ids[input_length:], skip_special_tokens=True).strip()


def _raw_generate(
    torch: Any,
    model: Any,
    tokenizer: Any,
    text: str,
    max_new_tokens: int,
    max_input_tokens: int,
) -> tuple[str, int, float]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_input_tokens,
    ).input_ids.to(model.device)
    output, elapsed = _timed(
        torch,
        lambda: model.generate(
            encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        ),
    )
    return _decode_continuation(tokenizer, output, encoded.shape[1]), int(encoded.shape[1]), elapsed


def _canonical(
    *,
    args: argparse.Namespace,
    record: dict[str, Any],
    base: str,
    prediction: str,
    score: float,
    input_tokens: int,
    state_tokens: int,
    compressed_seconds: float,
    encode_seconds: float,
    raw: dict[str, Any],
    no_context: dict[str, Any],
    peak_memory_bytes: int,
    method_meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "OFFICIAL_AUTHORS_CODE_AND_CHECKPOINT",
        "method": args.method,
        "variant": args.variant,
        "task": str(record["task"]),
        "item_id": str(record["item_id"]),
        "base": base,
        "checkpoint": args.checkpoint_id,
        "repository": {
            "path": str(Path(args.official_repo).resolve()),
            "commit": COMMITS[args.method],
        },
        "pred": prediction,
        "gold": _answers(record),
        "score": score,
        "input_tokens": input_tokens,
        "state_tokens": state_tokens,
        "latency": {
            "encode_seconds": encode_seconds,
            "reader_seconds": compressed_seconds,
        },
        "peak_memory_bytes": peak_memory_bytes,
        "references": {
            "no_context": no_context,
            "raw": raw,
        },
        "method_meta": method_meta,
    }


def _semi_dynamic(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sys.path.insert(0, args.official_repo)
    import torch
    from modeling_ctxcomp import CtxCompSemiDynamicModel
    from transformers import AutoTokenizer

    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    encoder_id = config.get("base_encoder_model_path") or config.get("base_embed_model_path")
    decoder_id = config.get("base_decoder_model_path") or config.get("base_gen_model_path")
    if not encoder_id or not decoder_id:
        raise ValueError("checkpoint config lacks official encoder/decoder IDs")
    model = CtxCompSemiDynamicModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        encoder_id, trust_remote_code=True, padding_side="left"
    )
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        decoder_id, trust_remote_code=True, padding_side="left"
    )
    placeholder = decoder_tokenizer.convert_ids_to_tokens(config["placeholder_token_id"])
    output_records = []
    for record in records:
        if record.get("options"):
            raise ValueError("Semi-Dynamic adapter supports its native short-QA scope, not MC")
        torch.cuda.reset_peak_memory_stats()
        context = encoder_tokenizer(
            str(record["context"]),
            return_tensors="pt",
            truncation=True,
            max_length=args.raw_max_tokens,
        )
        context_ids = context.input_ids.to(model.decoder.device)
        context_mask = context.attention_mask.to(model.decoder.device)
        fixed = args.variant == "fixed-32x"
        n_placeholders = max(1, math.ceil(context_ids.shape[1] / 32)) if fixed else 1
        user = f"Context: {placeholder * n_placeholders}\nQuestion: {record['question']}"
        prompt_ids = decoder_tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.decoder.device)
        compression_kwargs = {
            "context_input_ids": context_ids,
            "context_attention_mask": context_mask,
            "input_ids": prompt_ids,
            "attention_mask": torch.ones_like(prompt_ids),
            "comp_ratio_or_len_override": 0.03125 if fixed else None,
        }
        computed, encode_seconds = _timed(
            torch, lambda: model.compute_inputs_embeds(**compression_kwargs)
        )
        inputs_embeds, compressed_mask, _, valid, _, _, _ = computed
        model._switch_decoder_adapter()
        output, reader_seconds = _timed(
            torch,
            lambda: model.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=compressed_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=decoder_tokenizer.pad_token_id
                or decoder_tokenizer.eos_token_id,
            ),
        )
        prediction = _decode_continuation(decoder_tokenizer, output)
        state_tokens = int(valid[0].item())
        no_pred, no_tokens, no_seconds = _raw_generate(
            torch,
            model.decoder,
            decoder_tokenizer,
            f"Question: {record['question']}",
            args.max_new_tokens,
            args.raw_max_tokens,
        )
        raw_pred, raw_tokens, raw_seconds = _raw_generate(
            torch,
            model.decoder,
            decoder_tokenizer,
            f"Context: {record['context']}\nQuestion: {record['question']}",
            args.max_new_tokens,
            args.raw_max_tokens,
        )
        answers = _answers(record)
        output_records.append(
            _canonical(
                args=args,
                record=record,
                base=decoder_id,
                prediction=prediction,
                score=_qa_score(prediction, answers),
                input_tokens=int(context_mask.sum().item()),
                state_tokens=state_tokens,
                compressed_seconds=reader_seconds,
                encode_seconds=encode_seconds,
                raw={
                    "pred": raw_pred,
                    "score": _qa_score(raw_pred, answers),
                    "input_tokens": raw_tokens,
                    "latency_seconds": raw_seconds,
                    "truncated": raw_tokens >= args.raw_max_tokens,
                },
                no_context={
                    "pred": no_pred,
                    "score": _qa_score(no_pred, answers),
                    "input_tokens": no_tokens,
                    "latency_seconds": no_seconds,
                },
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
                method_meta={
                    "official_class": "CtxCompSemiDynamicModel",
                    "encoder": encoder_id,
                    "fixed_ratio_override": 0.03125 if fixed else None,
                },
            )
        )
    return output_records


def _option_nll(torch: Any, model: Any, prefix: Any, question: str, option: str, tokenizer: Any, softprompt: Any = None) -> float:
    unanswered = tokenizer(
        question, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(model.device)
    answer = tokenizer(
        f"{question}{option}", add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(model.device)
    option_ids = answer[:, unanswered.shape[1]:]
    tokens = answer if prefix is None else torch.cat([prefix, answer], dim=1)
    kwargs = {"use_cache": False}
    if softprompt is not None:
        kwargs["softprompt"] = softprompt
    logits = model(tokens, **kwargs)["logits"][:, -option_ids.shape[1] - 1:-1]
    return float(
        -torch.log_softmax(logits, dim=-1)
        .gather(2, option_ids.unsqueeze(-1))
        .mean()
        .item()
    )


def _autocompressor(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sys.path.insert(0, args.official_repo)
    import torch
    from auto_compressor import AutoCompressorModel, LlamaAutoCompressorModel
    from transformers import AutoTokenizer

    cls = (
        LlamaAutoCompressorModel
        if "llama" in args.checkpoint_id.lower()
        else AutoCompressorModel
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=False)
    model = cls.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16
    ).eval().cuda()
    output_records = []
    for record in records:
        torch.cuda.reset_peak_memory_stats()
        context_ids = tokenizer(
            str(record["context"]), add_special_tokens=True, return_tensors="pt"
        ).input_ids.to(model.device)
        lengths = [
            min(args.segment_size, context_ids.shape[1] - start)
            for start in range(0, context_ids.shape[1], args.segment_size)
        ]
        compressed, encode_seconds = _timed(
            torch,
            lambda: model(
                context_ids,
                segment_lengths=lengths,
                output_softprompt=True,
                use_cache=False,
            ),
        )
        softprompt = compressed.softprompt
        options = [str(value) for value in record.get("options", [])]
        question = str(record["question"])
        if options:
            labels = [chr(ord("A") + index) for index in range(len(options))]
            scored_options = [
                f" {label}. {option}"
                for label, option in zip(labels, options)
            ]
            compressed_nll, reader_seconds = _timed(
                torch,
                lambda: [
                    _option_nll(torch, model, None, question, option, tokenizer, softprompt)
                    for option in scored_options
                ],
            )
            prediction = labels[min(range(len(labels)), key=compressed_nll.__getitem__)]
            gold = str(record.get("gold", _answers(record)[0]))
            score = float(prediction == gold)
            raw_prefix = context_ids[:, -args.raw_max_tokens:]
            raw_nll, raw_seconds = _timed(
                torch,
                lambda: [
                    _option_nll(torch, model, raw_prefix, question, option, tokenizer)
                    for option in scored_options
                ],
            )
            no_nll, no_seconds = _timed(
                torch,
                lambda: [
                    _option_nll(torch, model, None, question, option, tokenizer)
                    for option in scored_options
                ],
            )
            raw_pred = labels[min(range(len(labels)), key=raw_nll.__getitem__)]
            no_pred = labels[min(range(len(labels)), key=no_nll.__getitem__)]
            raw = {
                "pred": raw_pred,
                "score": float(raw_pred == gold),
                "input_tokens": int(raw_prefix.shape[1]),
                "latency_seconds": raw_seconds,
                "truncated": context_ids.shape[1] > args.raw_max_tokens,
                "option_nll": raw_nll,
            }
            no_context = {
                "pred": no_pred,
                "score": float(no_pred == gold),
                "input_tokens": 0,
                "latency_seconds": no_seconds,
                "option_nll": no_nll,
            }
            method_meta = {"option_nll": compressed_nll}
        else:
            query_ids = tokenizer(
                question, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(model.device)
            generated, reader_seconds = _timed(
                torch,
                lambda: model.generate(
                    query_ids,
                    softprompt=softprompt,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                ),
            )
            prediction = _decode_continuation(tokenizer, generated, query_ids.shape[1])
            answers = _answers(record)
            score = _qa_score(prediction, answers)
            no_pred, no_tokens, no_seconds = _raw_generate(
                torch, model, tokenizer, question, args.max_new_tokens, args.raw_max_tokens
            )
            raw_pred, raw_tokens, raw_seconds = _raw_generate(
                torch,
                model,
                tokenizer,
                f"{record['context']}\n\n{question}",
                args.max_new_tokens,
                args.raw_max_tokens,
            )
            raw = {
                "pred": raw_pred,
                "score": _qa_score(raw_pred, answers),
                "input_tokens": raw_tokens,
                "latency_seconds": raw_seconds,
                "truncated": raw_tokens >= args.raw_max_tokens,
            }
            no_context = {
                "pred": no_pred,
                "score": _qa_score(no_pred, answers),
                "input_tokens": no_tokens,
                "latency_seconds": no_seconds,
            }
            method_meta = {}
        output_records.append(
            _canonical(
                args=args,
                record=record,
                base=str(getattr(model.config, "_name_or_path", args.checkpoint_id)),
                prediction=prediction,
                score=score,
                input_tokens=int(context_ids.shape[1]),
                state_tokens=int(softprompt.shape[1]),
                compressed_seconds=reader_seconds,
                encode_seconds=encode_seconds,
                raw=raw,
                no_context=no_context,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
                method_meta={
                    **method_meta,
                    "official_class": cls.__name__,
                    "segment_lengths": lengths,
                },
            )
        )
    return output_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(COMMITS), required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--segment-size", type=int, default=1536)
    parser.add_argument("--raw-max-tokens", type=int, default=6144)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("CUDA_VISIBLE_DEVICES must select one GPU")
    records = _load_records(Path(args.input))
    outputs = (
        _semi_dynamic(args, records)
        if args.method == "semi-dynamic"
        else _autocompressor(args, records)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as handle:
        for record in outputs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
