"""Shared prompt-formatting and tokenisation helpers.

Strategies and the BPTT trainer use these so the answer-span boundary is
identical across baselines and training — this is what makes the per-token
KL distillation align.
"""

from __future__ import annotations

from typing import Any

import torch


def format_query_block(query: str) -> str:
    """Uniform query suffix used by every baseline and by training."""

    return f"\n\n### Question\n{query}\n\n### Answer\n"


def tokenize_to_ids(tokenizer: Any, text: str, *, max_length: int) -> torch.Tensor:
    """Tokenize with **left** truncation so the question/answer suffix is preserved.

    Strategies and the trainer always end the text with format_query_block(),
    i.e. ``...### Question ... ### Answer\\n``. The default right-truncation
    would drop that suffix when the chunk text exceeds max_length, leaving
    the model with code-only input and no instruction — which made the
    ``full_context`` baseline silently output continuation tokens instead of
    answers. Left-truncation drops the (less essential) earliest chunk text.
    """

    # tokenizers fall back gracefully if they don't honor the side override;
    # but for the HF tokenizers we use (Qwen, Llama, etc.) this is supported.
    saved = getattr(tokenizer, "truncation_side", None)
    if saved != "left":
        tokenizer.truncation_side = "left"
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    finally:
        if saved is not None:
            tokenizer.truncation_side = saved
    return enc["input_ids"]


def attention_for(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    return (ids != pad_id).long()
