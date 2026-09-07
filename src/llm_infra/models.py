"""Frozen base-model loaders.

Loaders return a `(model, tokenizer)` pair with the base model in eval mode and
all parameters detached from the autograd graph. We freeze on load (not later)
to make accidental fine-tuning of the base impossible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

# Mapping kept here so the CLI / configs reference short names. The HF repo ids
# can be overridden via `extra_kwargs` when a config wants a specific revision.
BASE_REGISTRY: dict[str, dict[str, Any]] = {
    "gemma-2-2b-it": {
        "hf_id": "google/gemma-2-2b-it",
        "torch_dtype": "bfloat16",
    },
    "qwen3-8b-instruct": {
        "hf_id": "Qwen/Qwen3-8B-Instruct",
        "torch_dtype": "bfloat16",
    },
    "qwen25-7b-instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "torch_dtype": "bfloat16",
    },
    "llama-31-8b-instruct": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "torch_dtype": "bfloat16",
    },
}


@dataclass
class LoadedBase:
    model: Any  # transformers.PreTrainedModel
    tokenizer: Any  # transformers.PreTrainedTokenizerBase
    hidden_size: int
    name: str


def freeze(model: Any) -> Any:
    """Freeze every parameter and put the model in eval mode."""

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


def load_base(
    name: str,
    *,
    device: torch.device | str = "cpu",
    torch_dtype: str | None = None,
    trust_remote_code: bool = True,
    attn_implementation: str | None = None,
) -> LoadedBase:
    """Load a base model and return it frozen.

    `name` can be:

    - A registry short name from `BASE_REGISTRY` (e.g. ``"qwen3-8b-instruct"``),
      in which case the registered HF id and dtype are used.
    - An HF model id like ``"Qwen/Qwen3-8B"``.
    - A local filesystem path like ``"Qwen/Qwen3-8B"``.

    `torch_dtype` overrides the registry-default dtype; pass ``"float32"`` for
    CPU debugging, ``"bfloat16"`` for GPU (default for non-registry names).

    For unit tests that should not download HF weights, prefer
    `llm_infra.testing.tiny_base` instead.
    """

    if name in BASE_REGISTRY:
        spec = BASE_REGISTRY[name]
        hf_id = spec["hf_id"]
        dtype_str = torch_dtype or spec["torch_dtype"]
        display_name = name
    else:
        hf_id = name
        dtype_str = torch_dtype or "bfloat16"
        # For display: tail of path or full id.
        display_name = name.rstrip("/").split("/")[-1] or name

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_str)
    logger.info(f"loading frozen base: {hf_id} dtype={dtype_str} device={device}")
    tok_extra = {}
    if "mistral" in str(hf_id).lower() or "ministral" in str(hf_id).lower():
        # Recent transformers detects the legacy Mistral pre-tokenizer regex and warns that it
        # tokenizes some strings incorrectly unless this compatibility fix is explicit.
        tok_extra["fix_mistral_regex"] = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_id,
            trust_remote_code=trust_remote_code,
            **tok_extra,
        )
    except TypeError as exc:
        # Some recent tokenizer classes already pass this flag internally; forwarding it
        # again raises "multiple values for keyword argument fix_mistral_regex".
        if "fix_mistral_regex" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            hf_id,
            trust_remote_code=trust_remote_code,
        )
    extra = {"attn_implementation": attn_implementation} if attn_implementation else {}
    # eager is required when callers pass custom 4D additive attention masks (e.g. Gist gist-masking);
    # sdpa/flash silently ignore or mishandle arbitrary masks on some architectures/versions.
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, trust_remote_code=trust_remote_code, **extra
    )
    model = freeze(model).to(device)

    return LoadedBase(
        model=model,
        tokenizer=tokenizer,
        hidden_size=model.config.hidden_size,
        name=display_name,
    )
