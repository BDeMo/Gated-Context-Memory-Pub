"""Uniform generation helper used by every baseline.

Greedy decoding by default so results are deterministic. We always operate in
embedding space inside the loop — when the caller starts with `input_ids` we
look them up through the model's embedding table once and then concatenate
freshly embedded next tokens. This keeps the code path uniform between the
`full_context` / `summary` / `retrieval` baselines (token starts) and the
`wrapper` baseline (embedding starts), and avoids attention-mask / sequence
length mismatches that arise when switching modes mid-loop without KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class GenerationOutputs:
    text: str
    token_ids: list[int]


@torch.no_grad()
def generate_greedy(
    model: Any,
    tokenizer: Any,
    *,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    max_new_tokens: int = 32,
    eos_token_id: int | None = None,
    stop_strings: list[str] | tuple[str, ...] | None = ("\n",),
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> GenerationOutputs:
    """Greedy / stochastic decode in embedding space.

    ``stop_strings`` are checked after each newly generated token: when the
    decoded text ends with any of them, generation stops *before* that token
    is appended to the output (it is dropped, mirroring HF's standard
    ``stopping_criteria`` behavior). Pass ``None`` or ``()`` to disable.
    Defaults to ``("\\n",)`` because the answer span we score on
    (``### Answer\\n``) is followed by one short answer line — stopping at the
    first newline matches the natural end of the answer and prevents
    verbose continuations from diluting token-overlap metrics like F1.

    When ``do_sample=True``, the next token is drawn from
    ``softmax(logits / temperature)`` after optional top-k / top-p filtering.
    Sampling rollouts for RL / on-policy distillation typically want
    ``do_sample=True, temperature=1.0, stop_strings=()`` so the full
    ``max_new_tokens`` worth of tokens are produced and can be scored.
    """

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("provide exactly one of input_ids / inputs_embeds")
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
    stop_strings = tuple(stop_strings or ())

    # Defensive device alignment: callers (strategies, smoke scripts) may pass
    # tensors that live on CPU even when the model is on GPU. Move them onto
    # the model's device once here so individual call sites don't have to.
    model_device = next(model.parameters()).device
    if input_ids is not None and input_ids.device != model_device:
        input_ids = input_ids.to(model_device)
    if inputs_embeds is not None and inputs_embeds.device != model_device:
        inputs_embeds = inputs_embeds.to(model_device)
    if attention_mask is not None and attention_mask.device != model_device:
        attention_mask = attention_mask.to(model_device)

    embed_table = model.get_input_embeddings()
    if input_ids is not None:
        cur_embeds = embed_table(input_ids)
    else:
        cur_embeds = inputs_embeds

    if attention_mask is None:
        cur_attn = torch.ones(
            cur_embeds.shape[:2], dtype=torch.long, device=cur_embeds.device
        )
    elif attention_mask.shape[1] != cur_embeds.shape[1]:
        # When the wrapper prepended memory tokens, the caller may have built
        # the attention mask only for the original query span; pad with ones
        # on the left so the prepended memory positions are visible.
        pad = cur_embeds.shape[1] - attention_mask.shape[1]
        if pad < 0:
            raise ValueError("attention_mask longer than embeds")
        cur_attn = torch.cat(
            [
                torch.ones(
                    cur_embeds.shape[0], pad, dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ],
            dim=1,
        )
    else:
        cur_attn = attention_mask

    generated: list[int] = []
    for _ in range(max_new_tokens):
        outputs = model(inputs_embeds=cur_embeds, attention_mask=cur_attn, use_cache=False)
        next_logits = outputs.logits[:, -1, :]
        if not do_sample:
            next_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            scaled = next_logits / max(temperature, 1e-6)
            if top_k > 0:
                vk = torch.topk(scaled, k=min(top_k, scaled.shape[-1]), dim=-1)
                cutoff = vk.values[..., -1, None]
                scaled = torch.where(scaled >= cutoff, scaled, torch.full_like(scaled, float("-inf")))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = sorted_probs.cumsum(dim=-1)
                # Keep at least one token; drop those past top_p cumulative mass.
                drop = cumprobs > top_p
                drop[..., 1:] = drop[..., :-1].clone()
                drop[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
                scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
            probs = torch.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break

        # Stop-string check: decode the running output and look at the suffix.
        # Drop the matched stop substring from the output so callers get the
        # answer span only (matches HF stopping-criteria conventions and
        # keeps token-overlap metrics clean).
        if stop_strings:
            running = tokenizer.decode(generated, skip_special_tokens=True)
            stop_hit = next((s for s in stop_strings if running.endswith(s)), None)
            if stop_hit is not None:
                clipped = running[: -len(stop_hit)]
                re_enc = tokenizer(clipped, add_special_tokens=False)["input_ids"]
                # Re-encoding may not match the exact token sequence (tokenizers
                # are not always a left-inverse of decode); fall back to popping
                # the trailing tokens whose decoded length covers the stop hit.
                if len(re_enc) <= len(generated):
                    generated = list(re_enc)
                else:
                    generated = generated[:-1]
                break

        next_token = torch.tensor([[next_id]], device=cur_embeds.device, dtype=torch.long)
        cur_embeds = torch.cat([cur_embeds, embed_table(next_token)], dim=1)
        cur_attn = torch.cat([cur_attn, torch.ones_like(next_token)], dim=1)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return GenerationOutputs(text=text, token_ids=generated)


def first_token_log_probs(
    model: Any,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `log p(next-token | prompt)` for use by training-time KL distillation."""

    model_device = next(model.parameters()).device
    kwargs: dict[str, Any] = {}
    if input_ids is not None:
        kwargs["input_ids"] = input_ids.to(model_device) if input_ids.device != model_device else input_ids
    if inputs_embeds is not None:
        kwargs["inputs_embeds"] = (
            inputs_embeds.to(model_device) if inputs_embeds.device != model_device else inputs_embeds
        )
    if attention_mask is not None:
        kwargs["attention_mask"] = (
            attention_mask.to(model_device)
            if attention_mask.device != model_device
            else attention_mask
        )
    outputs = model(**kwargs, use_cache=False)
    logits = outputs.logits[:, -1, :]
    return F.log_softmax(logits, dim=-1)
