"""Test fixtures that work without network or GPU.

- `tiny_base()` builds a random-init `LlamaForCausalLM` from `LlamaConfig` so
  CPU smoke tests can run end-to-end without downloading multi-GB weights.
- `FakeTokenizer` is a deterministic word-level tokenizer that maps tokens to
  ids by hashing into the model's vocab range. Reproducible across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from llm_infra.models import LoadedBase, freeze


@dataclass
class FakeTokenizer:
    """Word-level deterministic tokenizer for blackbox tests.

    Notes:
        - `vocab_size`, `pad_token_id`, `bos_token_id`, `eos_token_id`
          mimic the HF tokenizer surface that the harness touches.
        - `__call__` returns a dict mirroring HF's `BatchEncoding` shape.
    """

    vocab_size: int = 512
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    model_max_length: int = 2048

    @property
    def _reserved(self) -> int:
        return 3

    def _word_to_id(self, word: str) -> int:
        # Deterministic hash; modulo into the non-reserved range.
        h = 1469598103934665603
        for ch in word:
            h ^= ord(ch)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return self._reserved + (h % (self.vocab_size - self._reserved))

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for token in text.split():
            ids.append(self._word_to_id(token))
        return ids

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        texts = [text] if isinstance(text, str) else list(text)
        seqs = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
        if truncation and max_length is not None:
            seqs = [s[:max_length] for s in seqs]
        if padding:
            max_len = max(len(s) for s in seqs)
            padded = [s + [self.pad_token_id] * (max_len - len(s)) for s in seqs]
            attn = [[1] * len(s) + [0] * (max_len - len(s)) for s in seqs]
        else:
            padded = seqs
            attn = [[1] * len(s) for s in seqs]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": attn}

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        return " ".join(f"<t{i}>" for i in ids if not (skip_special_tokens and i in special))

    def batch_decode(self, ids, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in ids]


def tiny_llama_config(
    vocab_size: int = 512,
    hidden_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    intermediate_size: int = 128,
    max_position_embeddings: int = 2048,
):
    """Build a `LlamaConfig` small enough for CPU smoke tests."""

    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=max(1, num_attention_heads // 2),
        max_position_embeddings=max_position_embeddings,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def tiny_base(
    *,
    vocab_size: int = 512,
    hidden_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    seed: int = 12345,
    device: str | torch.device = "cpu",
) -> LoadedBase:
    """Build a tiny frozen `LlamaForCausalLM` for CPU smoke tests."""

    from transformers import LlamaForCausalLM

    torch.manual_seed(seed)
    config = tiny_llama_config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
    )
    model = LlamaForCausalLM(config)
    model = freeze(model).to(device)
    tokenizer = FakeTokenizer(vocab_size=vocab_size)

    return LoadedBase(
        model=model,
        tokenizer=tokenizer,
        hidden_size=hidden_size,
        name="tiny-llama",
    )


class IdentityEncoder(nn.Module):
    """Trivial encoder for tests: pads/embeds tokens with a frozen embedding table.

    Returns `[batch, seq, hidden]` so it satisfies `ChunkEncoding.hidden` shape.
    """

    def __init__(self, vocab_size: int = 512, hidden_size: int = 64, seed: int = 7):
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        freeze(self)

    @torch.no_grad()
    def encode(
        self,
        chunks: list[str],
        tokenizer: FakeTokenizer,
        max_tokens: int = 128,
        device: str | torch.device = "cpu",
    ):
        from llm_infra.wrappers import ChunkEncoding

        out = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        ids = out["input_ids"].to(device)
        mask = out["attention_mask"].to(device)
        hidden = self.embed(ids)
        return ChunkEncoding(hidden=hidden, mask=mask)
