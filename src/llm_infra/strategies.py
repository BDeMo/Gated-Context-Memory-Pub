"""Strategy implementations for the four-baseline harness.

A `Strategy` turns a `LongContextItem` into a `BaseCall` (ready for the frozen
base model to generate from). The wrapper strategy additionally produces a
memory and registers wrapper-side cleanup on the `BaseCall.extra`.

Strategies do NOT call the base model. The harness owns generation so the
generation logic is uniform across baselines.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from llm_infra.prompting import attention_for, format_query_block, tokenize_to_ids
from llm_infra.wrappers import BaseCall, ChunkEncoding, MemoryState, Wrapper


EncoderFn = Callable[[list[str]], ChunkEncoding]
"""Callable supplied by the caller; turns chunk strings into ChunkEncoding."""


@dataclass
class StrategyOutputs:
    base_call: BaseCall
    n_input_tokens: int
    debug: dict[str, Any]


class Strategy(Protocol):
    name: str

    def prepare(
        self,
        item: LongContextItem,
        tokenizer: Any,
        max_input_tokens: int,
    ) -> StrategyOutputs: ...


# --- No context (lower bound) ------------------------------------------------


@dataclass
class NoContextStrategy:
    """Query-only — base model sees no chunks, no wrapper, no retrieval.

    This is the literal "no wrapper" lower bound the user asks us to beat
    first. The base model has only its parametric knowledge plus the
    query text; any signal from the chunks must come through the wrapper
    or another strategy.
    """

    name: str = "no_context"

    def prepare(self, item, tokenizer, max_input_tokens):
        text = format_query_block(item.query).lstrip("\n")
        ids = tokenize_to_ids(tokenizer, text, max_length=max_input_tokens)
        return StrategyOutputs(
            base_call=BaseCall(
                input_ids=ids,
                attention_mask=attention_for(ids, tokenizer.pad_token_id),
            ),
            n_input_tokens=int(ids.shape[1]),
            debug={"n_chunks_used": 0},
        )


# --- Full context ------------------------------------------------------------


@dataclass
class FullContextStrategy:
    name: str = "full_context"

    def prepare(self, item, tokenizer, max_input_tokens):
        text = "\n\n".join(item.chunks) + format_query_block(item.query)
        ids = tokenize_to_ids(tokenizer, text, max_length=max_input_tokens)
        return StrategyOutputs(
            base_call=BaseCall(input_ids=ids, attention_mask=attention_for(ids, tokenizer.pad_token_id)),
            n_input_tokens=int(ids.shape[1]),
            debug={"text_chars": len(text)},
        )


# --- Naive summary ------------------------------------------------------------


@dataclass
class SummaryStrategy:
    """Concatenate the first `head_tokens` words of every chunk."""

    head_words: int = 32
    name: str = "summary"

    def prepare(self, item, tokenizer, max_input_tokens):
        heads = []
        for chunk in item.chunks:
            words = chunk.split()
            heads.append(" ".join(words[: self.head_words]))
        text = "\n\n".join(heads) + format_query_block(item.query)
        ids = tokenize_to_ids(tokenizer, text, max_length=max_input_tokens)
        return StrategyOutputs(
            base_call=BaseCall(input_ids=ids, attention_mask=attention_for(ids, tokenizer.pad_token_id)),
            n_input_tokens=int(ids.shape[1]),
            debug={"head_words": self.head_words, "n_chunks_used": len(heads)},
        )


# --- BM25 retrieval over chunks ----------------------------------------------


_TOK_RE = re.compile(r"\w+")


def _bm25_rank(query: str, chunks: list[str], k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    query_terms = [t.lower() for t in _TOK_RE.findall(query)]
    if not query_terms:
        return [(i, 0.0) for i in range(len(chunks))]

    docs = [[t.lower() for t in _TOK_RE.findall(c)] for c in chunks]
    doc_lens = [len(d) for d in docs]
    avgdl = sum(doc_lens) / max(1, len(docs))

    df: Counter[str] = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1
    n_docs = len(docs)

    scores: list[tuple[int, float]] = []
    for i, doc in enumerate(docs):
        tf = Counter(doc)
        s = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            denom = tf[term] + k1 * (1 - b + b * (doc_lens[i] / avgdl) if avgdl > 0 else 1)
            s += idf * (tf[term] * (k1 + 1)) / max(denom, 1e-9)
        scores.append((i, s))
    return sorted(scores, key=lambda x: x[1], reverse=True)


@dataclass
class RetrievalStrategy:
    top_k: int = 3
    name: str = "retrieval"

    def prepare(self, item, tokenizer, max_input_tokens):
        ranked = _bm25_rank(item.query, item.chunks)
        keep_idx = sorted(i for i, _ in ranked[: self.top_k])
        retrieved = [item.chunks[i] for i in keep_idx]
        text = "\n\n".join(retrieved) + format_query_block(item.query)
        ids = tokenize_to_ids(tokenizer, text, max_length=max_input_tokens)
        return StrategyOutputs(
            base_call=BaseCall(input_ids=ids, attention_mask=attention_for(ids, tokenizer.pad_token_id)),
            n_input_tokens=int(ids.shape[1]),
            debug={"top_k": self.top_k, "kept": keep_idx, "ranks": ranked[:5]},
        )


# --- Wrapper memory ----------------------------------------------------------


@dataclass
class WrapperStrategy:
    """Run the wrapper across chunks, prepend resulting memory to the query.

    Args:
        wrapper:    A `Wrapper`-conforming module.
        encoder_fn: Callable producing a `ChunkEncoding` from a list of strings.
        embed_fn:   Callable taking input_ids and returning the base model's
                    input embeddings. We use it to turn the query tokens into
                    embeddings so they can be concatenated with memory tokens
                    that already live in embedding space.
    """

    wrapper: Wrapper
    encoder_fn: EncoderFn
    embed_fn: Callable[[torch.Tensor], torch.Tensor]
    name: str = "wrapper"

    def prepare(self, item, tokenizer, max_input_tokens):
        device = next(iter(self._wrapper_params()), torch.tensor(0.0)).device

        memory: MemoryState = self.wrapper.init_memory(batch=1, device=device)
        for chunk in item.chunks:
            chunk_enc = self.encoder_fn([chunk])
            memory = self.wrapper.update(memory, chunk_enc)

        query_text = format_query_block(item.query).lstrip("\n")
        query_ids = tokenize_to_ids(tokenizer, query_text, max_length=max_input_tokens)
        with torch.no_grad():
            query_embeds = self.embed_fn(query_ids.to(device))

        base_call = BaseCall(
            inputs_embeds=query_embeds,
            attention_mask=torch.ones(query_embeds.shape[:2], dtype=torch.long, device=device),
            extra={"memory": memory},
        )
        base_call = self.wrapper.apply(base_call, memory)
        n_tokens = int(base_call.inputs_embeds.shape[1]) if base_call.inputs_embeds is not None else int(base_call.input_ids.shape[1])
        return StrategyOutputs(
            base_call=base_call,
            n_input_tokens=n_tokens,
            debug={"n_chunks": len(item.chunks)},
        )

    def _wrapper_params(self):
        if hasattr(self.wrapper, "parameters") and callable(self.wrapper.parameters):
            yield from self.wrapper.parameters()
