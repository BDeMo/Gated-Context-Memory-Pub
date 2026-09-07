r"""Capacity & information-content measurements for the pre-experiment.

This module answers two questions with **no wrapper training required**:

1. *How much can we compress?* — compression-ratio, theoretical & effective
   memory capacity, information density of the source corpora.
2. *Where does the information live?* — per-chunk contribution, per-position
   surprisal, redundancy and Gini concentration.

It does so by computing perplexity on the gold answer span under several
context conditions (no_ctx / full_ctx / rand_mem / pooled_mem / summary /
retrieval) and reporting the recovery ratios.

All functions are `@torch.no_grad`; they assume a frozen base model.

Output structure mirrors the metric panel in the project docs:

    A. capacity            (compression ratio, theoretical & effective rank, entropy)
    B. recovery            (PPL under each condition, PRR, bits saved per slot)
    C. distribution        (per-chunk contribution, Gini, top-k coverage, redundancy)
    D. domain density      (bits / token, bits / char, surprisal heavy-tail)

`run_capacity_panel(...)` orchestrates everything and returns a dict that is
trivially JSON-serialisable for the precapacity_smoke script.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from loguru import logger

from llm_infra.datasets import LongContextItem
from llm_infra.prompting import format_query_block


# ============================================================================ #
# A. Capacity                                                                  #
# ============================================================================ #


def compression_ratio(n_full_ctx_tokens: int, n_compressed_tokens: int) -> float:
    """`tokens(full_ctx) / tokens(compressed_prompt)`. ≥1 means we compressed."""

    if n_compressed_tokens <= 0:
        return float("inf")
    return n_full_ctx_tokens / n_compressed_tokens


def theoretical_capacity_bits(n_memory: int, d_model: int, bits_per_element: float = 12.0) -> float:
    """Upper bound on bits a memory tensor can carry.

    `bits_per_element` defaults to 12 (effective bits of bf16 after
    quantisation noise; an aggressive lower bound on fp16 / bf16 entropy).
    """

    return float(n_memory) * float(d_model) * float(bits_per_element)


def bytes_per_memory_slot(text: str, n_memory: int, d_model: int, dtype_bytes: int = 2) -> float:
    """Source text bytes per memory slot of `dtype_bytes` per element.

    Smaller = denser source; larger = wasted memory. A "dilution ratio" proxy.
    """

    text_bytes = len(text.encode("utf-8"))
    mem_bytes = n_memory * d_model * dtype_bytes
    if text_bytes == 0:
        return 0.0
    return mem_bytes / text_bytes


def effective_rank(matrix: torch.Tensor, threshold: float = 0.95) -> int:
    """Smallest k such that the top-k singular values explain `threshold`
    fraction of the variance.

    `matrix`: `[N, D]`. Returns 0 if N < 2.
    """

    if matrix.dim() != 2 or matrix.shape[0] < 2:
        return 0
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered.float())
    except RuntimeError:
        return 0
    var = s.pow(2)
    if var.sum() <= 0:
        return 0
    cum = torch.cumsum(var, dim=0) / var.sum()
    k = int((cum < threshold).sum().item()) + 1
    return min(k, var.numel())


def empirical_entropy_quantile(matrix: torch.Tensor, n_bins: int = 64) -> float:
    """Cheap entropy estimate: bin all scalars into quantile buckets, then
    compute Shannon entropy (bits). Reports "effective bits per scalar"
    averaged over the whole tensor.

    Bounded above by `log2(n_bins)`.
    """

    if matrix.numel() == 0:
        return 0.0
    flat = matrix.flatten().float().cpu()
    q = torch.linspace(0.0, 1.0, n_bins + 1)
    edges = torch.quantile(flat, q)
    # bucketize; clamp to [0, n_bins-1]
    idx = torch.bucketize(flat, edges[1:-1])
    counts = torch.bincount(idx, minlength=n_bins).float()
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * torch.log2(nz)).sum().item())


# ============================================================================ #
# B. PPL recovery                                                              #
# ============================================================================ #


@torch.no_grad()
def answer_span_ppl_from_text(
    model: Any,
    tokenizer: Any,
    prefix_text: str,
    gold_text: str,
    *,
    max_input_tokens: int = 2048,
) -> tuple[float, int]:
    """Perplexity on the gold span when conditioned on `prefix_text`.

    Returns `(ppl, n_input_tokens)`. `n_input_tokens` is the length of the
    fully-formed (prefix + gold) sequence after tokenisation/truncation.
    """

    eos = getattr(tokenizer, "eos_token", "") or ""
    gold_with_eos = (gold_text + eos).strip("\n")
    full_text = prefix_text + gold_with_eos

    full = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
        add_special_tokens=True,
    )
    gold_enc = tokenizer(gold_with_eos, return_tensors="pt", add_special_tokens=False)
    gold_ids = gold_enc["input_ids"]
    n_gold = int(gold_ids.shape[1])

    n_full = int(full["input_ids"].shape[1])
    if n_full < n_gold + 1:
        return float("inf"), n_full

    device = next(model.parameters()).device
    input_ids = full["input_ids"].to(device)
    attn = full.get("attention_mask")
    attn = attn.to(device) if attn is not None else None

    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = out.logits[:, n_full - n_gold - 1 : n_full - 1, :]
    labels = input_ids[:, n_full - n_gold : n_full]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    per_tok = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    nll = float(-per_tok.mean().item())
    return float(math.exp(min(nll, 20.0))), n_full


@torch.no_grad()
def answer_span_ppl_from_embeds(
    model: Any,
    tokenizer: Any,
    prefix_embeds: torch.Tensor,
    gold_text: str,
    *,
    prefix_attn: torch.Tensor | None = None,
    max_input_tokens: int = 2048,
) -> tuple[float, int]:
    """Perplexity on the gold span when conditioned on an embedding-space prefix.

    Used for `rand_mem` / `pooled_mem` baselines where the "memory" lives in
    embedding space and not in the tokeniser vocabulary.
    """

    eos = getattr(tokenizer, "eos_token", "") or ""
    gold_with_eos = (gold_text + eos).strip("\n")
    gold_ids = tokenizer(gold_with_eos, return_tensors="pt", add_special_tokens=False)["input_ids"]
    n_gold = int(gold_ids.shape[1])
    if n_gold == 0:
        return float("inf"), int(prefix_embeds.shape[1])

    device = next(model.parameters()).device
    gold_ids = gold_ids.to(device)
    embed_table = model.get_input_embeddings()
    gold_embeds = embed_table(gold_ids)
    # Defensive dtype-align: callers occasionally pass f32 prefixes against a
    # bf16/fp16 model. Cast to the embedding table's dtype so the model's first
    # linear layer doesn't crash on mat1/mat2 dtype mismatch.
    prefix_embeds = prefix_embeds.to(device=device, dtype=gold_embeds.dtype)
    full_embeds = torch.cat([prefix_embeds, gold_embeds], dim=1)
    n_full = int(full_embeds.shape[1])
    if n_full > max_input_tokens:
        return float("inf"), n_full

    if prefix_attn is None:
        prefix_attn = torch.ones(
            prefix_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device
        )
    gold_attn = torch.ones(gold_ids.shape, dtype=prefix_attn.dtype, device=device)
    full_attn = torch.cat([prefix_attn, gold_attn], dim=1)

    out = model(inputs_embeds=full_embeds, attention_mask=full_attn, use_cache=False)
    logits = out.logits[:, n_full - n_gold - 1 : n_full - 1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    per_tok = log_probs.gather(-1, gold_ids.unsqueeze(-1)).squeeze(-1)
    nll = float(-per_tok.mean().item())
    return float(math.exp(min(nll, 20.0))), n_full


@torch.no_grad()
def build_rand_mem_prefix(
    model: Any, *, n_memory: int, std: float = 0.02, seed: int = 12345
) -> torch.Tensor:
    """Random `[1, K, d]` prefix; lower-bound baseline (no info, just gist slots).

    Returned dtype matches the model's input-embedding dtype so that
    ``model(inputs_embeds=...)`` does not raise the "expected mat1 and mat2 to
    have the same dtype" error on bf16/fp16 base models.
    """

    d = int(model.config.hidden_size)
    device = next(model.parameters()).device
    embed_dtype = model.get_input_embeddings().weight.dtype
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    # Sample in fp32 then cast: torch.randn() doesn't accept bf16 generators on CPU.
    sample = torch.randn(1, int(n_memory), d, generator=g, dtype=torch.float32) * std
    return sample.to(device=device, dtype=embed_dtype)


@torch.no_grad()
def build_pooled_mem_prefix(
    model: Any,
    tokenizer: Any,
    chunks: list[str],
    *,
    n_memory: int,
    max_chunk_tokens: int = 256,
) -> torch.Tensor:
    """Mean-pool each chunk's input-embedding into one vector; keep first K.

    The "zero-training upper bound" for embedding-space memory.
    """

    device = next(model.parameters()).device
    embed_table = model.get_input_embeddings()
    pooled: list[torch.Tensor] = []
    for chunk in chunks:
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=max_chunk_tokens,
            add_special_tokens=False,
        )
        ids = enc["input_ids"].to(device)
        if ids.shape[1] == 0:
            continue
        with torch.no_grad():
            embs = embed_table(ids)  # [1, T, d]
        attn = enc.get("attention_mask")
        if attn is not None:
            # Cast attn to embs.dtype (not always .float()) so the pooled vector
            # keeps the model's native dtype — bf16/fp16 models otherwise hit
            # the "mat1 and mat2 dtype" mismatch downstream.
            attn = attn.to(device=device, dtype=embs.dtype).unsqueeze(-1)
            embs = embs * attn
            denom = attn.sum(dim=1, keepdim=False).clamp_min(1.0)
        else:
            denom = torch.tensor(
                [float(embs.shape[1])], device=device, dtype=embs.dtype
            ).clamp_min(1.0)
        pooled.append(embs.sum(dim=1) / denom)  # [1, d], same dtype as embs
    if not pooled:
        return build_rand_mem_prefix(model, n_memory=n_memory)
    stacked = torch.stack(pooled, dim=1)  # [1, n_chunks, d]
    if stacked.shape[1] >= n_memory:
        return stacked[:, :n_memory, :]
    pad_count = n_memory - stacked.shape[1]
    pad = stacked.mean(dim=1, keepdim=True).expand(-1, pad_count, -1)
    return torch.cat([stacked, pad], dim=1)


def ppl_recovery_ratio(ppl_no_ctx: float, ppl_with_ctx: float, ppl_full: float) -> float:
    """(no - with) / (no - full). 0 = no recovery, 1 = matches full_ctx."""

    denom = ppl_no_ctx - ppl_full
    if abs(denom) < 1e-12:
        return 0.0
    return (ppl_no_ctx - ppl_with_ctx) / denom


def bits_saved_per_slot(ppl_no_ctx: float, ppl_with_ctx: float, n_memory: int) -> float:
    if n_memory <= 0 or ppl_no_ctx <= 0 or ppl_with_ctx <= 0:
        return 0.0
    return (math.log2(ppl_no_ctx) - math.log2(ppl_with_ctx)) / float(n_memory)


# ============================================================================ #
# C. Information distribution                                                  #
# ============================================================================ #


def gini(values: Iterable[float]) -> float:
    """Gini coefficient on non-negative magnitudes. 0 = uniform, 1 = max-skewed."""

    vals = sorted(abs(float(v)) for v in values)
    n = len(vals)
    s = sum(vals)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((2 * i - n - 1) * x for i, x in enumerate(vals, start=1))
    return float(cum / (n * s))


def top_k_coverage(values: list[float], k: int) -> float:
    """Fraction of total `|values|` carried by the top-k entries."""

    if not values:
        return 0.0
    mags = sorted((abs(v) for v in values), reverse=True)
    s = sum(mags)
    if s == 0:
        return 0.0
    return float(sum(mags[: max(0, k)]) / s)


def spearman_correlation(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Returns 0.0 if either side is constant (no
    ordering signal) or lengths mismatch.

    Uses average-rank for ties so constant inputs map to all-equal ranks
    and yield 0 correlation.
    """

    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        # Average-rank for tied values: dense ordering then average per group.
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        n = len(vals)
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _rank(list(xs))
    ry = _rank(list(ys))
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0 or den_y == 0:
        return 0.0
    return float(num / (den_x * den_y))


@torch.no_grad()
def per_chunk_contribution(
    model: Any,
    tokenizer: Any,
    item: LongContextItem,
    *,
    max_input_tokens: int = 2048,
) -> tuple[list[float], float]:
    """ΔPPL_i = PPL(full \\ chunk_i) - PPL(full) for every chunk.

    Returns `(contributions, ppl_full)`. Larger = removing the chunk hurts more.
    """

    full_text = "\n\n".join(item.chunks) + format_query_block(item.query)
    ppl_full, _ = answer_span_ppl_from_text(
        model, tokenizer, full_text, item.gold, max_input_tokens=max_input_tokens
    )
    contribs: list[float] = []
    for i in range(len(item.chunks)):
        ablated_chunks = item.chunks[:i] + item.chunks[i + 1 :]
        ablated = "\n\n".join(ablated_chunks) + format_query_block(item.query)
        ppl_ablated, _ = answer_span_ppl_from_text(
            model, tokenizer, ablated, item.gold, max_input_tokens=max_input_tokens
        )
        contribs.append(ppl_ablated - ppl_full)
    return contribs, ppl_full


def cross_chunk_redundancy(contribs: list[float], ppl_no_ctx: float, ppl_full: float) -> float:
    """sum(ΔPPL_i) / ΔPPL_all. > 1 = chunks redundant; ~1 = independent; < 1 = synergistic."""

    delta_all = ppl_no_ctx - ppl_full
    if abs(delta_all) < 1e-12:
        return 0.0
    return float(sum(contribs) / delta_all)


# ============================================================================ #
# D. Domain density                                                            #
# ============================================================================ #


@torch.no_grad()
def token_surprisals(
    model: Any, tokenizer: Any, text: str, *, max_tokens: int = 2048
) -> tuple[list[float], int]:
    """Per-position `-log p(token | prefix)` (in nats) over the full text.

    Returns `(surprisals, n_chars)` where the first token's surprisal is
    dropped (no prefix to condition on).
    """

    if not text:
        return [], 0
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    ids = enc["input_ids"]
    if ids.shape[1] < 2:
        return [], len(text)
    device = next(model.parameters()).device
    ids = ids.to(device)
    out = model(input_ids=ids, use_cache=False)
    log_probs = F.log_softmax(out.logits[:, :-1, :].float(), dim=-1)
    targets = ids[:, 1:]
    per_tok = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return per_tok[0].cpu().tolist(), len(text)


def bits_per_token_from_surprisals(surprisals_nats: list[float]) -> float:
    if not surprisals_nats:
        return 0.0
    return (sum(surprisals_nats) / len(surprisals_nats)) / math.log(2)


def bits_per_char_from_surprisals(text_chars: int, surprisals_nats: list[float]) -> float:
    if text_chars <= 0 or not surprisals_nats:
        return 0.0
    return (sum(surprisals_nats) / text_chars) / math.log(2)


def surprisal_heavy_tail(surprisals: list[float], lo_q: float = 0.5, hi_q: float = 0.99) -> float:
    """Ratio of high-quantile surprisal to median — heavy tail = bursty content."""

    if not surprisals:
        return 0.0
    t = torch.tensor(surprisals)
    lo = float(torch.quantile(t, lo_q).item())
    hi = float(torch.quantile(t, hi_q).item())
    if lo <= 0:
        return 0.0
    return hi / lo


def zipf_slope(surprisals: list[float]) -> float:
    """Slope of log(surprisal) vs log(rank). Steeper (more negative) = narrower vocabulary."""

    if len(surprisals) < 4:
        return 0.0
    sorted_s = sorted((s for s in surprisals if s > 0), reverse=True)
    if len(sorted_s) < 4:
        return 0.0
    xs = [math.log(i + 1) for i in range(len(sorted_s))]
    ys = [math.log(s) for s in sorted_s]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


# ============================================================================ #
# Orchestrator: one cell of (model × dataset × K)                              #
# ============================================================================ #


@dataclass
class CapacityCellResult:
    """All capacity metrics for a single (item-set, K) cell.

    Empty-init friendly: every numeric field defaults to 0/nan so partial runs
    serialise cleanly even when truncation kills some baselines.
    """

    # Setup
    n_items: int = 0
    n_memory: int = 0
    d_model: int = 0

    # A. Capacity
    a_compression_ratio_mean: float = 0.0
    a_bytes_per_slot_mean: float = 0.0
    a_theoretical_capacity_bits: float = 0.0

    # B. PPL recovery (means across items)
    b_ppl_no_ctx_mean: float = 0.0
    b_ppl_full_ctx_mean: float = 0.0
    b_ppl_rand_mem_mean: float = 0.0
    b_ppl_pooled_mem_mean: float = 0.0
    b_recovery_pooled_mem_mean: float = 0.0
    b_bits_saved_per_slot_pooled_mean: float = 0.0

    # C. Distribution (means across items)
    c_contribution_gini_mean: float = 0.0
    c_top25pct_chunk_coverage_mean: float = 0.0
    c_cross_chunk_redundancy_mean: float = 0.0
    c_position_importance_spearman_mean: float = 0.0

    # D. Domain density (corpus-level)
    d_bits_per_token_corpus: float = 0.0
    d_bits_per_char_corpus: float = 0.0
    d_surprisal_heavy_tail: float = 0.0
    d_zipf_slope: float = 0.0

    # Per-item raw values (kept for plotting; not for headline numbers)
    per_item: list[dict[str, Any]] = field(default_factory=list)


@torch.no_grad()
def run_capacity_panel(
    model: Any,
    tokenizer: Any,
    items: list[LongContextItem],
    *,
    n_memory: int,
    max_input_tokens: int = 2048,
    rand_mem_seed: int = 12345,
    max_chunk_tokens: int = 256,
) -> CapacityCellResult:
    """Compute every metric in the capacity panel for one (items, K) cell.

    Strategy is deliberately serial and CPU-friendly (we expect tiny LM for
    smoke and a single GPU for real Qwen / Gemma runs; throughput is not the
    point).
    """

    if not items:
        return CapacityCellResult(n_memory=n_memory)

    d_model = int(model.config.hidden_size)
    result = CapacityCellResult(
        n_items=len(items),
        n_memory=n_memory,
        d_model=d_model,
        a_theoretical_capacity_bits=theoretical_capacity_bits(n_memory, d_model),
    )

    per_item: list[dict[str, Any]] = []
    bytes_per_slot_list: list[float] = []
    cr_list: list[float] = []

    ppl_no_list: list[float] = []
    ppl_full_list: list[float] = []
    ppl_rand_list: list[float] = []
    ppl_pooled_list: list[float] = []
    recovery_pooled_list: list[float] = []
    bits_saved_pooled_list: list[float] = []

    gini_list: list[float] = []
    top25_list: list[float] = []
    redundancy_list: list[float] = []
    pos_corr_list: list[float] = []

    all_surprisals: list[float] = []
    all_chars = 0

    for it in items:
        full_text = "\n\n".join(it.chunks) + format_query_block(it.query)
        no_text = format_query_block(it.query)

        # ---- B: PPL under conditions
        ppl_no, n_no = answer_span_ppl_from_text(
            model, tokenizer, no_text, it.gold, max_input_tokens=max_input_tokens
        )
        ppl_full, n_full = answer_span_ppl_from_text(
            model, tokenizer, full_text, it.gold, max_input_tokens=max_input_tokens
        )

        rand_prefix = build_rand_mem_prefix(model, n_memory=n_memory, seed=rand_mem_seed)
        ppl_rand, n_rand = answer_span_ppl_from_embeds(
            model, tokenizer, rand_prefix, it.gold, max_input_tokens=max_input_tokens,
        )

        pooled_prefix = build_pooled_mem_prefix(
            model, tokenizer, it.chunks, n_memory=n_memory, max_chunk_tokens=max_chunk_tokens,
        )
        ppl_pooled, n_pooled = answer_span_ppl_from_embeds(
            model, tokenizer, pooled_prefix, it.gold, max_input_tokens=max_input_tokens,
        )

        ppl_no_list.append(ppl_no)
        ppl_full_list.append(ppl_full)
        ppl_rand_list.append(ppl_rand)
        ppl_pooled_list.append(ppl_pooled)
        if math.isfinite(ppl_no) and math.isfinite(ppl_full) and math.isfinite(ppl_pooled):
            recovery_pooled_list.append(ppl_recovery_ratio(ppl_no, ppl_pooled, ppl_full))
        if math.isfinite(ppl_no) and math.isfinite(ppl_pooled):
            bits_saved_pooled_list.append(bits_saved_per_slot(ppl_no, ppl_pooled, n_memory))

        # ---- A: capacity (per-item granularity)
        cr = compression_ratio(n_full, n_pooled)
        cr_list.append(cr if math.isfinite(cr) else 0.0)
        bytes_per_slot_list.append(bytes_per_memory_slot(full_text, n_memory, d_model))

        # ---- C: per-chunk distribution (skip if no chunks)
        if len(it.chunks) >= 2:
            contribs, _ = per_chunk_contribution(
                model, tokenizer, it, max_input_tokens=max_input_tokens
            )
            g = gini(contribs)
            gini_list.append(g)
            top_k = max(1, int(round(0.25 * len(contribs))))
            top25_list.append(top_k_coverage(contribs, top_k))
            if math.isfinite(ppl_no) and math.isfinite(ppl_full):
                redundancy_list.append(cross_chunk_redundancy(contribs, ppl_no, ppl_full))
            positions = list(range(len(contribs)))
            pos_corr_list.append(spearman_correlation(positions, contribs))
        else:
            contribs = []

        # ---- D: domain density on this item's full text
        surps, n_chars = token_surprisals(
            model, tokenizer, full_text, max_tokens=max_input_tokens
        )
        all_surprisals.extend(surps)
        all_chars += n_chars

        per_item.append({
            "item_id": it.item_id,
            "ppl_no_ctx": ppl_no,
            "ppl_full_ctx": ppl_full,
            "ppl_rand_mem": ppl_rand,
            "ppl_pooled_mem": ppl_pooled,
            "compression_ratio": cr,
            "n_full_ctx_tokens": n_full,
            "n_pooled_tokens": n_pooled,
            "per_chunk_contribution": contribs,
        })

    def _mean(xs: list[float]) -> float:
        finite = [x for x in xs if math.isfinite(x)]
        return sum(finite) / len(finite) if finite else 0.0

    result.a_compression_ratio_mean = _mean(cr_list)
    result.a_bytes_per_slot_mean = _mean(bytes_per_slot_list)

    result.b_ppl_no_ctx_mean = _mean(ppl_no_list)
    result.b_ppl_full_ctx_mean = _mean(ppl_full_list)
    result.b_ppl_rand_mem_mean = _mean(ppl_rand_list)
    result.b_ppl_pooled_mem_mean = _mean(ppl_pooled_list)
    result.b_recovery_pooled_mem_mean = _mean(recovery_pooled_list)
    result.b_bits_saved_per_slot_pooled_mean = _mean(bits_saved_pooled_list)

    result.c_contribution_gini_mean = _mean(gini_list)
    result.c_top25pct_chunk_coverage_mean = _mean(top25_list)
    result.c_cross_chunk_redundancy_mean = _mean(redundancy_list)
    result.c_position_importance_spearman_mean = _mean(pos_corr_list)

    result.d_bits_per_token_corpus = bits_per_token_from_surprisals(all_surprisals)
    result.d_bits_per_char_corpus = bits_per_char_from_surprisals(all_chars, all_surprisals)
    result.d_surprisal_heavy_tail = surprisal_heavy_tail(all_surprisals)
    result.d_zipf_slope = zipf_slope(all_surprisals)

    result.per_item = per_item

    logger.debug(
        f"capacity[K={n_memory}, n={len(items)}]: "
        f"PPL no={result.b_ppl_no_ctx_mean:.3f} full={result.b_ppl_full_ctx_mean:.3f} "
        f"pooled={result.b_ppl_pooled_mem_mean:.3f} PRR={result.b_recovery_pooled_mem_mean:.3f}"
    )
    return result


def cell_to_dict(result: CapacityCellResult) -> dict[str, Any]:
    return asdict(result)
