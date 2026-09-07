r"""Geometry probes on `MemoryState.extra["last_delta"]` and chunk encodings.

These probes test whether the recurrence

    m_t = m_{t-1} + α · Δm_t

learns a meaningful "translation vector" for each chunk — i.e. whether the
latent space behaves like the additive structure of a word-embedding space
(``vec(king) - vec(man) ≈ vec(queen) - vec(woman)``).

Each probe:

- consumes a list of `MemoryState`s sampled along the recurrence (one per
  chunk), along with the chunk encodings that produced them;
- returns a dict of scalar / list metrics that map onto the *E* section of
  the metric panel.

Probes never call the base model; everything is geometry on tensors the
wrapper already exposed. Cheap (~ms / item).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from llm_infra.wrappers import ChunkEncoding, MemoryState


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _pool_chunk(chunk_hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool over the chunk's token axis, respecting `mask`. `[B, T, d] -> [B, d]`."""

    if mask is None:
        return chunk_hidden.mean(dim=1)
    m = mask.unsqueeze(-1).to(chunk_hidden.dtype)
    s = (chunk_hidden * m).sum(dim=1)
    n = m.sum(dim=1).clamp_min(1.0)
    return s / n


def _flatten_state(state: torch.Tensor) -> torch.Tensor:
    """`[B, K, d] -> [B, K·d]`."""

    return state.reshape(state.shape[0], -1)


# --------------------------------------------------------------------------- #
# Per-step Δ statistics                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class DeltaNormStats:
    """Distribution of `‖Δm_t‖₂` across steps."""

    mean: float
    std: float
    p50: float
    p90: float
    p99: float
    heavy_tail_ratio: float


def delta_norm_stats(deltas: list[torch.Tensor]) -> DeltaNormStats:
    """`deltas[t]` shape `[B, K, d]`; flattens K·d before taking the L2 norm."""

    if not deltas:
        return DeltaNormStats(0, 0, 0, 0, 0, 0)
    norms: list[float] = []
    for d in deltas:
        flat = _flatten_state(d)
        for b in range(flat.shape[0]):
            norms.append(float(flat[b].norm().item()))
    if not norms:
        return DeltaNormStats(0, 0, 0, 0, 0, 0)
    t = torch.tensor(norms)
    p50 = float(torch.quantile(t, 0.50).item())
    p90 = float(torch.quantile(t, 0.90).item())
    p99 = float(torch.quantile(t, 0.99).item())
    hh = p99 / p50 if p50 > 0 else 0.0
    return DeltaNormStats(
        mean=float(t.mean().item()),
        std=float(t.std(unbiased=False).item()),
        p50=p50,
        p90=p90,
        p99=p99,
        heavy_tail_ratio=hh,
    )


def drift_curve(states: list[torch.Tensor]) -> list[float]:
    """`‖m_t - m_0‖_2` along the recurrence. Sub-linear growth = saturating
    memory (healthy). Linear growth = unbounded drift (bad)."""

    if len(states) < 1:
        return []
    base = _flatten_state(states[0])
    curve: list[float] = []
    for s in states:
        cur = _flatten_state(s)
        diff = cur - base
        curve.append(float(diff.norm(dim=-1).mean().item()))
    return curve


def delta_chunk_cosine(
    deltas: list[torch.Tensor], chunks: list[ChunkEncoding]
) -> list[float]:
    """For each step, cos similarity between Δm_t (pooled over K) and the
    pooled chunk encoding. Healthy wrappers learn ``cos > 0.3`` after training.
    """

    if len(deltas) != len(chunks):
        raise ValueError(
            f"deltas / chunks length mismatch: {len(deltas)} vs {len(chunks)}"
        )
    cosines: list[float] = []
    for d, c in zip(deltas, chunks):
        d_pool = d.mean(dim=1)  # [B, d_model]
        c_pool = _pool_chunk(c.hidden, c.mask)  # [B, d_model]
        # Resize if dims differ (encoder may project differently)
        if d_pool.shape[-1] != c_pool.shape[-1]:
            cosines.append(0.0)
            continue
        sim = F.cosine_similarity(d_pool, c_pool, dim=-1)
        cosines.append(float(sim.mean().item()))
    return cosines


def consecutive_delta_cosine(deltas: list[torch.Tensor]) -> list[float]:
    """cos(Δm_t, Δm_{t-1}). High = redundant updates; low = uncorrelated.
    Healthy range 0.1-0.4 after training."""

    if len(deltas) < 2:
        return []
    cosines: list[float] = []
    for t in range(1, len(deltas)):
        a = _flatten_state(deltas[t])
        b = _flatten_state(deltas[t - 1])
        sim = F.cosine_similarity(a, b, dim=-1)
        cosines.append(float(sim.mean().item()))
    return cosines


# --------------------------------------------------------------------------- #
# Latent geometry                                                             #
# --------------------------------------------------------------------------- #


def latent_variance_concentration(states: list[torch.Tensor]) -> float:
    """top-singular-value / total-singular-energy of the collected m_T matrix.

    < 0.3 = healthy (uses many dims); > 0.7 = latent collapsing to a 1D axis.
    """

    if not states:
        return 0.0
    flats = torch.cat([_flatten_state(s) for s in states], dim=0)
    if flats.shape[0] < 2:
        return 0.0
    centered = flats - flats.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(centered.float())
    except RuntimeError:
        return 0.0
    if s.numel() == 0 or s.sum() == 0:
        return 0.0
    return float((s[0] ** 2 / (s ** 2).sum()).item())


def latent_temporal_variance(states: list[torch.Tensor]) -> float:
    """var_t(m_t) / var_t(m_0). > 1 = state evolves over time; ≈ 1 = no update.

    Computed as mean per-dim variance across time, normalised by the per-dim
    variance of the initial state (taken across the batch dim)."""

    if len(states) < 2 or states[0].numel() == 0:
        return 0.0
    stacked = torch.stack([_flatten_state(s) for s in states], dim=0)  # [T, B, K·d]
    init_flat = _flatten_state(states[0])  # [B, K·d]
    var_init = init_flat.var(dim=0, unbiased=False).mean()
    var_time = stacked.var(dim=0, unbiased=False).mean()
    if float(var_init.item()) < 1e-12:
        return 0.0
    return float((var_time / var_init).item())


def analogy_score(
    states_a: list[torch.Tensor],
    states_b: list[torch.Tensor],
    states_ab: list[torch.Tensor],
) -> float:
    """Vector-arithmetic analogy: is `m(a → b)` (running b after a) close to
    `m(a) + Δm(b alone)`?

    All three lists are state trajectories. We compute, at every matched step
    t with t ≥ 1::

        lhs = states_ab[t] - states_a[t-1]   # the "joint" residual ending in b
        rhs = states_b[t] - states_b[t-1]    # the "b-only" residual

    and average their cosine similarity. > 0.6 ≈ word-embedding-style additive
    structure (passes basic analogy probe).
    """

    if not (len(states_a) == len(states_b) == len(states_ab)):
        return 0.0
    T = len(states_a)
    if T < 2:
        return 0.0
    cosines: list[float] = []
    for t in range(1, T):
        lhs = _flatten_state(states_ab[t] - states_a[t - 1])
        rhs = _flatten_state(states_b[t] - states_b[t - 1])
        sim = F.cosine_similarity(lhs, rhs, dim=-1)
        cosines.append(float(sim.mean().item()))
    return sum(cosines) / len(cosines) if cosines else 0.0


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class GeometryProbeResult:
    """E-panel summary for one (wrapper, dataset) cell."""

    # E1
    delta_norm_mean: float = 0.0
    delta_norm_p50: float = 0.0
    delta_norm_p99: float = 0.0
    delta_norm_heavy_tail: float = 0.0

    # E2
    drift_final: float = 0.0
    drift_per_step_slope: float = 0.0  # linear-fit slope of drift curve

    # E3
    delta_chunk_cosine_mean: float = 0.0

    # E4
    consecutive_delta_cosine_mean: float = 0.0

    # E5 (skipped if no analogy traces provided)
    analogy_score: float = 0.0

    # E6
    latent_variance_concentration: float = 0.0

    # E7
    latent_temporal_variance: float = 0.0


def _linear_slope(ys: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def run_geometry_panel(
    wrapper: Any,
    encoder_fn: Any,
    chunks_text: list[str],
    *,
    device: torch.device | str = "cpu",
    analogy_chunks: tuple[list[str], list[str]] | None = None,
) -> tuple[GeometryProbeResult, dict[str, Any]]:
    """Run the wrapper across `chunks_text`, collect Δm / m trajectories,
    compute every E-panel metric.

    `analogy_chunks` is an optional (chunks_a, chunks_b) pair used for the
    analogy probe. When provided, this function runs the wrapper three times
    (a, b, a∘b) and computes E5.

    Returns `(result, raw)` where `raw` is the trajectory dict for later
    plotting.
    """

    device_t = torch.device(device) if isinstance(device, str) else device
    states: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    chunk_encs: list[ChunkEncoding] = []

    def _as_tensor(payload: object) -> torch.Tensor | None:
        # Wrappers that follow the additive-recurrence contract expose payload
        # as a (B, K, d) Tensor. Baselines such as GistWrapper use a custom
        # dataclass payload (no .detach), so the geometry panel is moot for
        # them — bail out cleanly instead of crashing the whole run.
        return payload if isinstance(payload, torch.Tensor) else None

    mem: MemoryState = wrapper.init_memory(batch=1, device=device_t)
    init_state = _as_tensor(mem.payload)
    if init_state is None:
        return GeometryProbeResult(), {"states": [], "deltas": [], "chunks": []}
    states.append(init_state.detach().clone())
    for chunk in chunks_text:
        enc = encoder_fn([chunk])
        chunk_encs.append(enc)
        mem = wrapper.update(mem, enc)
        state_t = _as_tensor(mem.payload)
        if state_t is None:
            return GeometryProbeResult(), {"states": [], "deltas": [], "chunks": []}
        states.append(state_t.detach().clone())
        if "last_delta" in mem.extra and isinstance(mem.extra["last_delta"], torch.Tensor):
            deltas.append(mem.extra["last_delta"].detach().clone())

    if not deltas:
        return GeometryProbeResult(), {"states": [], "deltas": [], "chunks": []}

    d_stats = delta_norm_stats(deltas)
    drift = drift_curve(states)
    cos_chunk = delta_chunk_cosine(deltas, chunk_encs)
    cos_delta = consecutive_delta_cosine(deltas)

    analogy = 0.0
    if analogy_chunks is not None:
        a_chunks, b_chunks = analogy_chunks
        if len(a_chunks) == len(b_chunks) and a_chunks:
            states_a = _trajectory(wrapper, encoder_fn, a_chunks, device_t)
            states_b = _trajectory(wrapper, encoder_fn, b_chunks, device_t)
            states_ab = _trajectory(wrapper, encoder_fn, list(a_chunks) + list(b_chunks), device_t)
            # Align: compare states_b[t+1] (= a_b's b-portion) with the
            # corresponding slice of states_ab — only need same length b runs.
            n = min(len(states_b), len(states_ab))
            analogy = analogy_score(states_a[:n], states_b[:n], states_ab[:n])

    result = GeometryProbeResult(
        delta_norm_mean=d_stats.mean,
        delta_norm_p50=d_stats.p50,
        delta_norm_p99=d_stats.p99,
        delta_norm_heavy_tail=d_stats.heavy_tail_ratio,
        drift_final=drift[-1] if drift else 0.0,
        drift_per_step_slope=_linear_slope(drift),
        delta_chunk_cosine_mean=(sum(cos_chunk) / len(cos_chunk)) if cos_chunk else 0.0,
        consecutive_delta_cosine_mean=(sum(cos_delta) / len(cos_delta)) if cos_delta else 0.0,
        analogy_score=analogy,
        latent_variance_concentration=latent_variance_concentration(states),
        latent_temporal_variance=latent_temporal_variance(states),
    )
    raw = {
        "drift_curve": drift,
        "delta_chunk_cosine": cos_chunk,
        "consecutive_delta_cosine": cos_delta,
        "delta_norms": [float(_flatten_state(d).norm(dim=-1).mean().item()) for d in deltas],
    }
    return result, raw


def _trajectory(
    wrapper: Any, encoder_fn: Any, chunks: list[str], device: torch.device
) -> list[torch.Tensor]:
    mem: MemoryState = wrapper.init_memory(batch=1, device=device)
    out = [mem.payload.detach().clone()]
    for ch in chunks:
        enc = encoder_fn([ch])
        mem = wrapper.update(mem, enc)
        out.append(mem.payload.detach().clone())
    return out
