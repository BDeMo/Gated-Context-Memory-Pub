"""Self-Verifying Compressor: the encoder (compressor).

Architecture (v1.7, see training-design-2026-06-10.md):
  - The ENCODER is a SEPARATE TRAINABLE COPY of the base's first N transformer layers (init from the base
    weights; N is an ablation knob). It is NOT the shared base and NOT an adapter on it.
  - Input layout to the encoder: ``[ctx ; query ; K memory tokens]``. A causal + QUERY-MASK runs over it:
    masking the query (so the memory cannot attend the query) gives the UNCONDITIONAL memory M0; unmasking
    gives the CONDITIONAL memory Mq. Reading the K memory positions' final hidden states gives M.
  - The query-mask is encoder-internal ONLY: it decides what the COMPRESSION saw. The frozen base always
    receives the query at answer time (``[M ; query]``, M replacing the context) in both modes; that base
    forward + the do-no-harm gate live in the trainer/runtime, not here.
  - ``m_proj`` maps M into the base's input-embedding space so it can be injected as a soft prompt.

Eager attention is forced on the encoder copy so the custom 4D mask is honored (verified to work on tf 4.56).
The DECODER (reconstruction = the lossless L_uncond loss) is implemented here (``reconstruct``); the
verifier/gate (ΔCode/ΔLogits + structural signals) is added next.
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_encoder(base_model: nn.Module, n_layers: int, trainable: bool = True,
                  dtype: torch.dtype | None = None, init: str = "copy") -> nn.Module:
    """A trainable copy of the base's first ``n_layers`` transformer blocks (+ embeddings, final norm, rotary).
    Uses the base's own inner-model class + forward (handles RoPE / masks) for cross-version robustness.
    ``dtype`` overrides the parameter dtype (use float32 for a stable AdamW master copy).
    ``init``: "copy" loads the base's first-N block weights (strong init); "random" leaves the transformer
    layers randomly initialized (ablation: does the base init matter?). Embeddings/norm/rotary always copied."""
    cfg = deepcopy(base_model.config)
    cfg.num_hidden_layers = int(n_layers)
    cfg._attn_implementation = "eager"  # required so our custom 4D mask is honored
    inner_cls = type(base_model.model)  # e.g. Qwen3Model / LlamaModel
    enc = inner_cls(cfg)
    # copy weights: keep non-layer params (embed_tokens, norm, rotary) + (init=="copy") layers [0, n_layers)
    src = base_model.model.state_dict()
    keep: dict[str, torch.Tensor] = {}
    for k, v in src.items():
        if k.startswith("layers."):
            if init == "copy" and int(k.split(".")[1]) < n_layers:
                keep[k] = v
        else:
            keep[k] = v
    missing, unexpected = enc.load_state_dict(keep, strict=False)
    enc = enc.to(dtype=dtype or base_model.get_input_embeddings().weight.dtype,
                 device=base_model.get_input_embeddings().weight.device)
    for p in enc.parameters():
        p.requires_grad_(trainable)
    # Freeze the copied input embedding: it is a copy of the base's and should not be trained (huge, overfits
    # on small per-corpus data). We train the N transformer blocks (+ final norm), which is "the encoder".
    enc.get_input_embeddings().weight.requires_grad_(False)
    enc._svc_copy_report = {"missing": len(missing), "unexpected": len(unexpected)}  # for smoke
    return enc


class SelfVerifyingCompressor(nn.Module):
    def __init__(self, base_model: nn.Module, n_layers: int = 4, n_memory: int = 64,
                 n_dec_layers: int = 2, train_dtype: torch.dtype | None = None, init: str = "copy",
                 m_norm_match: str = "off", m_manifold_temp: float = 0.0):
        super().__init__()
        self.K = int(n_memory)
        self.n_layers = int(n_layers)
        self.n_dec_layers = int(n_dec_layers)
        self.m_norm_match = str(m_norm_match)   # A1 (v1.7.1.1): off / hard / learn -- fix M's OOD norm
        self.m_manifold_temp = float(m_manifold_temp)  # A2 (v1.7.1.2): >0 => project M onto token-embedding hull
        self.encoder = build_encoder(base_model, n_layers, trainable=True, dtype=train_dtype, init=init)
        d = base_model.config.hidden_size
        dev = base_model.get_input_embeddings().weight.device
        pdt = train_dtype or base_model.get_input_embeddings().weight.dtype
        # K learnable memory tokens (the compression queries / slots), appended after [ctx; query].
        self.mem_tokens = nn.Parameter((torch.randn(self.K, d) * 0.02).to(dev, pdt))
        # project the read-out memory into the base's input-embedding space (for soft-prompt injection).
        self.m_proj = nn.Linear(d, d, bias=False).to(dev, pdt)
        nn.init.eye_(self.m_proj.weight)
        # A1: target scale = mean L2 norm of the base's input embeddings. M is injected as an input embed, so
        # its norm should sit on that manifold; m_raw is post-RMSNorm hidden (~5-10x too big) => OOD without this.
        e0 = base_model.get_input_embeddings().weight.detach().float().norm(dim=-1).mean().to(dev, pdt).reshape(1)
        if self.m_norm_match == "learn":
            self.embed_scale = nn.Parameter(e0.clone())
        else:
            self.register_buffer("embed_scale", e0)
        # DECODER (autoencoder): a trainable copy of the base's first n_dec_layers blocks that must
        # reconstruct the context FROM the memory. Its CE is the lossless (L_uncond) signal. n_dec_layers=0
        # disables reconstruction. The vocab projection is the (frozen) tied base embedding, passed at call.
        self.decoder = build_encoder(base_model, n_dec_layers, trainable=True, dtype=train_dtype) \
            if n_dec_layers > 0 else None

    @property
    def dtype(self) -> torch.dtype:
        return self.mem_tokens.dtype

    @property
    def device(self) -> torch.device:
        return self.mem_tokens.device

    def _enc_mask(self, lc: int, lq: int, conditional: bool) -> torch.Tensor:
        """4D additive mask over [ctx(lc) ; query(lq) ; mem(K)]: causal everywhere, and (unconditional only)
        the memory rows cannot attend the query columns. mem always attends ctx; query never sees mem (causal)."""
        L = lc + lq + self.K
        mn = torch.finfo(self.dtype).min
        m = torch.triu(torch.full((L, L), mn, device=self.device, dtype=self.dtype), diagonal=1)  # causal
        if not conditional and lq > 0:
            m[lc + lq:, lc:lc + lq] = mn  # mem (rows >= lc+lq) cannot attend query cols [lc, lc+lq)
        return m.view(1, 1, L, L)

    def _encode_one(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, condition_on_query: bool) -> torch.Tensor:
        """One encoder pass over [ctx ; query ; K mem] -> the K memory-position hiddens (1, K, d)."""
        embed = self.encoder.get_input_embeddings()
        ce, qe = embed(ctx_ids), embed(query_ids)
        lc, lq = ce.shape[1], qe.shape[1]
        mem = self.mem_tokens.unsqueeze(0).to(ce.dtype)
        seq = torch.cat([ce, qe, mem], dim=1)
        attn = self._enc_mask(lc, lq, condition_on_query)
        pos = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
        out = self.encoder(inputs_embeds=seq, attention_mask=attn, position_ids=pos, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden[:, -self.K:, :]             # (1, K, d)

    def _readout(self, m_raw: torch.Tensor) -> torch.Tensor:
        """Project the memory-position hiddens into the base's input-embedding space (+ A1/A2 OOD fixes)."""
        m_base = self.m_proj(m_raw)               # (1, *, d) in the base's input-embedding space
        if self.m_manifold_temp > 0:              # A2: project each M token onto the token-embedding convex hull
            E = self.encoder.get_input_embeddings().weight   # (V,d) frozen base embeddings (keep dtype; no fp32 copy)
            attn = ((m_base @ E.t()) / self.m_manifold_temp).float().softmax(-1).to(m_base.dtype)
            m_base = attn @ E                                # convex combo of real embeddings => in-distribution
        elif self.m_norm_match != "off":          # A1: L2-normalize each M token to the embedding scale
            m_base = m_base / m_base.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.embed_scale.to(m_base.dtype)
        return m_base

    def encode(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, condition_on_query: bool = True) -> dict:
        """ctx_ids (1,Lc), query_ids (1,Lq) -> memory. condition_on_query=False masks the query (M0)."""
        m_raw = self._encode_one(ctx_ids, query_ids, condition_on_query)   # (1, K, d)
        return {
            "memory": self._readout(m_raw),       # inject before the query into the FROZEN base
            "memory_raw": m_raw,                   # pre-projection (for the decoder / signals)
            "mode": "conditioned" if condition_on_query else "agnostic",
        }

    def encode_chunked(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, chunk_size: int,
                       condition_on_query: bool = True, keep: int = 0) -> dict:
        """LENGTH-ADAPTIVE memory (v1.7.4): split ctx into chunks of ``chunk_size`` tokens, encode EACH chunk
        independently with its own K slots (per-chunk locality), and concatenate -> M of length S*K where
        S=ceil(Lc/chunk). The compression RATIO (chunk/K) is fixed, but the budget SCALES with length, removing
        the single fixed-K bottleneck that kills long context. ``keep>0`` => query-relevance ROUTING: keep only
        the top-``keep`` chunk-memories (soft retrieval; bounds the budget at keep*K for very long ctx)."""
        Lc = int(ctx_ids.shape[1])
        if chunk_size <= 0 or Lc <= chunk_size:                # short ctx => identical to single-pass encode
            return self.encode(ctx_ids, query_ids, condition_on_query)
        raws = [self._encode_one(ctx_ids[:, i:i + chunk_size], query_ids, condition_on_query)
                for i in range(0, Lc, chunk_size)]             # list of (1, K, d), one per chunk
        if keep and len(raws) > keep:                          # query-relevance routing (soft retrieval)
            with torch.no_grad():
                qe = self.encoder.get_input_embeddings()(query_ids).mean(1).float()   # (1, d)
                qn = qe / qe.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                score = [float((qn * (self._readout(r).mean(1).float() /
                                      self._readout(r).mean(1).float().norm(dim=-1, keepdim=True).clamp_min(1e-6))
                                ).sum()) for r in raws]
            idx = sorted(sorted(range(len(raws)), key=lambda i: score[i], reverse=True)[:keep])
            raws = [raws[i] for i in idx]
        m_raw = torch.cat(raws, dim=1)                         # (1, S*K, d)
        return {
            "memory": self._readout(m_raw),
            "memory_raw": m_raw,
            "mode": "conditioned" if condition_on_query else "agnostic",
            "n_chunks": len(raws),
        }

    def forward(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, condition_on_query: bool = True) -> dict:
        return self.encode(ctx_ids, query_ids, condition_on_query)

    def reconstruct(self, ctx_ids: torch.Tensor, memory: torch.Tensor, out_weight: torch.Tensor,
                    m_only: bool = False) -> torch.Tensor:
        """L_uncond: teacher-forced reconstruction of the context FROM the memory through the trainable decoder.
        ``memory`` (1,K,d) is the (unconditional) M; ctx tokens attend M + earlier ctx (causal). Returns CE.
        ``out_weight`` is the frozen tied LM head (base input-embedding weight) used for the vocab projection.
        C1 (v1.7.3.1) ``m_only``: block ctx->ctx attention so each ctx position can attend ONLY M -- the decoder
        cannot use language-modeling on earlier ctx, so reconstruction success TRULY measures M's information."""
        assert self.decoder is not None, "reconstruct requires n_dec_layers>0"
        cemb = self.decoder.get_input_embeddings()(ctx_ids)        # (1, Lc, d) decoder's own (frozen) embed
        m = memory.to(cemb.dtype)
        seq = torch.cat([m, cemb], dim=1)                          # [M ; ctx]
        L = seq.shape[1]
        mn = torch.finfo(seq.dtype).min
        am = torch.triu(torch.full((L, L), mn, device=seq.device, dtype=seq.dtype), diagonal=1).view(1, 1, L, L)
        if m_only:                                                 # C1: ctx rows attend ONLY the K memory cols
            am[:, :, m.shape[1]:, m.shape[1]:] = mn
        pos = torch.arange(L, device=seq.device).unsqueeze(0)
        out = self.decoder(inputs_embeds=seq, attention_mask=am, position_ids=pos, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        Lc = ctx_ids.shape[1]
        hc = h[:, m.shape[1] - 1:m.shape[1] + Lc - 1, :]           # positions predicting ctx[0..Lc-1]
        logits = F.linear(hc.to(out_weight.dtype), out_weight).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ctx_ids.reshape(-1))
