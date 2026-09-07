"""Self-compressor: the frozen base model is its own compressor.

Principle ("the model's own features are its own compressor"): we do NOT keep a separate trainable copy of
the base's layers. Instead we run the *frozen* base over ``[ctx ; query ; K learnable memory tokens]`` through
its normal forward pipeline; the base's own hidden features at the K memory positions (read after ``depth``
layers) are passed through a small **nonlinear** projection and **normalized onto the input-embedding manifold**
to become the compressed memory ``M``.

Trainable parameters: the K memory tokens (learnable "compression queries") + the projection MLP. The base is
frozen and shared with the read/answer path, so there are no duplicated weights.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

from .lora import set_enc_lora_enabled


class SelfCompressor(nn.Module):
    def __init__(self, base: nn.Module, n_memory: int = 16, depth: int | None = None,
                 proj_mult: int = 4, normalize: bool = True, norm_mode: str = "", manifold_temp: float = 1.0,
                 vae: bool = False, chunk_size: int = 0, xchunk: int = 0, recur: bool = False,
                 state_cap: int = 0) -> None:
        super().__init__()
        self.base = base                                   # frozen, SHARED with the read path
        for p in self.base.parameters():
            p.requires_grad_(False)
        cfg = base.config
        d = int(cfg.hidden_size)
        n_layers = int(getattr(cfg, "num_hidden_layers", None) or len(base.model.layers))
        self.d = d
        self.K = int(n_memory)
        self.depth = int(depth) if depth else n_layers     # read base features after this many layers (full = last)
        self.normalize = bool(normalize)
        self.norm_mode = (norm_mode or ("hard" if normalize else "off")).lower()  # off|hard|learn|manifold
        self.manifold_temp = float(manifold_temp)
        self.vae = bool(vae)                               # VAE: proj -> (mu, logvar); M = mu + eps*sigma; + KL prior
        self.chunk_size = int(chunk_size)                  # >0 => length-adaptive chunked memory (M scales with len)
        self.xchunk = int(xchunk)                          # >0 => cross-chunk refine: mix per-chunk memories through N frozen base layers
        self.recur = bool(recur)                           # True => recurrent encode: carry prior-chunk summaries as a soft prefix
        self.state_cap = int(state_cap)                    # 0 => legacy S*K concat; >0 => bounded recurrent state/output
        if self.state_cap < 0:
            raise ValueError("state_cap must be non-negative")
        if self.state_cap and not self.recur:
            raise ValueError("state_cap requires recurrent chunked encoding")
        self.last_kl = None                                # most-recent VAE KL (read by the trainer)
        self.use_enc_lora = False                          # enable the encode-phase LoRA during compression
        dev = base.get_input_embeddings().weight.device
        pdt = base.get_input_embeddings().weight.dtype
        # K learnable "compression query" tokens, appended after [ctx ; query]
        self.mem = nn.Parameter((torch.randn(self.K, d) * 0.02).to(dev, pdt))
        # NONLINEAR readout: hidden features -> injectable embedding (real capacity now that the base is frozen)
        # VAE => the readout emits 2d (mu || logvar) per memory slot.
        h = int(proj_mult * d)
        out_d = 2 * d if self.vae else d
        self.proj = nn.Sequential(nn.Linear(d, h, bias=True), nn.GELU(), nn.Linear(h, out_d, bias=True)).to(dev, pdt)
        # target scale = mean L2 norm of the base's input embeddings, so M sits ON the embedding manifold
        e0 = base.get_input_embeddings().weight.detach().float().norm(dim=-1).mean()
        if self.norm_mode == "learn":
            self.embed_scale = nn.Parameter(e0.clone())
        else:
            self.register_buffer("embed_scale", e0)

    @contextlib.contextmanager
    def _enc_phase(self):
        """Activate the encode-phase LoRA (if any) for the duration of a compression forward, then restore."""
        set_enc_lora_enabled(self.base, self.use_enc_lora)
        try:
            yield
        finally:
            set_enc_lora_enabled(self.base, False)

    def _mask(self, lc: int, lq: int, conditional: bool, device, dtype, lp: int = 0,
              tail_pad: int = 0) -> torch.Tensor:
        """4D additive mask over [prefix(lp) ; ctx(lc) ; query(lq) ; mem(K)]: causal everywhere; for the UNCONDITIONAL
        memory (conditional=False) the K memory rows cannot attend the query columns. lp = soft-prefix length (prior
        chunk summaries carried forward in the recurrent encoder); the prefix is attendable causally by everything."""
        L = lp + lc + lq + self.K + tail_pad
        mn = torch.finfo(dtype).min
        m = torch.triu(torch.full((L, L), mn, device=device, dtype=dtype), diagonal=1)
        if not conditional and lq > 0:
            qs = lp + lc
            m[qs + lq:qs + lq + self.K, qs:qs + lq] = mn
        # PyTorch's memory-efficient SDPA backward requires the attention-bias row stride
        # (strideM) to be a multiple of 4. Preserve the logical LxL mask while placing it
        # in aligned backing storage; no token or attention edge is added.
        aligned = (L + 3) // 4 * 4
        if aligned != L:
            storage = torch.full((L, aligned), mn, device=device, dtype=dtype)
            storage[:, :L] = m
            m = storage[:, :L]
        return m.unsqueeze(0).unsqueeze(0)

    def _norm_m(self, m):
        """Apply the M->embedding-manifold alignment. modes: off|hard|learn|manifold(=convex-hull projection)."""
        mode = self.norm_mode
        if mode == "off":
            return m
        if mode == "manifold":                              # A2: convex combination of REAL token embeddings
            E = self.base.get_input_embeddings().weight     # (V, d)
            attn = ((m @ E.t()) / self.manifold_temp).float().softmax(-1).to(m.dtype)
            return attn @ E
        # hard|learn: project onto sphere of radius embed_scale. eps INSIDE the sqrt keeps the
        # gradient finite even when ||m||->0 (norm().backward() divides by the UNclamped norm -> NaN)
        inv = torch.rsqrt((m.float() * m.float()).sum(-1, keepdim=True) + 1e-12).to(m.dtype)
        return m * inv * self.embed_scale.to(m.dtype)

    def _project(self, m_raw):
        """proj readout (+ VAE reparam) + manifold alignment. Sets self.last_kl (VAE only). Samples M only when
        grad is enabled (training); eval uses the mean mu (deterministic)."""
        p = self.proj(m_raw)
        if self.vae:
            mu, logvar = p[..., :self.d], p[..., self.d:].clamp(-8.0, 8.0)
            if torch.is_grad_enabled():
                m = mu + torch.randn_like(mu) * (0.5 * logvar).exp()       # reparameterization trick
            else:
                m = mu
            # KL(q(z|x)=N(mu,sigma^2) || N(0,I)): SUM over latent dim, MEAN over (batch, K) -- standard convention,
            # so lam_kl is on the usual O(1e-3) scale instead of being ~d x too small (a plain .mean() divides by d).
            self.last_kl = (-0.5 * (1.0 + logvar.float() - mu.float().pow(2) - logvar.float().exp())).sum(-1).mean()
        else:
            self.last_kl = None
            m = p
        return self._norm_m(m)

    def _encode_raw(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, conditional: bool,
                    prefix: torch.Tensor | None = None) -> torch.Tensor:
        """One frozen-base pass over [prefix ; ctx ; query ; K mem] -> the K memory-position hiddens (1, K, d).
        ``prefix`` (recurrent encoder): already-encoded prior-chunk summaries as soft tokens (embedding space), so the
        K mem slots compress THIS chunk while attending to the running summary of all previous chunks (causal cross-chunk)."""
        embed = self.base.get_input_embeddings()
        ce, qe = embed(ctx_ids), embed(query_ids)
        lc, lq = ce.shape[1], qe.shape[1]
        mem = self.mem.unsqueeze(0).to(ce.dtype)
        lp = int(prefix.shape[1]) if prefix is not None else 0
        parts = ([prefix.to(ce.dtype)] if prefix is not None else []) + [ce, qe, mem]
        seq = torch.cat(parts, dim=1)
        # SDPA's fused backward requires the actual sequence length (and therefore its
        # attention-bias row stride) to be a multiple of four. Append up to three causal
        # future slots; they cannot affect the earlier memory positions.
        tail_pad = (-int(seq.shape[1])) % 4
        if tail_pad:
            seq = torch.cat([seq, torch.zeros(1, tail_pad, seq.shape[-1], device=seq.device, dtype=seq.dtype)], 1)
        mask = self._mask(lc, lq, conditional, seq.device, seq.dtype, lp=lp, tail_pad=tail_pad)
        pos = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
        mem_start = lp + lc + lq
        # MEMORY/COMPUTE: only run the first `depth` layers (we read M after `depth` layers, so layers depth+1..N are
        # pure waste — they were the reason long-context TRAINING OOM'd, since their activations were stored for backprop).
        # Temporarily truncating the layer list yields the IDENTICAL hidden at `depth` but with cost ~ depth/N_layers.
        layers = self.base.model.layers
        # run depth+1 layers (HF appends the POST-final-norm hidden as the last entry, so we keep one extra raw layer
        # and read hidden_states[depth] = the RAW intermediate after `depth` layers -- bit-identical to the full run).
        trunc = 0 < int(self.depth) + 1 < len(layers)
        # OFFLOAD: for FULL-CONTEXT (no-chunk) encoding of long ctx, the linear-attn torch fallback stores huge
        # activations for backward and OOMs a single GPU. save_on_cpu() parks those saved tensors on CPU during the
        # encode forward -> the whole context is attended in ONE pass (cross-token aware) at ~const GPU memory.
        import os as _os, contextlib as _cl
        offload = _os.environ.get("GCM_ENC_OFFLOAD", "0").lower() in ("1", "true", "on") and torch.is_grad_enabled()
        _cm = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else _cl.nullcontext()
        with self._enc_phase(), _cm:
            if trunc:
                saved = self.base.model.layers
                self.base.model.layers = saved[: int(self.depth) + 1]
                try:
                    out = self.base.model(inputs_embeds=seq, attention_mask=mask, position_ids=pos,
                                          use_cache=False, output_hidden_states=True)
                    h = out.hidden_states[self.depth]          # raw hidden after `depth` layers (pre-norm intermediate)
                finally:
                    self.base.model.layers = saved
            else:
                out = self.base.model(inputs_embeds=seq, attention_mask=mask, position_ids=pos,
                                      use_cache=False, output_hidden_states=True)
                h = out.hidden_states[self.depth]
        return h[:, mem_start:mem_start + self.K, :]   # (1, K, d) features at the real mem positions

    def encode(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, conditional: bool = True) -> dict:
        """Compress ``[ctx ; query]`` into M via the FROZEN base's own features (grad flows to mem + proj only).
        chunk_size>0 uses chunked memory: legacy S*K output unless ``state_cap`` bounds recurrent state."""
        if self.chunk_size > 0 and int(ctx_ids.shape[1]) > self.chunk_size:
            return self.encode_chunked(ctx_ids, query_ids, self.chunk_size, conditional)
        m_raw = self._encode_raw(ctx_ids, query_ids, conditional)
        m = self._project(m_raw)                                # readout (+VAE) + manifold alignment
        return {"memory": m, "memory_raw": m_raw,
                "mode": "conditioned" if conditional else "agnostic"}

    def encode_chunked(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, chunk_size: int,
                       conditional: bool = True) -> dict:
        """Chunked memory with either legacy length-adaptive or bounded state.

        ``state_cap == 0`` preserves the original behavior: carry and return all
        per-chunk summaries (S*K slots). A positive cap keeps only the newest
        ``state_cap`` recurrent slots after each update and returns that bounded
        state, so memory presented to the reader cannot grow with context length.
        """
        Lc = int(ctx_ids.shape[1])
        if chunk_size <= 0 or Lc <= chunk_size:
            return self.encode(ctx_ids, query_ids, conditional)
        if self.recur:                                          # RECURRENT (AutoCompressor-style): carry prior summaries
            mems = []                                           # projected per-chunk summaries (embedding space)
            prefix = None
            n_chunks = 0
            for i in range(0, Lc, chunk_size):
                m_raw_i = self._encode_raw(ctx_ids[:, i:i + chunk_size], query_ids, conditional, prefix=prefix)
                m_i = self._project(m_raw_i)                    # (1,K,d) on the embedding manifold -> usable as soft prefix
                n_chunks += 1
                if not self.state_cap:
                    mems.append(m_i)
                # DETACH bounds backprop while retaining the forward recurrent
                # signal. The legacy mode carries all S*K summaries; capped mode
                # is a true fixed-state update rather than final-output slicing.
                parts = mems if not self.state_cap else (
                    [prefix, m_i] if prefix is not None else [m_i]
                )
                state = torch.cat(parts, dim=1)
                if self.state_cap:
                    state = state[:, -self.state_cap:, :]
                prefix = state.detach()
            m = torch.cat(mems, dim=1) if not self.state_cap else state
            return {"memory": m, "memory_raw": m, "n_chunks": n_chunks,
                    "state_cap": self.state_cap,
                    "mode": "conditioned" if conditional else "agnostic"}
        raws = [self._encode_raw(ctx_ids[:, i:i + chunk_size], query_ids, conditional)
                for i in range(0, Lc, chunk_size)]
        m_raw = torch.cat(raws, dim=1)                          # (1, S*K, d)
        if self.xchunk > 0 and len(raws) > 1:                   # CROSS-CHUNK: let the per-chunk summaries see each other
            m_raw = self._refine_xchunk(m_raw)
        m = self._project(m_raw)
        return {"memory": m, "memory_raw": m_raw, "n_chunks": len(raws),
                "mode": "conditioned" if conditional else "agnostic"}

    def _refine_xchunk(self, m_raw: torch.Tensor) -> torch.Tensor:
        """Cross-chunk refinement. The per-chunk memories were each encoded in ISOLATION (no chunk sees another), so
        a root cause that spans chunks is invisible at compression time. Here we put ALL S*K chunk-summaries into ONE
        sequence (relative positions = chunk order) and run them through the NEXT ``xchunk`` FROZEN base layers, so each
        summary integrates the others via the base's own pretrained dynamics. Cost is tiny (S*K ~ a few hundred tokens)
        and there are NO new parameters -> the adapter format is unchanged. Read-LoRA stays OFF (encode phase)."""
        layers = self.base.model.layers
        nref = min(int(self.xchunk), len(layers) - 1)
        if nref <= 0:
            return m_raw
        L = m_raw.shape[1]
        pos = torch.arange(L, device=m_raw.device).unsqueeze(0)
        with self._enc_phase():
            saved = self.base.model.layers
            # run the concatenated chunk-summaries through the FIRST nref frozen layers so they attend to each other.
            # (output_hidden_states is unreliable under a sliced layer-list, but the forward DOES run all nref layers,
            # so last_hidden_state = the mixed output after nref layers -- which _project then reads.)
            self.base.model.layers = saved[:nref]
            try:
                out = self.base.model(inputs_embeds=m_raw, position_ids=pos, use_cache=False)
                h = out.last_hidden_state
            finally:
                self.base.model.layers = saved
        return h

    def forward(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, conditional: bool = True) -> dict:
        return self.encode(ctx_ids, query_ids, conditional)

    def encode_ar(self, ctx_ids, query_ids):
        """Autoregressive M: produce the K memory vectors one at a time, each conditioned on the previously
        produced ones (fed back as input embeddings). KV-cache for the frozen [ctx;query] prefill; produced M
        are DETACHED when fed back (truncated BPTT) so memory ~= one forward. w = the first mem slot (write-query).
        Returns M [1, K, d]. (grad -> proj + mem(w) each step; not through the produced-M chain.)"""
        embed = self.base.get_input_embeddings()
        ce, qe = embed(ctx_ids.to(self.mem.device)), embed(query_ids.to(self.mem.device))
        w = self.mem[:1].unsqueeze(0).to(ce.dtype)               # (1,1,d) learnable write-query
        with self._enc_phase():
            with torch.no_grad():                                # frozen prefill over [ctx; query]
                pre = self.base.model(inputs_embeds=torch.cat([ce, qe], 1), use_cache=True)
                past = pre.past_key_values
            produced = []
            for t in range(self.K):
                inp = w if not produced else torch.cat([produced[-1].detach().to(w.dtype), w], 1)
                out = self.base.model(inputs_embeds=inp, past_key_values=past, use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                h = out.hidden_states[self.depth][:, -1:, :]     # hidden at the write-query position
                produced.append(self._project(h))                # (1,1,d)
        return torch.cat(produced, 1)                            # (1,K,d)


    def encode_batch(self, ctx_list, query_list):
        """Batched conditional encode (grad flows to mem+proj). Left-pad [ctx;query] to common length, append
        the K mem tokens at the tail, run the frozen base with a 2D causal+pad mask (validated to match per-item
        encode to bf16 noise on Qwen3.5 linear-attn). Returns M of shape [B, K, d]. No position_ids (model derives
        them from the mask; validated)."""
        embed = self.base.get_input_embeddings()
        dev = self.mem.device
        pieces, lens = [], []
        for c, q in zip(ctx_list, query_list):
            e = torch.cat([embed(c.to(dev)), embed(q.to(dev))], dim=1)
            pieces.append(e); lens.append(int(e.shape[1]))
        B = len(pieces); T = max(lens); d = self.d
        _eos_c = getattr(self.base.config, "eos_token_id", 0) or 0
        if isinstance(_eos_c, (list, tuple)): _eos_c = _eos_c[0] if _eos_c else 0
        pad_id = int(_eos_c)
        pad_emb = embed(torch.tensor([[pad_id]], device=dev))
        mem = self.mem.unsqueeze(0).to(pieces[0].dtype)
        rows = []
        attn = torch.zeros(B, T + self.K, dtype=torch.long, device=dev)
        for i, e in enumerate(pieces):
            n = T - e.shape[1]
            parts = ([pad_emb.to(e.dtype).expand(1, n, d)] if n > 0 else []) + [e, mem.to(e.dtype)]
            rows.append(torch.cat(parts, dim=1))
            attn[i, n:] = 1
        emb_batch = torch.cat(rows, dim=0)
        with self._enc_phase():
            out = self.base.model(inputs_embeds=emb_batch, attention_mask=attn,
                                  use_cache=False, output_hidden_states=True)
        m_raw = out.hidden_states[self.depth][:, -self.K:, :]
        m = self._project(m_raw)
        return m


    def trainable_parameters(self) -> list[nn.Parameter]:
        extra = [self.embed_scale] if isinstance(self.embed_scale, nn.Parameter) else []
        return [self.mem, *self.proj.parameters(), *extra]

    # ---- adapter-only checkpoint (the base is shared/frozen and NOT saved) ----
    def save_adapters(self, path: str) -> None:
        import torch as _t
        _t.save({"mem": self.mem.detach().cpu(), "proj": self.proj.state_dict(),
                 "embed_scale": self.embed_scale.detach().cpu(),
                 "config": {"K": self.K, "depth": self.depth, "normalize": self.normalize}}, path)

    def load_adapters(self, path: str, map_location=None) -> "SelfCompressor":
        import torch as _t
        sd = _t.load(path, map_location=map_location or self.mem.device, weights_only=False)
        with _t.no_grad():
            self.mem.copy_(sd["mem"].to(self.mem))
            if "embed_scale" in sd:
                self.embed_scale.copy_(sd["embed_scale"].to(self.embed_scale))
        self.proj.load_state_dict(sd["proj"])
        return self
