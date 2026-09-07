"""v1.7 baseline compressors as "methods".

A method only has to provide, for an eval item, the PREFIX it puts before the query (and, for Gist, the
masked context length). Trainable methods (Cartridge, Gist) train their small parameter first on the
training corpus. Everything else (generation, MC, scoring) is shared in ``runtime.py``.

Baselines here are VANILLA (no gate) and run in their ORIGINAL form: frozen base is OUR method's principle,
NOT the baselines'. We do not impose frozen-base on them.

TARGET (strict-original):
  - Cartridge (Eyuboglu et al. 2025): a TRAINED KV-CACHE per corpus, learned by self-study +
    context-distillation. The base is frozen because that is Cartridges' OWN design (not us imposing it).
  - Gist (Mu et al. 2023): K gist tokens + gist attention mask + per-training-set LoRA FINE-TUNING of the base
    (gist-masked CE; base adapted via LoRA, NOT frozen). Both the gist token embeddings and the LoRA adapter
    are trained. (LoRA over full FT for feasibility across the grid; disclosed.)

STATUS: the code below is an INTERIM simplified version (Cartridge = K soft prefix tokens; Gist = frozen base
training only the gist embeddings). It is being reimplemented to the strict-original forms above
(Cartridge KV-cache; Gist base fine-tune). See training-design-2026-06-10.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .runtime import Runtime


@dataclass
class TrainCfg:
    steps: int = 1800
    lr: float = 3e-4
    distill_tokens: int = 16
    distill_topk: int = 64
    distill_temp: float = 1.0
    # OURS (svc) only: encoder/decoder depth, autoencoder reconstruction (L_uncond) + min-dev weights, fp32.
    enc_layers: int = 4
    n_dec_layers: int = 2
    lam_rec: float = 1.0
    lam_dev: float = 0.05
    train_fp32: bool = False      # default bf16 (matches deployment + the bf16 baselines); opt into fp32 explicitly
    m_norm_match: str = "off"     # A1 (v1.7.1.1): off/hard/learn -- L2-normalize M to the embedding scale (fix OOD)
    m_manifold_temp: float = 0.0  # A2 (v1.7.1.2): >0 => project M onto the token-embedding convex hull (in-dist)
    base_lora_rank: int = 0       # A3 (v1.7.1.3): >0 => add a LoRA on the frozen base that learns to read M
    inject: str = "prefix"        # A4 (v1.7.1.4): prefix (input-embed) | kv (per-layer KV-cache)
    recon_m_only: bool = False    # C1 (v1.7.3.1): decoder reconstructs ctx from M ONLY (block ctx->ctx) = true lossless
    chunk_size: int = 0           # v1.7.4: >0 => length-adaptive chunked memory (fixed ratio chunk/K, budget scales with len)
    chunk_keep: int = 0           # v1.7.4: >0 => query-relevance routing, keep top-`keep` chunk-memories (soft retrieval)
    grad_accum: int = 8           # v1.7.5 STABILITY: effective batch = grad_accum micro-steps (batch-1 -> noisy; >1 smooths)
    warmup_frac: float = 0.05     # v1.7.5: LR warmup fraction of optimizer steps
    lr_cosine: bool = True        # v1.7.5: cosine LR decay after warmup (vs constant)
    patience: int = 0             # v1.7.6: >0 => early-stop after N val-loss checks w/o improvement (train-until-converge)
    # OURS task loss: gold-answer CE (lam_task) is the primary per-query signal (forces M to carry the answer,
    # like Gist); teacher context-distillation (lam_distill) is an optional auxiliary, off by default because the
    # full-ctx teacher is weak on noisy chunks and the teacher-forced response is too easy a target.
    lam_task: float = 1.0
    lam_distill: float = 0.0
    enc_init: str = "copy"        # ablation: "copy" (base first-N blocks) vs "random" (random-init layers)
    gcm_agnostic: bool = False    # ablation: train+eval the query-AGNOSTIC memory M0 (vs conditioned Mq)
    lam_adv: float = 0.0          # adversarial losslessness: make [M;q] base-state indistinguishable from [ctx;q]
    adv_layer: int = 18           # base layer whose hidden the discriminator reads
    lam_contrast: float = 0.0     # v1.8.0: InfoNCE semantic alignment of [M;q] vs [ctx;q] hidden (alt. to adv)
    lam_align: float = 0.0        # v1.8.0: pure positive alignment (NO negatives) — cosine pull only (ablation)
    align_layer: int = 18         # shared base layer for adv/contrast alignment (defaults to adv_layer)
    contrast_temp: float = 0.2    # InfoNCE temperature
    contrast_bank: int = 256      # MoCo-style FIFO negative-bank size (for batch-1 training)


@runtime_checkable
class Method(Protocol):
    name: str
    trainable: bool

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None: ...

    def prefix(self, rt: Runtime, item: Any) -> tuple[torch.Tensor | None, int]:
        """Return (prefix_embeds | None, ctx_len). ctx_len>0 triggers the gist mask (Gist only)."""
        ...


class NoCtx:
    """Floor: bare base, query only."""
    name = "no_ctx"
    trainable = False

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None:
        return None

    def prefix(self, rt: Runtime, item: Any) -> tuple[torch.Tensor | None, int]:
        return None, 0


class FullCtx:
    """Ceiling: full context prepended; query attends to it (no mask)."""
    name = "full_ctx"
    trainable = False

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None:
        return None

    def prefix(self, rt: Runtime, item: Any) -> tuple[torch.Tensor | None, int]:
        return rt.embed(rt.ctx_ids(item)), 0


class Cartridge:
    """Faithful Cartridge (Eyuboglu et al. 2025): a per-corpus TRAINABLE KV-CACHE ``Z`` of shape
    (L, n_kv, p, head_dim) -- key+value vectors at EVERY layer (prefix-tuning), base frozen. Trained by
    self-study + context-distillation; injected as past_key_values at inference. Z is prefill-initialized from
    the first corpus chunk's KV (the paper finds init matters). NOTE: needs standard attention KV (Qwen3-8B);
    on hybrid linear-attn bases (Qwen3.5) some layers lack standard KV."""
    name = "cartridge"
    trainable = True

    def __init__(self) -> None:
        self.zk: nn.Parameter | None = None
        self.zv: nn.Parameter | None = None

    def kv_prefix(self, rt: Runtime, item: Any):
        assert self.zk is not None, "Cartridge.train must run before kv_prefix"
        return self.zk, self.zv   # per-corpus, item-independent

    @staticmethod
    def _layer_kv(pkv, layer: int):
        if hasattr(pkv, "key_cache"):
            return pkv.key_cache[layer], pkv.value_cache[layer]
        if hasattr(pkv, "layers"):
            return pkv.layers[layer].keys, pkv.layers[layer].values
        return pkv[layer][0], pkv[layer][1]

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None:
        model, embed, tok, dev = rt.model, rt.embed, rt.tok, rt.dev
        L = model.config.num_hidden_layers
        p, T, topk = rt.K, cfg.distill_tokens, cfg.distill_topk
        # prefill-init Z from the first item's context (first p KV positions per layer)
        with torch.no_grad():
            pkv = model(inputs_embeds=embed(rt.ctx_ids(train_items[0])), use_cache=True).past_key_values
            zk0, zv0 = [], []
            for layer in range(L):
                k, v = self._layer_kv(pkv, layer)
                k, v = k[0], v[0]                                  # (n_kv, s, hd)
                s = k.shape[1]
                if s >= p:
                    k, v = k[:, :p], v[:, :p]
                else:
                    k = torch.cat([k, k[:, -1:].expand(-1, p - s, -1)], 1)
                    v = torch.cat([v, v[:, -1:].expand(-1, p - s, -1)], 1)
                zk0.append(k.clone()); zv0.append(v.clone())
        self.zk = nn.Parameter(torch.stack(zk0).to(dev, rt.bdt))   # (L, n_kv, p, hd)
        self.zv = nn.Parameter(torch.stack(zv0).to(dev, rt.bdt))
        opt = torch.optim.AdamW([self.zk, self.zv], lr=cfg.lr)

        study: list[dict] = []
        with torch.no_grad():
            for it in train_items:
                qid, cid = rt.query_ids(it), rt.ctx_ids(it)
                ce, qe = embed(cid), embed(qid)
                seq, R = torch.cat([ce, qe], dim=1), []
                for _ in range(T):
                    nxt = int(model(inputs_embeds=seq, use_cache=False).logits[:, -1].argmax(-1))
                    if nxt == getattr(tok, "eos_token_id", -1):
                        break
                    R.append(nxt)
                    seq = torch.cat([seq, embed(torch.tensor([[nxt]], device=dev))], dim=1)
                if not R:
                    R = tok(" " + str(it.gold).strip(), add_special_tokens=False).input_ids[:T]
                if not R:
                    continue
                Rt = torch.tensor([R], device=dev)
                Pt = ce.shape[1] + qe.shape[1]
                tlg = model(inputs_embeds=torch.cat([ce, qe, embed(Rt)], 1),
                            use_cache=False).logits[0, Pt - 1:Pt - 1 + len(R)].float()
                tkl, tki = tlg.topk(topk, dim=-1)
                study.append({"qid": qid, "R": Rt, "tk_idx": tki, "tk_logit": tkl})
        logger.info(f"[cartridge] faithful KV-cache Z: L={L} p={p}; self-study targets={len(study)}")

        temp = cfg.distill_temp
        step = 0
        while step < cfg.steps and study:
            for s in study:
                qe = embed(s["qid"])
                seq = torch.cat([qe, embed(s["R"])], dim=1)        # [q ; R] -- the context lives in Z, not the prompt
                Ps = qe.shape[1]
                slg = rt.kv_logits(self.zk, self.zv, seq)[0, Ps - 1:Ps - 1 + s["R"].shape[1]].float()
                slp = F.log_softmax(slg / temp, dim=-1).gather(-1, s["tk_idx"])
                pt = F.softmax(s["tk_logit"] / temp, dim=-1)
                loss = (pt * (F.log_softmax(s["tk_logit"] / temp, dim=-1) - slp)).sum(-1).mean() * (temp * temp)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.zk, self.zv], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % max(1, cfg.steps // 10) == 0:
                    logger.info(f"[cartridge] step {step}/{cfg.steps} distillKL={float(loss):.4f}")
                if step >= cfg.steps:
                    break
        self.zk.requires_grad_(False)
        self.zv.requires_grad_(False)


class Gist:
    """Faithful Gist (Mu et al. 2023): instruction-finetune the LM (via LoRA) WITH the gist attention mask, so the
    MODEL learns to compress the prompt into the gist-token activations (the original finetunes the model, not just a
    learned embedding). gist tokens (gist_p) + gist mask retained; base now LoRA-adapted."""
    name = "gist"
    trainable = True
    use_base_lora = True

    def __init__(self) -> None:
        self.gist_p: nn.Parameter | None = None

    def prefix(self, rt: Runtime, item: Any) -> tuple[torch.Tensor | None, int]:
        assert self.gist_p is not None, "Gist.train must run before prefix"
        ce = rt.embed(rt.ctx_ids(item))
        pre = torch.cat([ce, self.gist_p.unsqueeze(0)], dim=1)
        return pre, ce.shape[1]  # ctx_len = Lc -> gist mask blocks query/output from ctx

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None:
        model, embed, tok, dev = rt.model, rt.embed, rt.tok, rt.dev
        self.gist_p = nn.Parameter((torch.randn(rt.K, rt.d) * 0.02).to(dev, rt.bdt))
        from svc.lora import add_lora   # faithful: the LM itself learns gisting (LoRA) under the gist mask
        rank = max(int(getattr(cfg, "base_lora_rank", 0)), 16)
        lora = add_lora(model, rank, dtype=rt.bdt)
        # Long QuALITY prompts plus the dense gist mask exceed one H100 during backward without
        # activation rematerialization. This changes memory use only, not the Gisting objective.
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
        logger.info(f"[gist] faithful: base LoRA rank={rank} (+{len(lora)} tensors) trained under the gist mask")
        opt = torch.optim.AdamW([self.gist_p] + lora, lr=cfg.lr)
        step = 0
        while step < cfg.steps:
            for it in train_items:
                tgt = tok(" " + str(it.gold).strip(), add_special_tokens=False).input_ids
                if not tgt:
                    continue
                tgt = torch.tensor(tgt, device=dev).unsqueeze(0)
                qid = rt.query_ids(it)
                ce = embed(rt.ctx_ids(it))
                mem = torch.cat([ce, self.gist_p.unsqueeze(0)], dim=1)
                ctx_len = ce.shape[1]
                qe, te = embed(qid), embed(tgt)
                seq = torch.cat([mem, qe, te], dim=1)
                am = rt.gist_mask(seq.shape[1], mem.shape[1], ctx_len)
                keep = int(tgt.shape[1]) + 1
                import contextlib, os
                offload = os.environ.get("GCM_GIST_OFFLOAD", "0") == "1"
                cm = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else contextlib.nullcontext()
                with cm:
                    out = model(
                        inputs_embeds=seq,
                        attention_mask=am,
                        use_cache=False,
                        logits_to_keep=keep,
                    )
                lg = out.logits[:, :tgt.shape[1]].reshape(-1, out.logits.shape[-1])
                loss = F.cross_entropy(lg.float(), tgt.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_([self.gist_p], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % max(1, cfg.steps // 10) == 0:
                    logger.info(f"[gist] step {step}/{cfg.steps} ce={float(loss):.4f}")
                if step >= cfg.steps:
                    break
        self.gist_p.requires_grad_(False)


class TruncateK:
    """Trivial baseline (NO training): keep the FIRST K context-token embeddings. Necessity test —
    does the trained encoder beat simply truncating the context to K tokens?"""
    name = "trunc"
    trainable = False

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None:
        return None

    def prefix(self, rt: Runtime, item: Any) -> tuple[torch.Tensor | None, int]:
        return rt.embed(rt.ctx_ids(item))[:, : rt.K], 0


class MeanPoolK:
    """Trivial baseline (NO training): K segment-means of the context embeddings (uniform pooling)."""
    name = "meanpool"
    trainable = False

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None:
        return None

    def prefix(self, rt: Runtime, item: Any) -> tuple[torch.Tensor | None, int]:
        ce = rt.embed(rt.ctx_ids(item))
        L, K = ce.shape[1], rt.K
        if L <= K:
            return ce, 0
        bnd = torch.linspace(0, L, K + 1).round().long()
        segs = [ce[:, bnd[i]:bnd[i + 1]].mean(1, keepdim=True) for i in range(K) if bnd[i + 1] > bnd[i]]
        return torch.cat(segs, 1), 0


class RandomK:
    """Floor baseline (NO training): K random embeddings — should sit at ~no_ctx (sanity check that the
    memory SLOT positions alone carry nothing)."""
    name = "randk"
    trainable = False

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None:
        return None

    def prefix(self, rt: Runtime, item: Any) -> tuple[torch.Tensor | None, int]:
        return torch.randn(1, rt.K, rt.d, device=rt.dev, dtype=rt.bdt) * 0.02, 0


class RetrievalTopK:
    """Hard-retrieval baseline (NO training): split ctx into CHUNK-token chunks, score each by query lexical
    overlap (distinct shared tokens = a BM25 proxy), keep the top-KEEP chunks, and feed their RAW embeddings
    (no compression). The natural RIVAL to chunked routing (v1.7.4): does a LEARNED compressor beat simply
    retrieving the relevant raw chunk? Budget = KEEP*CHUNK raw tokens (overridable via MEM_RETR_{CHUNK,KEEP})."""
    name = "retrieval"
    trainable = False

    def __init__(self) -> None:
        import os
        self.chunk = int(os.environ.get("MEM_RETR_CHUNK", "64"))
        self.keep = int(os.environ.get("MEM_RETR_KEEP", "4"))

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None:
        return None

    def mem_len(self, n_ctx: int) -> int:
        return min(int(n_ctx), self.chunk * self.keep) + 0  # raw tokens fed (for the cost column)

    def prefix(self, rt: Runtime, item: Any) -> tuple[torch.Tensor | None, int]:
        cids, qids = rt.ctx_ids(item), rt.query_ids(item)
        L = cids.shape[1]
        if L <= self.chunk * self.keep:                 # short ctx: nothing to retrieve, feed all
            return rt.embed(cids), 0
        qset = set(qids[0].tolist())
        spans = [(i, cids[:, i:i + self.chunk]) for i in range(0, L, self.chunk)]
        # score = # distinct query tokens present in the chunk (lexical overlap; BM25-style term presence)
        ranked = sorted(spans, key=lambda s: len(qset & set(s[1][0].tolist())), reverse=True)[: self.keep]
        ranked = sorted(ranked, key=lambda s: s[0])     # restore reading order
        sel = torch.cat([c for _, c in ranked], dim=1)  # (1, KEEP*CHUNK) raw token ids
        return rt.embed(sel), 0


class TokenPrune:
    """Hard-prompt token-pruning baseline (NO training; LLMLingua / Selective-Context family, query-AGNOSTIC):
    keep the K context tokens with the highest SELF-INFORMATION (surprisal -log p under the frozen base), in
    reading order, and feed their raw embeddings. The query-agnostic counterpart to `retrieval` (query-aware
    selection) — the 'prune the low-information tokens' point reviewers expect. One base forward over the ctx."""
    name = "tokprune"
    trainable = False

    def train(self, rt: Runtime, train_items: list, cfg: TrainCfg) -> None:
        return None

    def prefix(self, rt: Runtime, item: Any) -> tuple[torch.Tensor | None, int]:
        cids = rt.ctx_ids(item)
        L, K = cids.shape[1], rt.K
        if L <= K:
            return rt.embed(cids), 0
        with torch.no_grad():
            lg = rt.model(inputs_embeds=rt.embed(cids), use_cache=False).logits[0]   # (L, V)
            logp = lg[:-1].float().log_softmax(-1)                                   # (L-1, V) p(tok_t | <t)
            surp = -logp.gather(-1, cids[0, 1:].unsqueeze(-1)).squeeze(-1)           # (L-1,) surprisal
            surp = torch.cat([surp.new_zeros(1), surp])                             # token 0 => 0
        keep = surp.topk(K).indices.sort().values                                    # top-K informative, in order
        return rt.embed(cids[:, keep]), 0


REGISTRY: dict[str, type] = {
    "no_ctx": NoCtx,
    "full_ctx": FullCtx,
    "cartridge": Cartridge,
    "gist": Gist,
    "trunc": TruncateK,
    "meanpool": MeanPoolK,
    "randk": RandomK,
    "retrieval": RetrievalTopK,
    "tokprune": TokenPrune,
}

try:  # 2025+ context-compression baselines (ICAE/AOC/Beacon/X500/ComprExIT/LCC)
    from .baselines2025 import register as _register_2025
    _register_2025(REGISTRY)
except Exception as _e:  # noqa: BLE001
    import logging
    logging.getLogger(__name__).warning(f"2025 baselines unavailable: {_e!r}")
