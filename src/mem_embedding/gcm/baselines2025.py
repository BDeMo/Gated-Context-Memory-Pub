"""2025+ CONTEXT-COMPRESSION baselines — FAITHFUL re-implementations of the original architectures (verified
against the papers; see faithfulness audit in results-v1.7.3/baselines-faithfulness.md). Only the TRAINING
hyper-parameters (steps / lr / data amount) are set to our matched/fair budget; the MECHANISM matches the
original. All compress the context into K units the FROZEN base reads (budget-comparable to GCM, fair amortized
efficiency) — NOT token pruning/selection.

  - ICAE  (Ge et al., ICLR'24, 2307.06945): encoder = base + LoRA(q,v) over [ctx ; K memory tokens]; the memory
            tokens' OUTPUT HIDDEN STATES are the slots; FROZEN base is the decoder; a learnable [AE] token marks
            autoencoding. Trained AE (reconstruct ctx) + task (answer).  -> soft prefix (embeddings).
  - X500  (500xCompressor, 2408.03094): identical to ICAE but the decoder conditions on the compressed tokens'
            KV VALUES (per layer) instead of output embeddings; [BOS]-triggered.  -> KV cache.
  - AOC   (2501.06730): ICAE with an ATTENTION-ONLY encoder (MLP sublayers removed), trained fully.  -> embeddings.
  - Beacon(Activation Beacon, 2401.03462): interleave one beacon token after every alpha-token unit; compress
            into the beacons' KV.  -> KV cache (single-pass faithful core; the paper's cross-chunk KV accumulation
            is simplified to one pass at our <=1024-tok budget).
  - ComprExIT (2602.03784): LLM-as-encoder + explicit info transmission into anchor tokens (faithful core).
  - LCC   (Latent Context Compilation, 2602.21221): TEST-TIME per-context buffer optimized by reconstruction +
            a context-agnostic regularizer (no global training; faithful core).
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


def _inner(model):
    return model.model if hasattr(model, "model") else model   # Qwen3ForCausalLM -> Qwen3Model


def _cache_to_zkv(pkv, layers: int, K: int):
    """Take the LAST K positions' (key,value) per layer from a past_key_values -> stacked (L,n_kv,K,hd)."""
    def lk(layer):
        if hasattr(pkv, "key_cache"):
            return pkv.key_cache[layer], pkv.value_cache[layer]
        if hasattr(pkv, "layers"):
            return pkv.layers[layer].keys, pkv.layers[layer].values
        return pkv[layer][0], pkv[layer][1]
    zk, zv = [], []
    for layer in range(layers):
        k, v = lk(layer)
        zk.append(k[0, :, -K:, :]); zv.append(v[0, :, -K:, :])
    return torch.stack(zk), torch.stack(zv)


# ============================================================================= ICAE (faithful)
class ICAE:
    name = "icae"; trainable = True; use_base_lora = False

    def __init__(self) -> None:
        self.mem: nn.Parameter | None = None   # learnable memory-token embeddings e_m (K, d)
        self.ae: nn.Parameter | None = None    # learnable [AE] token embedding
        self.lora: list = []
        self._kv = False                       # ICAE uses output embeddings; X500 overrides to KV

    def _init_extra(self, rt: Any) -> None:    # ComprExIT overrides to add cross-layer transmission params
        return None

    def _extra_params(self) -> list:
        return []

    def _encode(self, rt: Any, cid: torch.Tensor):
        """LoRA-ON pass of base over [ctx ; K mem tokens]; return memory slots (hidden) or KV of the K tokens."""
        from svc.lora import set_lora_enabled
        set_lora_enabled(rt.model, True)
        seq = torch.cat([rt.embed(cid), self.mem.unsqueeze(0).to(rt.bdt)], dim=1)
        out = _inner(rt.model)(inputs_embeds=seq, use_cache=self._kv)
        set_lora_enabled(rt.model, False)
        if self._kv:
            return _cache_to_zkv(out.past_key_values, rt.model.config.num_hidden_layers, self.mem.shape[0])
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return h[:, -self.mem.shape[0]:, :]    # the K memory tokens' output hidden states = slots

    def prefix(self, rt: Any, item: Any):
        with torch.no_grad():
            return self._encode(rt, rt.ctx_ids(item)).to(rt.bdt), 0

    def train(self, rt: Any, train_items: list, cfg: Any) -> None:
        from svc.lora import add_lora, lora_disabled
        model, embed, tok, dev = rt.model, rt.embed, rt.tok, rt.dev
        d = rt.d
        self.mem = nn.Parameter((torch.randn(rt.K, d) * 0.02).to(dev, rt.bdt))
        self.ae = nn.Parameter((torch.randn(1, d) * 0.02).to(dev, rt.bdt))
        rank = max(int(getattr(cfg, "base_lora_rank", 0)), 64)       # OUR matched-budget LoRA (NOT paper-aligned; the architecture is what we align)
        self.lora = add_lora(model, rank, targets=("q_proj", "v_proj"), dtype=rt.bdt)
        self._init_extra(rt)
        from svc.lora import set_lora_enabled
        set_lora_enabled(model, False)
        params = [self.mem, self.ae] + self.lora + self._extra_params()
        opt = torch.optim.AdamW(params, lr=cfg.lr)
        lam_rec = float(getattr(cfg, "lam_rec", 1.0))
        study = []
        for it in train_items:
            g = tok(" " + str(it.gold).strip(), add_special_tokens=False).input_ids[: cfg.distill_tokens]
            if g:
                study.append({"cid": rt.ctx_ids(it), "qid": rt.query_ids(it), "gold": torch.tensor([g], device=dev)})
        logger.info(f"[{self.name}] base+LoRA(q,v) rank={rank} encoder + frozen decoder; K={rt.K}; {len(study)} targets")
        step = 0
        while step < cfg.steps and study:
            for s in study:
                slots = self._encode(rt, s["cid"])                   # (1,K,d), carries LoRA grad
                with lora_disabled(model):                           # decoder = FROZEN base (LoRA off)
                    # AE: [slots ; [AE] ; ctx] -> reconstruct ctx
                    cap = min(
                        int(s["cid"].shape[1]),
                        int(os.environ.get("MEM_B25_RECON_MAX", "512")),
                    )
                    rcid = s["cid"][:, :cap]
                    ae_seq = torch.cat([slots, self.ae.unsqueeze(0).to(rt.bdt), embed(rcid)], dim=1)
                    Pae = slots.shape[1] + 1
                    ae_lg = model(
                        inputs_embeds=ae_seq,
                        use_cache=False,
                        logits_to_keep=cap + 1,
                    ).logits[0, :cap].float()
                    l_ae = F.cross_entropy(ae_lg, rcid[0])
                    # task: [slots ; query ; gold] -> predict gold
                    t_seq = torch.cat([slots, embed(s["qid"]), embed(s["gold"])], dim=1)
                    keep = int(s["gold"].shape[1]) + 1
                    t_lg = model(
                        inputs_embeds=t_seq,
                        use_cache=False,
                        logits_to_keep=keep,
                    ).logits[0, :s["gold"].shape[1]].float()
                    l_task = F.cross_entropy(t_lg, s["gold"][0])
                loss = l_task + lam_rec * l_ae
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
                step += 1
                if step % max(1, cfg.steps // 10) == 0:
                    logger.info(f"[{self.name}] step {step}/{cfg.steps} task={float(l_task):.3f} ae={float(l_ae):.3f}")
                if step >= cfg.steps:
                    break
        for p in params:
            p.requires_grad_(False)


# ============================================================================= 500xCompressor (faithful, KV)
class X500(ICAE):
    """500xCompressor: ICAE but the decoder conditions on the compressed tokens' KV (per layer), [BOS]-triggered."""
    name = "x500"

    def __init__(self) -> None:
        super().__init__()
        self._kv = True

    def kv_prefix(self, rt: Any, item: Any):
        with torch.no_grad():
            zk, zv = self._encode(rt, rt.ctx_ids(item))
        return zk.to(rt.bdt), zv.to(rt.bdt)

    def prefix(self, rt: Any, item: Any):   # not used (KV path), but keep for safety
        raise RuntimeError("X500 uses kv_prefix")

    def train(self, rt: Any, train_items: list, cfg: Any) -> None:
        from svc.lora import add_lora, set_lora_enabled
        model, embed, tok, dev = rt.model, rt.embed, rt.tok, rt.dev
        self.mem = nn.Parameter((torch.randn(rt.K, rt.d) * 0.02).to(dev, rt.bdt))
        self.ae = nn.Parameter(torch.zeros(1, rt.d).to(dev, rt.bdt))      # unused (KV uses [BOS]=query)
        rank = max(int(getattr(cfg, "base_lora_rank", 0)), 64)           # OUR matched-budget LoRA (KV-conditioning is the faithful 500x design, not the LoRA size)
        self.lora = add_lora(model, rank, targets=("q_proj", "v_proj"), dtype=rt.bdt)
        set_lora_enabled(model, False)
        params = [self.mem] + self.lora
        opt = torch.optim.AdamW(params, lr=cfg.lr)
        study = []
        for it in train_items:
            g = tok(" " + str(it.gold).strip(), add_special_tokens=False).input_ids[: cfg.distill_tokens]
            if g:
                study.append({"cid": rt.ctx_ids(it), "qid": rt.query_ids(it), "gold": torch.tensor([g], device=dev)})
        logger.info(f"[x500] LoRA(q,v) rank={rank} encoder -> K KV pairs; frozen decoder; {len(study)} targets")
        step = 0
        while step < cfg.steps and study:
            for s in study:
                zk, zv = self._encode(rt, s["cid"])                       # (L,n_kv,K,hd), LoRA grad
                seq = torch.cat([embed(s["qid"]), embed(s["gold"])], dim=1)
                keep = int(s["gold"].shape[1]) + 1
                glg = rt.kv_logits(zk, zv, seq, logits_to_keep=keep)[0, :s["gold"].shape[1]].float()
                loss = F.cross_entropy(glg, s["gold"][0])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
                step += 1
                if step % max(1, cfg.steps // 10) == 0:
                    logger.info(f"[x500] step {step}/{cfg.steps} loss={float(loss):.3f}")
                if step >= cfg.steps:
                    break
        for p in params:
            p.requires_grad_(False)


# ============================================================================= AOC (faithful: MLP-free encoder)
class _ZeroMLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class AOC:
    """AOC (2501.06730): ICAE-style but the encoder is a SEPARATE copy of the base with the MLP sublayers REMOVED
    (attention-only, ~67% fewer encoder params), trained fully; frozen base decoder; AE + task."""
    name = "aoc"; trainable = True; use_base_lora = False

    def __init__(self) -> None:
        self.svc: Any = None

    def prefix(self, rt: Any, item: Any):
        with torch.no_grad():
            m = self.svc.encode(rt.ctx_ids(item), rt.query_ids(item), condition_on_query=False)["memory"]
        return m.to(rt.bdt), 0

    def train(self, rt: Any, train_items: list, cfg: Any) -> None:
        from svc.compressor import SelfVerifyingCompressor
        model, embed, tok, dev = rt.model, rt.embed, rt.tok, rt.dev
        full = int(model.config.num_hidden_layers)   # AOC faithful: the ENCODER IS the base model (all layers, attention-only), trained fully
        self.svc = SelfVerifyingCompressor(model, n_layers=full, n_memory=rt.K, n_dec_layers=2)
        layers = self.svc.encoder.layers if hasattr(self.svc.encoder, "layers") else self.svc.encoder.model.layers
        removed = 0
        for layer in layers:                          # REMOVE MLP sublayers (attention-only encoder)
            if hasattr(layer, "mlp"):
                layer.mlp = _ZeroMLP(); removed += 1
        logger.info(f"[aoc] attention-only encoder: removed {removed} MLP sublayers")
        out_w = embed.weight
        params = [p for p in self.svc.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr)
        lam_rec = float(getattr(cfg, "lam_rec", 1.0))
        study = []
        for it in train_items:
            g = tok(" " + str(it.gold).strip(), add_special_tokens=False).input_ids[: cfg.distill_tokens]
            if g:
                study.append({"cid": rt.ctx_ids(it), "qid": rt.query_ids(it), "gold": torch.tensor([g], device=dev)})
        logger.info(f"[aoc] {len(study)} targets, K={rt.K}")
        step = 0
        while step < cfg.steps and study:
            for s in study:
                m = self.svc.encode(s["cid"], s["qid"], condition_on_query=False)["memory"].to(rt.bdt)
                seq = torch.cat([m, embed(s["qid"]), embed(s["gold"])], dim=1)
                keep = int(s["gold"].shape[1]) + 1
                glg = model(
                    inputs_embeds=seq,
                    use_cache=False,
                    logits_to_keep=keep,
                ).logits[0, :s["gold"].shape[1]].float()
                loss = F.cross_entropy(glg, s["gold"][0])
                if self.svc.decoder is not None and lam_rec > 0:
                    m0 = self.svc.encode(s["cid"], s["qid"], condition_on_query=False)["memory"]
                    loss = loss + lam_rec * self.svc.reconstruct(s["cid"], m0, out_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
                step += 1
                if step % max(1, cfg.steps // 10) == 0:
                    logger.info(f"[aoc] step {step}/{cfg.steps} loss={float(loss):.3f}")
                if step >= cfg.steps:
                    break
        for p in self.svc.parameters():
            p.requires_grad_(False)


# ============================================================================= Beacon (faithful core, beacon KV)
class Beacon:
    """Activation Beacon (2401.03462) core: interleave one beacon token after every alpha-token unit; the base
    (+LoRA) compresses each unit's info into the beacons' KV (keys/values). We inject the beacons' KV into the
    frozen decoder. Single-pass at our <=1024 budget (the paper's cross-chunk KV accumulation is simplified)."""
    name = "beacon"; trainable = True; use_base_lora = False

    def __init__(self) -> None:
        self.beacon: nn.Parameter | None = None
        self.lora: list = []
        self.alpha = 16                      # compression ratio: one beacon per 16 ctx tokens (OUR fixed budget; not paper-sampled)

    def _interleave(self, rt: Any, cid: torch.Tensor):
        embed = rt.embed
        L = cid.shape[1]
        parts, bpos = [], []
        for i in range(0, L, self.alpha):
            parts.append(embed(cid[:, i:i + self.alpha]))
            bpos.append(sum(p.shape[1] for p in parts))      # beacon goes right after this unit
            parts.append(self.beacon.unsqueeze(0).to(rt.bdt))
        return torch.cat(parts, dim=1), bpos

    def kv_prefix(self, rt: Any, item: Any):
        from svc.lora import set_lora_enabled
        with torch.no_grad():
            set_lora_enabled(rt.model, True)
            seq, bpos = self._interleave(rt, rt.ctx_ids(item))
            pkv = _inner(rt.model)(inputs_embeds=seq, use_cache=True).past_key_values
            set_lora_enabled(rt.model, False)
            Ln = rt.model.config.num_hidden_layers
            zk, zv = [], []
            for layer in range(Ln):
                if hasattr(pkv, "key_cache"):
                    k, v = pkv.key_cache[layer], pkv.value_cache[layer]
                elif hasattr(pkv, "layers"):
                    k, v = pkv.layers[layer].keys, pkv.layers[layer].values
                else:
                    k, v = pkv[layer][0], pkv[layer][1]
                zk.append(k[0, :, bpos, :]); zv.append(v[0, :, bpos, :])   # keep ONLY beacon positions
            return torch.stack(zk).to(rt.bdt), torch.stack(zv).to(rt.bdt)

    def train(self, rt: Any, train_items: list, cfg: Any) -> None:
        from svc.lora import add_lora, set_lora_enabled
        model, embed, tok, dev = rt.model, rt.embed, rt.tok, rt.dev
        self.beacon = nn.Parameter((torch.randn(1, rt.d) * 0.02).to(dev, rt.bdt))   # shared beacon embedding
        rank = max(int(getattr(cfg, "base_lora_rank", 0)), 64)
        self.lora = add_lora(model, rank, targets=("q_proj", "v_proj"), dtype=rt.bdt)
        set_lora_enabled(model, False)
        params = [self.beacon] + self.lora
        opt = torch.optim.AdamW(params, lr=cfg.lr)
        lam_rec = float(getattr(cfg, "lam_rec", 1.0))
        study = []
        for it in train_items:
            g = tok(" " + str(it.gold).strip(), add_special_tokens=False).input_ids[: cfg.distill_tokens]
            if g:
                study.append({"cid": rt.ctx_ids(it), "qid": rt.query_ids(it), "gold": torch.tensor([g], device=dev)})
        logger.info(f"[beacon] alpha={self.alpha} beacon-KV + ctx-recon, LoRA rank={rank}, {len(study)} targets")
        step = 0
        while step < cfg.steps and study:
            for s in study:
                set_lora_enabled(model, True)
                seq, bpos = self._interleave(rt, s["cid"])
                pkv = _inner(model)(inputs_embeds=seq, use_cache=True).past_key_values
                set_lora_enabled(model, False)
                Ln = model.config.num_hidden_layers
                zk = torch.stack([(pkv.key_cache[l] if hasattr(pkv, "key_cache") else
                                   (pkv.layers[l].keys if hasattr(pkv, "layers") else pkv[l][0]))[0, :, bpos, :]
                                  for l in range(Ln)])
                zv = torch.stack([(pkv.value_cache[l] if hasattr(pkv, "key_cache") else
                                   (pkv.layers[l].values if hasattr(pkv, "layers") else pkv[l][1]))[0, :, bpos, :]
                                  for l in range(Ln)])
                seqd = torch.cat([embed(s["qid"]), embed(s["gold"])], dim=1)
                keep = int(s["gold"].shape[1]) + 1
                glg = rt.kv_logits(zk, zv, seqd, logits_to_keep=keep)[0, :s["gold"].shape[1]].float()
                loss = F.cross_entropy(glg, s["gold"][0])
                if lam_rec > 0 and s["cid"].shape[1] > 1:              # paper: condensed beacons must reconstruct the unit (LM)
                    cap = min(
                        int(s["cid"].shape[1]),
                        int(os.environ.get("MEM_B25_RECON_MAX", "512")),
                    )
                    rcid = s["cid"][:, :cap]
                    ce = embed(rcid)
                    rlg = rt.kv_logits(zk, zv, ce, logits_to_keep=cap)[0, :cap - 1].float()
                    loss = loss + lam_rec * F.cross_entropy(rlg, rcid[0, 1:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
                step += 1
                if step % max(1, cfg.steps // 10) == 0:
                    logger.info(f"[beacon] step {step}/{cfg.steps} loss={float(loss):.3f}")
                if step >= cfg.steps:
                    break
        for p in params:
            p.requires_grad_(False)


# ============================================================================= ComprExIT (faithful core, cross-layer)
class ComprExIT(ICAE):
    """ComprExIT (2602.03784) faithful core: LLM-as-encoder (LoRA) over [ctx ; K anchor tokens], but the anchors'
    representation is an EXPLICIT cross-LAYER transmission — a learned softmax-weighted combination of the anchor
    positions' hidden states across ALL layers. This is mechanistically distinct from ICAE (last-layer slots only)
    and from AOC (a separate attention-only encoder): the 'transmission into anchors across layers' is the point."""
    name = "comprexit"

    def __init__(self) -> None:
        super().__init__()
        self.layer_w: nn.Parameter | None = None   # learned cross-layer transmission weights

    def _init_extra(self, rt: Any) -> None:
        nl = rt.model.config.num_hidden_layers + 1
        self.layer_w = nn.Parameter(torch.zeros(nl, device=rt.dev, dtype=rt.bdt))

    def _extra_params(self) -> list:
        return [self.layer_w]

    def _encode(self, rt: Any, cid: torch.Tensor):
        from svc.lora import set_lora_enabled
        set_lora_enabled(rt.model, True)
        seq = torch.cat([rt.embed(cid), self.mem.unsqueeze(0).to(rt.bdt)], dim=1)
        out = _inner(rt.model)(inputs_embeds=seq, use_cache=False, output_hidden_states=True)
        set_lora_enabled(rt.model, False)
        K = self.mem.shape[0]
        hs = out.hidden_states                                          # tuple (L+1) of (1, T, d)
        mem_layers = torch.stack([h[:, -K:, :] for h in hs], dim=0)     # (L+1, 1, K, d): anchor states per layer
        w = torch.softmax(self.layer_w.float(), dim=0).to(mem_layers.dtype).view(-1, 1, 1, 1)
        return (mem_layers * w).sum(0)                                  # (1, K, d): cross-layer transmission


# ============================================================================= LCC (faithful core, test-time)
class LCC:
    """Latent Context Compilation (2602.21221) core: NO global training. Per context, optimize K buffer tokens at
    TEST TIME via a reconstruction objective + a context-agnostic regularizer (keep the buffer on-manifold), then
    inject as a soft prefix. Faithful core (the paper uses a disposable LoRA 'compiler'; here a buffer)."""
    name = "lcc"; trainable = False

    def __init__(self) -> None:
        import os
        self.steps = int(os.environ.get("MEM_LCC_STEPS", "20"))      # more compile steps (paper optimizes per ctx)
        self.lr = float(os.environ.get("MEM_LCC_LR", "0.05"))
        self.max_recon = int(os.environ.get("MEM_LCC_RECON", "768")) # reconstruct (near-)full ctx, not just first 256

    def train(self, rt: Any, train_items: list, cfg: Any) -> None:
        return None

    def prefix(self, rt: Any, item: Any):
        model, embed, dev = rt.model, rt.embed, rt.dev
        cid = rt.ctx_ids(item); K = rt.K
        e0 = embed.weight.detach().float().norm(dim=-1).mean()
        buf = nn.Parameter((torch.randn(1, K, rt.d, device=dev, dtype=torch.float32) * 0.02))
        opt = torch.optim.Adam([buf], lr=self.lr)
        cap = min(cid.shape[1], self.max_recon)              # FULL-context reconstruction target (paper: compile whole ctx)
        ce = embed(cid).detach()
        for _ in range(self.steps):
            seq = torch.cat([buf.to(rt.bdt), ce[:, :cap]], dim=1)
            lg = model(
                inputs_embeds=seq,
                use_cache=False,
                logits_to_keep=cap + 1,
            ).logits[0, :cap].float()
            loss = F.cross_entropy(lg, cid[0, :cap]) + 1e-3 * (buf.float().norm(dim=-1).mean() - e0) ** 2
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return (F.normalize(buf.detach(), dim=-1) * e0).to(rt.bdt), 0


def register(registry: dict) -> None:
    for cls in (ICAE, X500, AOC, Beacon, ComprExIT, LCC):
        registry[cls.name] = cls
