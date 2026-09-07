"""GCMModel: one frozen base + self-compressor + toggleable read-LoRA + do-no-harm gate.

`encode()` compresses with the read-LoRA OFF (the base's own features). `generate()` computes a label-free gate
signal and either trusts the compressed memory M (read-LoRA ON: base reads M) or falls back to the full context
(read-LoRA OFF: the exact original base). Adapters (memory tokens, projection, read-LoRA) are the only trained
weights; the base is shared and frozen.
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from .compressor import SelfCompressor
from .lora import LoRALinear, add_enc_lora, add_lora, set_lora_enabled


def build_layer_copy(base: nn.Module, n_layers: int, dtype=None) -> nn.Module:
    """A separate TRAINABLE copy of the base inner-model's first ``n_layers`` blocks (+ embeddings/norm/rotary),
    weights initialized from the base (v1.7.5-style). Embeddings are frozen; the transformer blocks train.
    Used as the reconstruction decoder so it can specialize, independent of the frozen answer base."""
    cfg = deepcopy(base.config)
    cfg.num_hidden_layers = int(n_layers)
    cfg._attn_implementation = "eager"
    inner = type(base.model)(cfg)
    src = base.model.state_dict()
    keep = {k: v for k, v in src.items()
            if not k.startswith("layers.") or int(k.split(".")[1]) < n_layers}
    inner.load_state_dict(keep, strict=False)
    dev = base.get_input_embeddings().weight.device
    inner = inner.to(device=dev, dtype=dtype or base.get_input_embeddings().weight.dtype)
    for p in inner.parameters():
        p.requires_grad_(True)
    inner.get_input_embeddings().weight.requires_grad_(False)   # freeze the (huge) embedding copy
    return inner


class GCMModel(nn.Module):
    def __init__(self, base: nn.Module, tokenizer, n_memory: int = 16, depth: int | None = None,
                 proj_mult: int = 2, lora_rank: int = 32, normalize: bool = True, norm_mode: str = "", manifold_temp: float = 1.0,
                 vae: bool = False, chunk_size: int = 0, enc_lora_rank: int = 0, dec_layers: int = 0, xchunk: int = 0,
                 recur: bool = False, state_cap: int = 0,
                 lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj",
                                        "gate_proj", "up_proj", "down_proj")) -> None:
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.tok = tokenizer
        self.compressor = SelfCompressor(base, n_memory=n_memory, depth=depth, proj_mult=proj_mult, normalize=normalize,
                                         norm_mode=norm_mode, manifold_temp=manifold_temp, vae=vae, chunk_size=chunk_size,
                                         xchunk=xchunk, recur=recur, state_cap=state_cap)
        # cover BOTH attn types (full q/k/v/o + DeltaNet in_proj_qkv/out_proj) AND the MLP (gate/up/down_proj),
        # for every layer. enc-LoRA attaches to these same wrapped modules, so it gets the same coverage.
        self.lora_params = add_lora(base, rank=lora_rank, targets=lora_targets, dtype=torch.bfloat16)
        # encode-phase LoRA (ablation: should the base's ENCODER be trained?) -- a 2nd adapter, active only in encode.
        self.enc_lora_params = []
        if enc_lora_rank > 0:
            self.enc_lora_params = add_enc_lora(base, rank=enc_lora_rank, dtype=torch.bfloat16)
            self.compressor.use_enc_lora = True
        self._embed = base.get_input_embeddings()
        # learnable position-query slot, repeated to Lc in reconstruct() (M-only reconstruction; carries no ctx content)
        self.rec_slot = nn.Parameter((torch.randn(self.compressor.d) * 0.02).to(self._embed.weight.device, self._embed.weight.dtype))
        # optional SEPARATE trainable N-layer decoder for reconstruction (v1.7.5-style); 0 => use the frozen base + read-LoRA
        self.decoder = build_layer_copy(base, dec_layers, dtype=torch.bfloat16) if dec_layers > 0 else None

    # ---- compression ----
    def encode(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, conditional: bool = True) -> torch.Tensor:
        set_lora_enabled(self.base, False)              # encode uses the frozen base's own features (no read-LoRA)
        return self.compressor.encode(ctx_ids, query_ids, conditional)["memory"]

    def reconstruct(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, m_only: bool = True) -> torch.Tensor:
        """Phase-1 reconstruction objective: the memory M ALONE must regenerate the context. The decoder input is
        ONLY M (which already carries VAE noise when enabled) followed by Lc learnable position-query SLOTS, ordered
        [M ; slots] under PLAIN CAUSAL (no custom mask): M attends only M; each slot attends M + earlier slots. The ctx
        tokens are NEVER fed, so there is no teacher-forcing leak (we changed the INPUT instead of masking). Slot j ->
        tied LM head -> predict ctx[j]. cond = GCM_RECON_COND (default Mq). Grad -> mem+proj(+enc/read-LoRA)+rec_slot."""
        import os
        if os.environ.get("GCM_RECON_MODE", "slot").lower() == "repeat":     # kvzip-style repeat-prompt reconstruction
            return self.recon_repeat(ctx_ids, query_ids)[0]
        cap = int(os.environ.get("GCM_RECON_MAXCTX", "1024"))               # bound recon memory
        if ctx_ids.shape[1] > cap:
            ctx_ids = ctx_ids[:, :cap]
        cond = os.environ.get("GCM_RECON_COND", "Mq").lower() not in ("0", "m0", "off", "false")  # default Mq
        set_lora_enabled(self.base, False)                                  # M read from the frozen base's features
        M = self.compressor.encode(ctx_ids, query_ids, conditional=cond)["memory"]     # [1,K,d] (+VAE noise if enabled)
        set_lora_enabled(self.base, True)                                   # the read path IS the decoder
        K = M.shape[1]; Lc = int(ctx_ids.shape[1]); d = M.shape[-1]
        slots = self.rec_slot.to(M.dtype).view(1, 1, d).expand(1, Lc, d)    # position-query slots: NO ctx content
        seq = torch.cat([M, slots], 1)                                      # [M ; slots]: M FIRST => plain causal suffices
        # No custom mask: causal already gives M attends only M, slots attend all M (+ content-free earlier slots).
        # No ctx tokens are in the input, so there is no leak to mask out -- we changed the INPUT instead of masking.
        dec = self.decoder if self.decoder is not None else self.base.model   # separate trainable decoder, else frozen base+read-LoRA
        h = dec(inputs_embeds=seq, use_cache=False).last_hidden_state
        hc = h[:, K:K + Lc, :]                                              # slot j -> predicts ctx[j] (parallel readout)
        logits = torch.nn.functional.linear(hc, self._embed.weight).float()
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ctx_ids.reshape(-1))

    def _repeat_prompt_ids(self) -> torch.Tensor:
        """Cached ids for the kvzip-style repeat instruction (query-agnostic)."""
        if getattr(self, "_rp_ids", None) is None:
            import os
            txt = os.environ.get("GCM_REPEAT_PROMPT", "\nRepeat the previous context verbatim:\n")
            ids = self.tok(txt, add_special_tokens=False, return_tensors="pt").input_ids
            self._rp_ids = ids.to(self._embed.weight.device)
        return self._rp_ids

    def recon_repeat(self, ctx_ids: torch.Tensor, query_ids: torch.Tensor, cond: bool | None = None):
        """KVzip-style export of the soft memory: prompt the FROZEN base to REPEAT the context from M alone,
        teacher-forced, in ONE forward. Returns (L_repeat, r_t) where
          L_repeat = mean_t -log p(ctx_t | M, <repeat>, ctx_<t)   -> trains M as a sufficient statistic (grad->mem+proj),
          r_t      = per-token reconstruction NLL = the kvzip reconstruction-IMPORTANCE of ctx token t w.r.t. OUR M
                     (high = M failed to store it = a needle to keep verbatim / a do-no-harm gate trigger).
        By default query-AGNOSTIC (M0), matching kvzip's query-agnostic importance."""
        import os
        cap = int(os.environ.get("GCM_RECON_MAXCTX", "1024"))
        if ctx_ids.shape[1] > cap:
            ctx_ids = ctx_ids[:, :cap]
        if cond is None:                                                     # kvzip is query-agnostic -> default M0
            cond = os.environ.get("GCM_RECON_COND", "M0").lower() not in ("0", "m0", "off", "false")
        set_lora_enabled(self.base, False)
        M = self.compressor.encode(ctx_ids, query_ids, conditional=cond)["memory"]     # [1,K,d]
        return self._repeat_nll_fromM(M, ctx_ids)

    def _repeat_nll_fromM(self, M: torch.Tensor, ctx_ids: torch.Tensor):
        """Core of the repeat-prompt reconstruction: given ANY memory M, teacher-force the base to repeat ctx and
        return (mean NLL, per-token NLL). Factored out so a control can pass a RANDOM M (leak/trick test)."""
        set_lora_enabled(self.base, True)                                    # the read path IS the decoder
        K = M.shape[1]
        pids = self._repeat_prompt_ids(); P = pids.shape[1]; Lc = int(ctx_ids.shape[1])
        seq = torch.cat([M, self._embed(pids), self._embed(ctx_ids)], 1)     # [M ; <repeat> ; ctx] teacher-forced
        h = self.base.model(inputs_embeds=seq, use_cache=False).last_hidden_state
        hc = h[:, K + P - 1: K + P - 1 + Lc, :]                             # position (before ctx_t) -> predict ctx_t
        logits = torch.nn.functional.linear(hc, self._embed.weight).float()
        r = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), ctx_ids.reshape(-1),
                                              reduction="none")              # per-token NLL = reconstruction importance
        return r.mean(), r.detach()

    @torch.no_grad()
    def gate_signal(self, M: torch.Tensor, ctx_ids: torch.Tensor, query_ids: torch.Tensor) -> dict:
        """Label-free do-no-harm signals. (a) answer-token uncertainty on [M;q]; (b) the M0<->Mq gap + reconstruction
        faithfulness (neg_recon / dlogit) -- the strongest gates in v1.7.5. Oriented so HIGHER = safer-to-compress."""
        import os
        requested = {
            value.strip()
            for value in os.environ.get(
                "GCM_GATE_SIGNALS",
                "conf,margin,neg_entropy,neg_recon,dlogit,targ,neg_recon_repeat,neg_maxr",
            ).split(",")
            if value.strip()
        }
        set_lora_enabled(self.base, True)
        qe = self._embed(query_ids)
        pq = self.base(inputs_embeds=torch.cat([M, qe], 1), use_cache=False, logits_to_keep=1).logits[0, -1].float().softmax(-1)
        top2 = pq.topk(2).values
        sig = {"conf": float(top2[0]), "margin": float(top2[0] - top2[1]),
               "neg_entropy": float((pq * (pq + 1e-9).log()).sum()), "neg_recon": 0.0, "dlogit": 0.0,
               "targ": 0.0, "no_conf": 0.0}
        try:                                              # TARG baseline: training-free base-uncertainty (no-ctx, read-LoRA OFF)
            if "targ" not in requested and "no_conf" not in requested:
                raise RuntimeError("TARG signal not requested")
            set_lora_enabled(self.base, False)
            pb = self.base(inputs_embeds=qe, use_cache=False, logits_to_keep=1).logits[0, -1].float().softmax(-1)
            tb = pb.topk(2).values
            sig["targ"] = float(tb[0] - tb[1]); sig["no_conf"] = float(tb[0])   # base prior confidence (no context)
            set_lora_enabled(self.base, True)
        except Exception:
            pass
        try:                                              # M0<->Mq gap + reconstruction (query-independent faithfulness)
            if not requested.intersection({"neg_recon", "dlogit"}):
                raise RuntimeError("M0/reconstruction signals not requested")
            m0 = self.encode(ctx_ids, query_ids, conditional=False)
            set_lora_enabled(self.base, True)
            p0 = self.base(inputs_embeds=torch.cat([m0, qe], 1), use_cache=False, logits_to_keep=1).logits[0, -1].float().softmax(-1)
            sig["dlogit"] = float((pq * ((pq + 1e-9).log() - (p0 + 1e-9).log())).sum())   # KL(Mq||M0), first token
            sig["neg_recon"] = -float(self.reconstruct(ctx_ids, query_ids, m_only=True))  # high = M reconstructs ctx
        except Exception:
            pass
        try:                                              # kvzip-style repeat-prompt reconstruction (query-agnostic)
            if not requested.intersection({"neg_recon_repeat", "neg_maxr", "neg_recon_repeat_rand"}):
                raise RuntimeError("repeat-reconstruction signals not requested")
            import os as _os
            _l, _r = self.recon_repeat(ctx_ids, query_ids)
            set_lora_enabled(self.base, True)
            sig["neg_recon_repeat"] = -float(_l)          # high = M repeats ctx well -> safe to compress
            sig["neg_maxr"] = -float(_r.max())            # high (=low max NLL) = no un-repeatable needle was dropped
            if _os.environ.get("GCM_GATE_RANDM", "0") == "1":   # LEAK CONTROL: random M matched to M's stats.
                cap = int(_os.environ.get("GCM_RECON_MAXCTX", "1024")); _c = ctx_ids[:, :cap]  # if AUROC(rand)~=AUROC(real),
                set_lora_enabled(self.base, False)                                             # the signal is ctx-intrinsic (trick), not M.
                _M = self.compressor.encode(_c, query_ids, conditional=False)["memory"]
                _Mr = torch.randn_like(_M) * _M.std() + _M.mean()
                _lr, _ = self._repeat_nll_fromM(_Mr, _c)
                sig["neg_recon_repeat_rand"] = -float(_lr)
                set_lora_enabled(self.base, True)
        except Exception:
            pass
        return sig

    @torch.no_grad()
    def gate_probe_features(
        self,
        M: torch.Tensor,
        query_ids: torch.Tensor,
        ctx_tokens: int,
    ) -> dict[str, float]:
        """Cheap joint query-memory features for the Belikova-style probe baseline."""
        mf = M[0].float()
        qf = self._embed(query_ids)[0].float()
        mnorm = mf.norm(dim=-1)
        qnorm = qf.norm(dim=-1)
        mm = mf.mean(0)
        qm = qf.mean(0)
        mmn = torch.nn.functional.normalize(mm, dim=0)
        qmn = torch.nn.functional.normalize(qm, dim=0)
        slots = torch.nn.functional.normalize(mf, dim=-1)
        query_tokens = torch.nn.functional.normalize(qf, dim=-1)
        slot_q = slots @ qmn
        token_m = query_tokens @ mmn
        anisotropy = (slots @ mmn).mean()
        return {
            "memory_tokens": float(mf.shape[0]),
            "query_tokens": float(qf.shape[0]),
            "context_tokens": float(ctx_tokens),
            "memory_ratio": float(mf.shape[0] / max(1, ctx_tokens)),
            "memory_norm_mean": float(mnorm.mean()),
            "memory_norm_std": float(mnorm.std()),
            "memory_norm_max": float(mnorm.max()),
            "query_norm_mean": float(qnorm.mean()),
            "query_norm_std": float(qnorm.std()),
            "mean_cosine": float((mmn * qmn).sum()),
            "max_slot_query_cosine": float(slot_q.max()),
            "mean_slot_query_cosine": float(slot_q.mean()),
            "max_query_memory_cosine": float(token_m.max()),
            "memory_anisotropy": float(anisotropy),
        }

    @torch.no_grad()
    def mc_loglik(self, prefix: torch.Tensor | None, query_ids: torch.Tensor, options: list, use_memory: bool) -> list:
        """Length-normalized loglik of each option given [prefix ; query]. prefix=None => no_ctx. Higher = more likely."""
        set_lora_enabled(self.base, bool(use_memory))
        qe = self._embed(query_ids)
        base_seq = torch.cat([prefix, qe], 1) if prefix is not None else qe
        option_ids = [
            self.tok(" " + str(opt).strip(), add_special_tokens=False).input_ids
            for opt in options
        ]
        # Letter-scored MC (QuALITY, MuSR, RCA) has one token per option. One next-token
        # distribution scores every letter, avoiding four identical long-context forwards.
        if option_ids and all(len(ids) == 1 for ids in option_ids):
            lp = self.base(inputs_embeds=base_seq, use_cache=False, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
            return [float(lp[ids[0]]) for ids in option_ids]
        out = []
        for ids in option_ids:
            oid = torch.tensor([ids], device=qe.device, dtype=query_ids.dtype)
            if oid.shape[1] == 0:
                out.append(float("-inf")); continue
            seq = torch.cat([base_seq, self._embed(oid)], 1)
            keep = int(oid.shape[1]) + 1
            lp = self.base(inputs_embeds=seq, use_cache=False, logits_to_keep=keep).logits[0, :oid.shape[1]].float().log_softmax(-1)
            out.append(float(lp.gather(-1, oid[0].unsqueeze(-1)).sum() / oid.shape[1]))
        return out

    @torch.no_grad()
    def _gen(self, prefix: torch.Tensor | None, query_ids: torch.Tensor, use_memory: bool, max_new: int) -> str:
        set_lora_enabled(self.base, bool(use_memory))   # ON to read M; OFF = exact base on full ctx (do-no-harm)
        qe = self._embed(query_ids)
        seq = torch.cat([prefix, qe], 1) if prefix is not None else qe
        ids: list[int] = []
        for _ in range(max_new):
            nxt = int(self.base(inputs_embeds=seq, use_cache=False, logits_to_keep=1).logits[:, -1].argmax(-1))
            if nxt == getattr(self.tok, "eos_token_id", -1):
                break
            ids.append(nxt)
            seq = torch.cat([seq, self._embed(torch.tensor([[nxt]], device=seq.device))], 1)
        return self.tok.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def _gen_batch(self, prefixes, query_ids_list, use_memory: bool, max_new: int, batch_size: int = 8) -> list:
        """Batched greedy generation with KV cache (HF generate). Same greedy decode as _gen() per item, but O(n)
        per token via cache and B items at once. prefixes[i]: [1,P_i,d] input-embeds or None; query_ids_list[i]:
        [1,Q_i] token ids. Left-pads each [prefix;query] within a batch (decoder-only => pad on the LEFT)."""
        set_lora_enabled(self.base, bool(use_memory))
        dev = self._embed.weight.device
        eos = getattr(self.tok, "eos_token_id", None)
        pad_id = getattr(self.tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos if eos is not None else 0
        pad_emb = self._embed(torch.tensor([[int(pad_id)]], device=dev))                  # [1,1,d]
        out: list = []
        for s in range(0, len(query_ids_list), batch_size):
            qs = query_ids_list[s:s + batch_size]
            ps = prefixes[s:s + batch_size]
            seqs = []
            for p, q in zip(ps, qs):
                qe = self._embed(q.to(dev))
                seqs.append(torch.cat([p.to(dev).to(qe.dtype), qe], 1) if p is not None else qe)   # [1,L_i,d]
            L = max(int(x.shape[1]) for x in seqs); B = len(seqs); d = int(seqs[0].shape[-1])
            emb = pad_emb.to(seqs[0].dtype).expand(B, L, d).clone()
            mask = torch.zeros(B, L, dtype=torch.long, device=dev)
            for i, x in enumerate(seqs):
                li = int(x.shape[1])
                emb[i, L - li:] = x[0]; mask[i, L - li:] = 1                               # LEFT pad
            gen = self.base.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new,
                                     do_sample=False, num_beams=1, use_cache=True,
                                     pad_token_id=int(pad_id), eos_token_id=eos)
            for i in range(B):
                out.append(self.tok.decode(gen[i], skip_special_tokens=True))
        return out

    @torch.no_grad()
    def generate(self, context: str, query: str, tau: float = 0.0, max_new: int = 16) -> dict:
        """Compress the context; trust M when the gate signal >= tau, else fall back to the full context."""
        dev = self.base.get_input_embeddings().weight.device
        cid = self.tok(context, return_tensors="pt").input_ids.to(dev)
        qid = self.tok(query, return_tensors="pt").input_ids.to(dev)
        M = self.encode(cid, qid)
        sig = self.gate_signal(M, cid, qid)
        if sig["margin"] >= tau:
            return {"text": self._gen(M, qid, True, max_new), "path": "compress", "signal": sig}
        return {"text": self._gen(self._embed(cid), qid, False, max_new), "path": "full", "signal": sig}

    # ---- adapters only (base is shared/frozen, never saved) ----
    def trainable_parameters(self) -> list[nn.Parameter]:
        dec = [p for p in self.decoder.parameters() if p.requires_grad] if self.decoder is not None else []
        return self.compressor.trainable_parameters() + self.lora_params + self.enc_lora_params + [self.rec_slot] + dec

    def save_adapters(self, path: str) -> None:
        loras = [m for m in self.base.modules() if isinstance(m, LoRALinear)]
        torch.save({"mem": self.compressor.mem.detach().cpu(),
                    "proj": self.compressor.proj.state_dict(),
                    "embed_scale": self.compressor.embed_scale.detach().cpu(),
                    "lora": [(m.lora_A.detach().cpu(), m.lora_B.detach().cpu()) for m in loras],
                    "enc_lora": [
                        (m.enc_A.detach().cpu(), m.enc_B.detach().cpu())
                        for m in loras if m.enc_A is not None
                    ],
                    "rec_slot": self.rec_slot.detach().cpu(),
                    "decoder": self.decoder.state_dict() if self.decoder is not None else None,
                    "config": {
                        "K": self.compressor.K,
                        "depth": self.compressor.depth,
                        "lora_modules": len(loras),
                        "enc_lora_modules": sum(m.enc_A is not None for m in loras),
                        "has_decoder": self.decoder is not None,
                    }}, path)

    def load_adapters(self, path: str, map_location=None) -> "GCMModel":
        sd = torch.load(path, map_location=map_location or self.base.get_input_embeddings().weight.device, weights_only=False)
        loras = [m for m in self.base.modules() if isinstance(m, LoRALinear)]
        saved_loras = sd["lora"]
        if len(saved_loras) != len(loras):
            raise ValueError(
                f"adapter LoRA module count mismatch: checkpoint={len(saved_loras)} model={len(loras)}"
            )
        with torch.no_grad():
            self.compressor.mem.copy_(sd["mem"].to(self.compressor.mem))
            self.compressor.proj.load_state_dict(sd["proj"])
            if "embed_scale" in sd:
                self.compressor.embed_scale.copy_(
                    sd["embed_scale"].to(self.compressor.embed_scale)
                )
            for m, (A, B) in zip(loras, saved_loras):
                m.lora_A.copy_(A.to(m.lora_A)); m.lora_B.copy_(B.to(m.lora_B))
            saved_enc_loras = sd.get("enc_lora")
            current_enc_loras = [m for m in loras if m.enc_A is not None]
            if saved_enc_loras is not None:
                if len(saved_enc_loras) != len(current_enc_loras):
                    raise ValueError(
                        "adapter encode-LoRA module count mismatch: "
                        f"checkpoint={len(saved_enc_loras)} model={len(current_enc_loras)}"
                    )
                for module, (A, B) in zip(current_enc_loras, saved_enc_loras):
                    module.enc_A.copy_(A.to(module.enc_A))
                    module.enc_B.copy_(B.to(module.enc_B))
            if "rec_slot" in sd:
                self.rec_slot.copy_(sd["rec_slot"].to(self.rec_slot))
            if self.decoder is not None and sd.get("decoder") is not None:
                self.decoder.load_state_dict(sd["decoder"])
        return self
