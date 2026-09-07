"""GCM (the Gated Compressor Module), as a gcm-harness Method, so it slots into the same harness + decision
reporter as the baselines. (Method name "gcm"; in paper tables labelled "GCM (ours)".)

Training (v3) is the full autoencoder objective (bf16 by default, matching deployment + the bf16 baselines):
  - L_task (primary): gold-answer CE -- [Mq ; query ; gold] through the frozen base must predict the gold tokens,
    forcing M to actually carry the answer (the recipe that makes Gist win in-task). v2 distilled the full-ctx
    teacher's teacher-forced response instead, which is too weak (the teacher is weak on noisy chunks and the
    response sits in the input), so OURS-v2 sat below no-ctx; v3 learns from gold.
  - L_distill (optional, off by default): top-k KL to the full-ctx teacher over its own response R.
  - L_uncond: the trainable decoder reconstructs the context FROM the unconditional memory M0 (lossless signal).
  - min-dev: ||Mq - M0||^2 anchors the conditioned memory to the (context-faithful) unconditional one, so Mq
    deviates only when the query needs it.
At eval the K soft tokens M replace the context ([M ; query] -> frozen base -> answer); query always fed.

Next increment (Q3): the gate/signals (ΔCode/ΔLogits from the M0<->Mq gap + structural signals).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger


class GCM:
    name = "gcm"
    trainable = True

    def __init__(self, n_layers: int = 4, n_memory: int = 64, condition_on_query: bool = True):
        self.n_layers = int(n_layers)
        self.n_memory = int(n_memory)
        self.condition = bool(condition_on_query)
        self.chunk_size = 0         # v1.7.4: >0 => length-adaptive chunked memory (fixed ratio, scales with len)
        self.chunk_keep = 0         # v1.7.4: >0 => query-relevance routing, keep top-`keep` chunk-memories
        self.svc: Any = None
        self.disc: Any = None       # adversarial discriminator (set iff trained with lam_adv>0); doubles as the gate
        self.adv_layer: int = 18
        self.use_base_lora: bool = False   # set in train(); harness keeps LoRA on for compress, off for fallback/full

    def _enc(self, rt: Any, cid: torch.Tensor, qid: torch.Tensor, cond: bool) -> dict:
        """Dispatch to length-adaptive chunked encode (v1.7.4) when chunk_size>0, else single-pass encode."""
        if self.chunk_size and self.chunk_size > 0:
            return self.svc.encode_chunked(cid, qid, self.chunk_size, condition_on_query=cond, keep=self.chunk_keep)
        return self.svc.encode(cid, qid, condition_on_query=cond)

    def mem_len(self, n_ctx: int) -> int:
        """Deployed soft-prompt length (for the cost column): scales with ctx when chunked, else fixed K."""
        if self.chunk_size and self.chunk_size > 0 and n_ctx > self.chunk_size:
            import math
            s = math.ceil(n_ctx / self.chunk_size)
            if self.chunk_keep and self.chunk_keep > 0:
                s = min(s, self.chunk_keep)
            return s * self.n_memory
        return self.n_memory

    def prefix(self, rt: Any, item: Any) -> tuple[torch.Tensor, int]:
        assert self.svc is not None, "GCM.train must run before prefix"
        with torch.no_grad():
            out = self._enc(rt, rt.ctx_ids(item), rt.query_ids(item), self.condition)
        return out["memory"].to(rt.bdt), 0  # (1,*,d) soft prefix, no gist mask

    @torch.no_grad()
    def signals(self, rt: Any, item: Any) -> dict:
        """Per-item gate signals (the verifier read-out), logged at eval so the reporter can score each as a
        compress/fallback gate (AUROC / F1 / cost-coverage). Three families:
          - VERIFIABILITY (the M0<->Mq gap + reconstruction): neg_recon (how well M0 reconstructs THIS context;
            high => the memory is faithful => safe), dcode/dlogit (how much the query moves the memory/behaviour).
          - BASE-READABLE uncertainty on the deployed [Mq;q] run: conf / margin / neg_entropy (TARG-style).
          - GEOMETRY: mnorm (memory norm). Signals oriented so HIGHER = safer-to-compress; raw + negated logged
            for the ambiguous gap signals so the reporter's AUROC reveals direction."""
        assert self.svc is not None, "GCM.train must run before signals"
        cid, qid = rt.ctx_ids(item), rt.query_ids(item)
        e0 = self._enc(rt, cid, qid, False)
        eq = self._enc(rt, cid, qid, True)
        m0, mq = e0["memory"].to(rt.bdt), eq["memory"].to(rt.bdt)
        qe = rt.embed(qid)

        def first_logits(m: torch.Tensor) -> torch.Tensor:
            return rt.model(inputs_embeds=torch.cat([m, qe], 1), use_cache=False).logits[0, -1].float()

        lg0, lgq = first_logits(m0), first_logits(mq)
        p0, pq = lg0.softmax(-1), lgq.softmax(-1)
        eps = 1e-9
        dlogit = float((pq * ((pq + eps).log() - (p0 + eps).log())).sum())   # KL(Mq||M0) on the first answer token
        top2 = torch.topk(pq, 2).values
        conf, margin = float(top2[0]), float(top2[0] - top2[1])
        ent = float(-((pq + eps).log() * pq).sum())
        dcode = float((eq["memory_raw"] - e0["memory_raw"]).pow(2).mean().sqrt())
        mnorm = float(mq.float().pow(2).sum(-1).sqrt().mean())
        recon = float(self.svc.reconstruct(cid, e0["memory"], rt.embed.weight)) if self.svc.decoder is not None else 0.0
        sig = {"conf": conf, "margin": margin, "neg_entropy": -ent, "neg_recon": -recon,
               "dcode": dcode, "neg_dcode": -dcode, "dlogit": dlogit, "neg_dlogit": -dlogit, "mnorm": mnorm}
        if self.disc is not None:
            # the adversarial discriminator as a LEARNED gate: p = P(D thinks [Mq;q] is COMPRESSED) at layer ell.
            # D was trained while the compressor pushed it toward "full"; so low p = M fooled D (looks lossless),
            # high p = D detected an artifact, p~0.5 = D is unsure (OOD). We log p + its orientations/entropy so
            # the reporter can score the competing gate strategies (compress-iff-low-p / iff-confident / iff-high-p).
            hs = rt.model(inputs_embeds=torch.cat([mq, qe], 1), use_cache=False,
                          output_hidden_states=True).hidden_states[self.adv_layer]
            hq = hs[0, mq.shape[1]:].mean(0).unsqueeze(0)   # bf16 (disc is bf16)
            p = float(torch.sigmoid(self.disc(hq).float()).item())
            hbin = -(p * math.log(p + eps) + (1 - p) * math.log(1 - p + eps))   # binary entropy of D
            sig.update({"disc_p": p, "disc_negp": -p, "disc_conf": abs(p - 0.5), "disc_negent": -hbin})
        return sig

    def train(self, rt: Any, train_items: list, cfg: Any) -> None:
        from svc.compressor import SelfVerifyingCompressor

        model, embed, tok, dev = rt.model, rt.embed, rt.tok, rt.dev
        T, topk, temp = cfg.distill_tokens, cfg.distill_topk, cfg.distill_temp
        enc_layers = int(getattr(cfg, "enc_layers", self.n_layers))
        n_dec = int(getattr(cfg, "n_dec_layers", 2))
        lam_rec, lam_dev = float(getattr(cfg, "lam_rec", 1.0)), float(getattr(cfg, "lam_dev", 0.05))
        lam_task, lam_distill = float(getattr(cfg, "lam_task", 1.0)), float(getattr(cfg, "lam_distill", 0.0))
        enc_init = str(getattr(cfg, "enc_init", "copy"))
        if bool(getattr(cfg, "gcm_agnostic", False)):  # ablation: query-AGNOSTIC memory (M0) for train + eval
            self.condition = False
        self.chunk_size = int(getattr(cfg, "chunk_size", 0))   # v1.7.4 length-adaptive memory
        self.chunk_keep = int(getattr(cfg, "chunk_keep", 0))
        # Default bf16 trainable params (matches deployment + the bf16 baselines, so the comparison is
        # apples-to-apples); opt into an fp32 AdamW master copy via --train-fp32 only for stability debugging.
        tdt = torch.float32 if bool(getattr(cfg, "train_fp32", False)) else None
        self.n_memory = rt.K
        self.svc = SelfVerifyingCompressor(model, n_layers=enc_layers, n_memory=rt.K,
                                           n_dec_layers=n_dec, train_dtype=tdt, init=enc_init,
                                           m_norm_match=str(getattr(cfg, "m_norm_match", "off")),
                                           m_manifold_temp=float(getattr(cfg, "m_manifold_temp", 0.0)))
        if self.chunk_size > 0:   # v1.7.4: chunked training retains S encoder-graphs => checkpoint to fit memory
            for mdl in (self.svc.encoder, self.svc.decoder):
                if mdl is not None:
                    try:
                        mdl.train()
                        mdl.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        logger.info(f"[gcm] grad-checkpointing ON for chunked training ({mdl.__class__.__name__})")
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"[gcm] grad-checkpointing unavailable: {e!r}")
        base_lora_rank = int(getattr(cfg, "base_lora_rank", 0))   # A3 (v1.7.1.3) — non-merged: on for compress, off for fallback
        lora_params: list = []
        self.use_base_lora = base_lora_rank > 0   # harness toggles LoRA on for the compressed path, off for full/fallback
        if base_lora_rank > 0:
            from svc.lora import add_lora
            lora_params = add_lora(model, base_lora_rank, dtype=rt.bdt)   # bf16 (no fp32)
            logger.info(f"[gcm] A3 base LoRA rank={base_lora_rank}: +{len(lora_params)} param tensors (base stays frozen)")
        params = [p for p in self.svc.parameters() if p.requires_grad] + lora_params
        opt = torch.optim.AdamW(params, lr=cfg.lr)
        out_w = embed.weight  # frozen tied LM head for the reconstruction (L_uncond) vocab projection

        # ADVERSARIAL losslessness (optional): a discriminator reads the FROZEN base's layer-`adv_layer` hidden at
        # the query positions and tries to tell M-induced from full-context-induced; the compressor is trained to
        # FOOL it, i.e. to make [M;q] induce the SAME internal state as [ctx;q]. If D cannot tell, M is
        # behaviorally lossless. This is a direct "make M carry the context" signal, unlike gold-CE/reconstruction.
        lam_adv = float(getattr(cfg, "lam_adv", 0.0))
        lam_contrast = float(getattr(cfg, "lam_contrast", 0.0))   # v1.8.0: InfoNCE alignment (alternative to adv)
        lam_align = float(getattr(cfg, "lam_align", 0.0))         # v1.8.0: PURE positive alignment, NO negatives (ablation)
        align_layer = int(getattr(cfg, "align_layer", getattr(cfg, "adv_layer", 18)))
        adv_layer = align_layer                                   # adv + contrast share the alignment layer
        contrast_temp = float(getattr(cfg, "contrast_temp", 0.2))
        bank_size = int(getattr(cfg, "contrast_bank", 256))
        need_hidden = (lam_adv > 0) or (lam_contrast > 0) or (lam_align > 0)
        cbank_all = None                                          # InfoNCE teacher-negative pool (set after val split)
        disc = d_opt = None
        if lam_adv > 0:
            dh = model.config.hidden_size
            disc = torch.nn.Sequential(torch.nn.Linear(dh, dh // 4), torch.nn.GELU(),
                                       torch.nn.Linear(dh // 4, 1)).to(dev, rt.bdt)   # bf16 (no fp32)
            d_opt = torch.optim.AdamW(disc.parameters(), lr=2e-4)
            self.disc, self.adv_layer = disc, adv_layer  # persist: the trained D is reused as the learned gate

        # cache per-item targets: the GOLD answer ids (primary task signal) + (optional) full-ctx teacher top-k
        # over its own response R (auxiliary context-distillation, only if lam_distill>0).
        study: list[dict] = []
        with torch.no_grad():
            for it in train_items:
                cid, qid = rt.ctx_ids(it), rt.query_ids(it)
                g = tok(" " + str(it.gold).strip(), add_special_tokens=False).input_ids[:T]
                if not g:
                    continue
                rec = {"cid": cid, "qid": qid, "gold": torch.tensor([g], device=dev)}
                if need_hidden:  # cache the full-context align_layer hidden at the query positions (adv/contrast target)
                    ce0, qe0 = embed(cid), embed(qid)
                    hf = model(inputs_embeds=torch.cat([ce0, qe0], 1), use_cache=False,
                               output_hidden_states=True).hidden_states[align_layer][0, ce0.shape[1]:]
                    rec["h_full"] = hf.mean(0).detach()
                if lam_distill > 0:
                    # v1.7.6 PERF: distill the full-ctx teacher's distribution over the GOLD answer in ONE forward.
                    # (The old token-by-token self-generation = up to T no-cache forwards/item, the precompute bottleneck;
                    #  gold-forced context-distillation is a standard, far cheaper target and keeps distill load-bearing.)
                    ce, qe = embed(cid), embed(qid)
                    Rt = rec["gold"]
                    Pt = ce.shape[1] + qe.shape[1]
                    tlg = model(inputs_embeds=torch.cat([ce, qe, embed(Rt)], 1),
                                use_cache=False).logits[0, Pt - 1:Pt - 1 + Rt.shape[1]].float()
                    rec["R"], (rec["tk_logit"], rec["tk_idx"]) = Rt, tlg.topk(topk, dim=-1)
                study.append(rec)
        logger.info(f"[gcm] targets: {len(study)} (enc={enc_layers}, dec={n_dec}, K={rt.K}, cond={self.condition}, "
                    f"fp32={tdt is not None}, lam_task={lam_task}, lam_distill={lam_distill}, lam_rec={lam_rec}, lam_dev={lam_dev})")

        # ---- v1.7.5 STABILITY: effective-batch via grad accumulation + LR warmup/cosine + EMA log + held-out val ----
        accum = max(1, int(getattr(cfg, "grad_accum", 8)))
        warm = float(getattr(cfg, "warmup_frac", 0.05))
        cosine = bool(getattr(cfg, "lr_cosine", True))
        nval = min(8, len(study) // 10)
        val_study = study[-nval:] if nval else []
        if nval:
            study = study[:-nval]
        if lam_contrast > 0 and study:  # teacher negatives = OTHER train samples' full-induced pooled hidden (SimCLR-style)
            for _i, _s in enumerate(study):
                _s["cbank_idx"] = _i                              # so each sample can exclude its OWN positive from negs
            cbank_all = F.normalize(torch.stack([s["h_full"] for s in study]).float(), dim=-1)
        opt_total = max(1, cfg.steps // accum)
        warm_steps = max(1, int(warm * opt_total))

        def _lr_lambda(o: int) -> float:
            if o < warm_steps:
                return (o + 1) / warm_steps
            prog = (o - warm_steps) / max(1, opt_total - warm_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))) if cosine else 1.0

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        ema: float | None = None

        @torch.no_grad()
        def _val_loss() -> float:
            if not val_study:
                return float("nan")
            tot = 0.0
            for s in val_study:
                m = self._enc(rt, s["cid"], s["qid"], self.condition)["memory"].to(rt.bdt)
                qv, gv = embed(s["qid"]), embed(s["gold"])
                P = m.shape[1] + qv.shape[1]
                lg = model(inputs_embeds=torch.cat([m, qv, gv], 1), use_cache=False).logits[0, P - 1:P - 1 + s["gold"].shape[1]].float()
                tot += float(F.cross_entropy(lg, s["gold"][0]))
            return tot / len(val_study)

        step = 0
        _patience = int(getattr(cfg, "patience", 0)); _best_val = float("inf"); _no_improve = 0  # v1.7.6 early-stop on val plateau
        while step < cfg.steps and study:
            for i, s in enumerate(study):
                # the deployed memory (conditioned Mq by default; M0 if gcm_agnostic) drives the task loss
                encq = self._enc(rt, s["cid"], s["qid"], self.condition)
                mq = encq["memory"].to(rt.bdt)
                qe = embed(s["qid"])
                # gold-answer CE: [Mq ; q ; gold] must predict the gold tokens (forces M to carry the answer)
                ge = embed(s["gold"])
                seq = torch.cat([mq, qe, ge], 1)
                Pg = mq.shape[1] + qe.shape[1]
                out_m = model(inputs_embeds=seq, use_cache=False, output_hidden_states=need_hidden)
                glg = out_m.logits[0, Pg - 1:Pg - 1 + s["gold"].shape[1]].float()
                l_task = F.cross_entropy(glg, s["gold"][0])
                loss = lam_task * l_task
                l_adv = torch.zeros((), device=dev)
                l_contrast = torch.zeros((), device=dev)
                l_align = torch.zeros((), device=dev)
                if need_hidden:
                    # M-run align_layer hidden at the query positions (carries grad through mq -> compressor)
                    hM = out_m.hidden_states[align_layer][0, mq.shape[1]:mq.shape[1] + qe.shape[1]].mean(0).unsqueeze(0)
                    hF = s["h_full"].unsqueeze(0)
                    if lam_adv > 0:  # ADVERSARIAL: D tells M-induced(=1) from full-induced(=0); compressor fools D
                        ones, zeros = torch.ones(1, 1, device=dev), torch.zeros(1, 1, device=dev)
                        d_loss = F.binary_cross_entropy_with_logits(disc(hM.detach()), ones) + \
                            F.binary_cross_entropy_with_logits(disc(hF), zeros)
                        d_opt.zero_grad(); d_loss.backward(); d_opt.step()
                        l_adv = F.binary_cross_entropy_with_logits(disc(hM), zeros)  # fool D -> call M "full"
                        loss = loss + lam_adv * l_adv
                    if lam_contrast > 0:  # CONTRASTIVE (InfoNCE): pull (hM_i,hF_i) together, push hM_i from other hF_j
                        hM_n = F.normalize(hM.float(), dim=-1)
                        hF_n = F.normalize(hF.float(), dim=-1).detach()       # positive target (no grad to teacher)
                        ci = int(s.get("cbank_idx", -1))                      # EXCLUDE this sample's own positive from negs
                        if 0 <= ci < cbank_all.shape[0]:
                            negs = torch.cat([cbank_all[:ci], cbank_all[ci + 1:]], 0)
                        else:
                            negs = torch.cat([cbank_all[:i], cbank_all[i + 1:]], 0)  # EXCLUDE self (its own full is the positive)
                        if negs.shape[0] > bank_size:                         # subsample negatives if the pool is large
                            negs = negs[torch.randperm(negs.shape[0], device=dev)[:bank_size]]
                        pos = hM_n @ hF_n.T / contrast_temp                   # (1,1) positive
                        neg = hM_n @ negs.T / contrast_temp                   # (1,K) other samples' full (negatives)
                        l_contrast = F.cross_entropy(torch.cat([pos, neg], 1), torch.zeros(1, dtype=torch.long, device=dev))
                        loss = loss + lam_contrast * l_contrast
                    if lam_align > 0:  # PURE positive alignment (NO negatives): just pull hM toward hF in cosine space
                        hM_n = F.normalize(hM.float(), dim=-1)
                        hF_n = F.normalize(hF.float(), dim=-1).detach()
                        l_align = (1.0 - (hM_n * hF_n).sum(-1)).mean()       # 1 - cos(hM, hF); no contrast/negatives
                        loss = loss + lam_align * l_align
                l_dist = torch.zeros((), device=dev)
                if lam_distill > 0:
                    Re = embed(s["R"])
                    sq = torch.cat([mq, qe, Re], 1)
                    Ps = mq.shape[1] + qe.shape[1]
                    slg = model(inputs_embeds=sq, use_cache=False).logits[0, Ps - 1:Ps - 1 + s["R"].shape[1]].float()
                    slp = F.log_softmax(slg / temp, -1).gather(-1, s["tk_idx"])
                    pt = F.softmax(s["tk_logit"] / temp, -1)
                    l_dist = (pt * (F.log_softmax(s["tk_logit"] / temp, -1) - slp)).sum(-1).mean() * (temp * temp)
                    loss = loss + lam_distill * l_dist
                l_rec, l_dev = torch.zeros((), device=dev), torch.zeros((), device=dev)
                if self.svc.decoder is not None:
                    # unconditional memory M0: reconstruct the context (L_uncond) + anchor Mq to M0 (min-dev)
                    enc0 = self._enc(rt, s["cid"], s["qid"], False)
                    l_rec = self.svc.reconstruct(s["cid"], enc0["memory"], out_w,
                                                 m_only=bool(getattr(cfg, "recon_m_only", False)))
                    if (encq["memory_raw"].shape == enc0["memory_raw"].shape
                            and not (self.chunk_keep and self.chunk_keep > 0)):  # routing may pick DIFFERENT chunks ⇒ dev ill-defined
                        l_dev = ((encq["memory_raw"] - enc0["memory_raw"]) ** 2).mean()
                    loss = loss + lam_rec * l_rec + lam_dev * l_dev
                (loss / accum).backward()              # grad accumulation => effective batch = accum
                lt = float(l_task.detach())
                ema = lt if ema is None else 0.9 * ema + 0.1 * lt
                if (step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % max(1, cfg.steps // 50) == 0:    # EMA-smoothed train loss + periodic held-out val loss
                    vl = _val_loss() if step % max(1, cfg.steps // 10) == 0 else float("nan")
                    extra = (f" adv={float(l_adv.detach()):.4f}" if lam_adv > 0 else "") + \
                            (f" ctr={float(l_contrast.detach()):.4f}" if lam_contrast > 0 else "") + \
                            (f" aln={float(l_align.detach()):.4f}" if lam_align > 0 else "")
                    logger.info(f"[gcm] step {step}/{cfg.steps} task_ema={ema:.4f} task={lt:.4f} "
                                f"rec={float(l_rec.detach()):.4f} dev={float(l_dev.detach()):.6f} "
                                f"lr={sched.get_last_lr()[0]:.2e}{extra}" + (f" val={vl:.4f}" if vl == vl else ""))
                    if _patience > 0 and vl == vl:   # v1.7.6: stop when held-out val stops improving (converge)
                        if vl < _best_val - 1e-4:
                            _best_val = vl; _no_improve = 0
                        else:
                            _no_improve += 1
                            if _no_improve >= _patience:
                                logger.info(f"[gcm] early-stop @ step {step}/{cfg.steps} (val plateau {_no_improve}x, best={_best_val:.4f})")
                                step = cfg.steps
                if step >= cfg.steps:
                    break
        for p in self.svc.parameters():
            p.requires_grad_(False)
