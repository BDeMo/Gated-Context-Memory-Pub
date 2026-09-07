"""Trainer for the GCM self-compressor (adapters only: memory tokens + nonlinear projection + read-LoRA).

Losses: gold-answer cross-entropy (forces M to carry the answer) + optional gold-forced context-distillation
(match the full-context teacher's first-/answer-token distribution). The base is frozen.

Data parallel: launch one process per GPU; each rank trains on its own shard with per-rank batch 1, and adapter
grads are all-reduced (averaged) every optimizer step, giving an effective batch of ``world_size * grad_accum``.
Only rank 0 logs. Logged to Weights & Biases (if a run is passed): loss components, lr, per-group grad norms, the
memory-vector scale vs the embedding manifold, and the do-no-harm gate signals + gate AUROC on the val set.
"""
from __future__ import annotations

import contextlib
import math
import os
import time
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .lora import set_lora_enabled
from .signals import auroc, first_token_signal


def _wlog(run, data: dict, step: int) -> None:
    if run is not None:
        run.log(data, step=step)


def _gnorm(params) -> float:
    sq = [(p.grad.detach().float() ** 2).sum() for p in params if p.grad is not None]
    return float(torch.sqrt(torch.stack(sq).sum())) if sq else 0.0


def _allreduce_grads(params, world_size: int) -> None:
    if world_size <= 1:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size


@torch.no_grad()
def _val_metrics(model, items) -> dict:
    """Compress-path loss/accuracy + do-no-harm gate signals (compress vs full) + gate AUROC on the val set."""
    embed = model._embed
    ce_sum, n = 0.0, 0
    mc, mf, cc, cf, ec, ef = [], [], [], [], [], []
    score_margin, score_conf, correct = [], [], []
    score_recrep, score_maxr = [], []                               # kvzip-style repeat-recon gate signals
    for cid, qid, gold in items:
        M = model.encode(cid, qid)                                  # read-LoRA off inside
        sc = first_token_signal(model, M, qid, True)               # compress path
        sf = first_token_signal(model, embed(cid), qid, False)     # full path
        try:
            sg = model.gate_signal(M, cid, qid)                    # label-free do-no-harm signals (incl. repeat-recon)
        except Exception:
            sg = {}
        set_lora_enabled(model.base, True)
        seq = torch.cat([M, embed(qid), embed(gold)], 1)
        P = M.shape[1] + qid.shape[1]
        lg = model.base(inputs_embeds=seq, use_cache=False).logits[0, P - 1:P - 1 + gold.shape[1]].float()
        ce_sum += float(F.cross_entropy(lg, gold[0])); n += 1
        ok = int(sc["argmax"] == int(gold[0, 0]))
        mc.append(sc["margin"]); mf.append(sf["margin"]); cc.append(sc["conf"]); cf.append(sf["conf"])
        ec.append(sc["entropy"]); ef.append(sf["entropy"])
        score_margin.append(sc["margin"]); score_conf.append(sc["conf"]); correct.append(ok)
        score_recrep.append(sg.get("neg_recon_repeat", 0.0)); score_maxr.append(sg.get("neg_maxr", 0.0))
    mean = lambda a: float(sum(a) / max(1, len(a)))
    return {"val/answer_ce": ce_sum / max(1, n),                    # compressed-path gold CE on held-out val
            "val/compress_first_token_acc": mean(correct),         # first answer token correct from M
            "gate/margin_compress": mean(mc), "gate/margin_full": mean(mf),   # top1-top2 prob margin, each path
            "gate/conf_compress": mean(cc), "gate/conf_full": mean(cf),       # top1 prob, each path
            "gate/entropy_compress": mean(ec), "gate/entropy_full": mean(ef), # predictive entropy, each path
            "gate/margin_gap_full_minus_compress": mean(mf) - mean(mc),       # how much full beats compressed
            "gate/auroc_margin": auroc(score_margin, correct),     # margin predicts compressed-correct (key metric)
            "gate/auroc_conf": auroc(score_conf, correct),
            "gate/auroc_recon_repeat": auroc(score_recrep, correct),  # kvzip repeat-recon predicts compress-correct
            "gate/auroc_maxr": auroc(score_maxr, correct)}            # "no dropped needle" predicts compress-correct


def train_compressor(model, train_items: Sequence[tuple], eval_items: Sequence[tuple] | None = None, *,
                     epochs: int = 3, lr: float = 3e-4, grad_accum: int = 1, warmup_frac: float = 0.05,
                     patience: int = 0, max_steps: int = 0, batch_size: int = 1, encode_mode: str = "parallel", lam_distill: float = 0.5,
                     lam_recon: float = 0.0, recon_m_only: bool = False, lam_repeat: float = 0.0, lam_kl: float = 0.0, lam_dev: float = 0.0,
                     lam_adv: float = 0.0, adv_layer: int = 18, adv_mode: str = "alternating",
                     lam_contrast: float = 0.0, contrast_temp: float = 0.2, joint_task: bool = True,
                     distill_topk: int = 64, log_every: int = 5, val_every: int = 0, wandb_run=None,
                     rank: int = 0, world_size: int = 1, shuffle: bool = False,
                     domains: Sequence | None = None, curriculum: bool = False, domain_warmup_frac: float = 0.1,
                     full_maxctx: int = 0, enc_varlen: bool = False, seed: int = 1234) -> None:
    """Batched trainer (batch_size items per micro-step). batch_size=1 reduces to the per-item path.
    train_items: (ctx_ids, query_ids, gold_ids) per item on device. `steps` counts micro-batch steps."""
    if adv_mode not in ("alternating", "simultaneous"):
        raise ValueError(f"unknown adversarial update mode: {adv_mode}")
    base, embed = model.base, model._embed
    groups = {"mem": [model.compressor.mem], "proj": list(model.compressor.proj.parameters()), "lora": model.lora_params}
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr)
    disc = d_opt = None                                            # adversarial losslessness discriminator (if lam_adv>0)
    if lam_adv > 0:
        dh = model.compressor.d; ddev = model.base.get_input_embeddings().weight.device
        disc = torch.nn.Sequential(torch.nn.Linear(dh, dh // 4), torch.nn.GELU(),
                                   torch.nn.Linear(dh // 4, 1)).to(ddev, torch.float32)
        d_opt = torch.optim.AdamW(disc.parameters(), lr=2e-4)
    hF_bank: list = []                                             # rolling full-induced-hidden bank (InfoNCE negatives, bs=1)
    n = len(train_items); B = max(1, int(batch_size))
    # over-window: encode() sees the full (≤ENC) ctx; the single-forward teacher/recon truncate to the window.
    _ft = (lambda c: c[:, :full_maxctx]) if full_maxctx and full_maxctx > 0 else (lambda c: c)
    steps_per_epoch = max(1, (n + B - 1) // B)
    steps = max_steps if max_steps > 0 else epochs * steps_per_epoch
    opt_total = max(1, (steps + grad_accum - 1) // grad_accum)
    warm = max(1, int(warmup_frac * opt_total))
    _eos_t = getattr(base.config, "eos_token_id", 0) or 0
    if isinstance(_eos_t, (list, tuple)): _eos_t = _eos_t[0] if _eos_t else 0
    pad_id = int(_eos_t)
    train_offload = os.environ.get("GCM_TRAIN_OFFLOAD", "0").lower() in ("1", "true", "on")
    empty_cache_every = int(os.environ.get("GCM_EMPTY_CACHE_EVERY", "0"))

    def _saved_tensor_context():
        if train_offload and torch.is_grad_enabled():
            return torch.autograd.graph.save_on_cpu(pin_memory=True)
        return contextlib.nullcontext()

    def lr_at(o: int) -> float:
        if o < warm:
            return (o + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (o - warm) / max(1, opt_total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    # ---- shuffle / curriculum permutation (rebuilt each epoch) ----
    import random as _rnd
    _crng = _rnd.Random(int(seed) + rank)
    _dom = list(domains) if (curriculum and domains is not None and len(domains) == n) else None

    def _build_perm():
        """Return (perm, blocks). blocks=[(domain, start_pos, len)] over perm positions for per-domain LR warmup.
        curriculum: order domains EASY->HARD (mean ctx length), SHUFFLE within each domain (both, as requested)."""
        if _dom is not None:
            from collections import defaultdict
            g = defaultdict(list)
            for i, d in enumerate(_dom):
                g[d].append(i)
            order = sorted(g, key=lambda d: sum(int(train_items[i][0].shape[1]) for i in g[d]) / max(1, len(g[d])))
            perm, blocks = [], []
            for d in order:
                idx = g[d][:]; _crng.shuffle(idx)
                blocks.append((d, len(perm), len(idx))); perm += idx
            return perm, blocks
        perm = list(range(n))
        if shuffle: _crng.shuffle(perm)
        return perm, [("all", 0, n)]

    _perm, _blocks = _build_perm()

    def _lr_mult(step: int) -> float:
        """LR multiplier. curriculum => warmup-then-cosine WITHIN each domain block (re-warm at every domain).
        else => global warmup+cosine over opt_total."""
        if _dom is not None:
            pos = (step * B) % n
            for (d, s, L) in _blocks:
                if s <= pos < s + L:
                    f = (pos - s) / max(1, L)                                  # fraction through this domain
                    if f < domain_warmup_frac:
                        return max(1e-3, f / domain_warmup_frac)               # re-warmup entering the domain
                    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (f - domain_warmup_frac) / max(1e-6, 1 - domain_warmup_frac))))
            return 1.0
        o = (step + 1) // grad_accum
        return lr_at(o)

    def gold_logits_batched(prefixes, qids, golds, lora_on):
        """Left-pad [prefix_i ; query_i ; gold_i], one batched forward, return per-item gold-position logits."""
        set_lora_enabled(base, lora_on)
        dev = prefixes[0].device; d = prefixes[0].shape[-1]
        seqs, spans, L = [], [], 0
        for pf, q, g in zip(prefixes, qids, golds):
            s = torch.cat([pf, embed(q), embed(g)], 1)
            seqs.append(s); spans.append((pf.shape[1] + q.shape[1], g.shape[1])); L = max(L, int(s.shape[1]))
        L = (L + 3) // 4 * 4  # fused SDPA backward requires attention-bias row stride divisible by four
        pad_emb = embed(torch.tensor([[pad_id]], device=dev))
        attn = torch.zeros(len(seqs), L, dtype=torch.long, device=dev); rows, offs = [], []
        for i, s in enumerate(seqs):
            nn = L - s.shape[1]
            rows.append(torch.cat([pad_emb.to(s.dtype).expand(1, nn, d), s], 1) if nn > 0 else s)
            attn[i, nn:] = 1; offs.append(nn)
        keep = max(Lg for _, Lg in spans) + 1
        with _saved_tensor_context():
            logits = base(
                inputs_embeds=torch.cat(rows, 0),
                attention_mask=attn,
                use_cache=False,
                logits_to_keep=keep,
            ).logits
        return [
            logits[i, keep - Lg - 1: keep - 1].float()
            for i, (_, Lg) in enumerate(spans)
        ]

    teach: dict = {}
    def teacher_batch(idxs, cids, qids, golds):
        need = [j for j, ix in enumerate(idxs) if ix not in teach]
        if need:
            with torch.no_grad():
                tl = gold_logits_batched([embed(_ft(cids[j])) for j in need], [qids[j] for j in need],
                                         [golds[j] for j in need], False)
                for k, j in enumerate(need):
                    teach[idxs[j]] = tl[k].topk(distill_topk, dim=-1)
        return [teach[ix] for ix in idxs]

    val_every = val_every or max(1, min(steps // 10, 250))   # cap cadence so early-stop bounds big-n_train runs
    opt.zero_grad(); t0 = time.time(); best_ce = float("inf"); bad_checks = 0
    for step in range(steps):
        if step > 0 and (step * B) % n < B and (shuffle or _dom is not None):   # epoch boundary -> rebuild perm
            _perm, _blocks = _build_perm()
        idxs = [_perm[(step * B + j) % n] for j in range(B)]
        items = [train_items[ix] for ix in idxs]
        cids = [x[0] for x in items]; qids = [x[1] for x in items]; golds = [x[2] for x in items]
        if enc_varlen and model.compressor.chunk_size > 0:        # ENCODE-LENGTH ROBUSTNESS: vary #chunks per step so the
            cs = int(model.compressor.chunk_size)                 # encoder/reader generalize to any context length (fixes
            cids = [c[:, :min(c.shape[1], _crng.randint(1, max(1, (c.shape[1] + cs - 1) // cs)) * cs)] for c in cids]  # the OOD-at-other-lengths fragility / over-window degradation)
        set_lora_enabled(base, False)
        if encode_mode == "ar":                                        # autoregressive M (single-item per encode)
            Ms = [model.compressor.encode_ar(c, q) for c, q in zip(cids, qids)]            # list of [1,K,d]
        elif model.compressor.chunk_size > 0:                          # adaptive: M length varies per item (ragged)
            Ms = [model.compressor.encode(c, q)["memory"] for c, q in zip(cids, qids)]
        else:
            Mb = model.compressor.encode_batch(cids, qids)             # [B,K,d], grad -> mem+proj
            Ms = [Mb[i:i + 1] for i in range(B)]
        # deviation anchor needs the UNCONDITIONAL M0 (computed here while read-LoRA is still off).
        # STOP-GRAD on M0: it is the reference target (anchor Mq->M0), so no grad graph -> halves encode
        # memory and lets long-ctx (chunked 32k) dev cells fit on a single GPU.
        if lam_dev > 0:
            with torch.no_grad():
                M0s = [model.compressor.encode(c, q, conditional=False)["memory"].detach() for c, q in zip(cids, qids)]
        else:
            M0s = None
        Mdev = Ms[0].device
        # ===== TASK + DISTILL. GCM_JOINT (default, the v1.7.5 recipe): the answer-CE trains the ENCODER too, so M
        # learns to CARRY the answer (not just be context-faithful). GCM_JOINT off: strict phase split -> task trains
        # the read-LoRA only (M detached) and the encoder is shaped by the phase-1 losses alone.
        prefix = Ms if joint_task else [m.detach() for m in Ms]
        # Compute the frozen teacher first. The student forward below is gradient-checkpointed on long
        # Qwen3.5 runs, so read-LoRA must remain ON from that forward through its immediate backward.
        # Running the LoRA-OFF teacher between those two operations changes checkpoint recomputation.
        tch = teacher_batch(idxs, cids, qids, golds) if lam_distill > 0 else None
        st = gold_logits_batched(prefix, qids, golds, True)           # student (read-LoRA on)
        l_task = sum(F.cross_entropy(st[i], golds[i][0]) for i in range(B)) / B
        l_dist = torch.zeros((), device=Mdev)
        if lam_distill > 0:
            parts = []
            for i in range(B):
                tv, ti = tch[i]; slp = F.log_softmax(st[i], -1).gather(-1, ti)
                parts.append((F.softmax(tv, -1) * (F.log_softmax(tv, -1) - slp)).sum(-1).mean())
            l_dist = sum(parts) / B
        l_kl = getattr(model.compressor, "last_kl", None)             # VAE prior term (set by compressor when enabled)
        l_dev = torch.zeros((), device=Mdev)
        if lam_dev > 0:                                                # anchor Mq -> M0 (query shouldn't distort the memory)
            l_dev = sum(((Ms[i] - M0s[i].to(Ms[i].dtype)) ** 2).mean() for i in range(B)) / B
        # kl/dev share M's encode graph; in JOINT mode task also consumes it, so fold them into ONE backward
        # (separate backwards would double-traverse M's graph -> "backward a second time").
        main = l_task + lam_distill * l_dist
        if joint_task:
            if lam_kl > 0 and l_kl is not None: main = main + lam_kl * l_kl
            if lam_dev > 0: main = main + lam_dev * l_dev
            (main / grad_accum).backward()                            # -> encoder (task+kl+dev) + read-LoRA (task)
        else:
            (main / grad_accum).backward()                           # -> read-LoRA only (M detached, base frozen)
            p1 = torch.zeros((), device=Mdev)
            if lam_kl > 0 and l_kl is not None: p1 = p1 + lam_kl * l_kl
            if lam_dev > 0: p1 = p1 + lam_dev * l_dev
            if p1.requires_grad:
                (p1 / grad_accum).backward()                          # -> encoder only (read-LoRA was off in encode)
        l_rec = torch.zeros((), device=Mdev)
        if lam_recon > 0:                                             # reconstruct ctx from M -> encoder/decoder only.
            for p in model.lora_params: p.requires_grad_(False)       # SEPARATE backward (else OOM at bs=1); read-LoRA frozen
            with _saved_tensor_context():
                l_rec = sum(model.reconstruct(_ft(cids[i]), qids[i], m_only=recon_m_only) for i in range(B)) / B
            (lam_recon * l_rec / grad_accum).backward()
            for p in model.lora_params: p.requires_grad_(True)

        l_rep = torch.zeros((), device=Mdev)
        if lam_repeat > 0:                                            # kvzip-style repeat-prompt reconstruction (native repeat prior);
            for p in model.lora_params: p.requires_grad_(False)       # composes WITH slot-recon + VAE. Grad -> encoder (mem+proj).
            l_rep = sum(model.recon_repeat(_ft(cids[i]), qids[i])[0] for i in range(B)) / B
            (lam_repeat * l_rep / grad_accum).backward()
            for p in model.lora_params: p.requires_grad_(True)

        l_adv = torch.zeros((), device=Mdev); l_disc = torch.zeros((), device=Mdev); l_con = torch.zeros((), device=Mdev)
        if lam_adv > 0 or lam_contrast > 0:                            # PHASE-1 alignment: M-induced layer-l hidden should
            advs, discs, cons = [], [], []                             # match full-ctx-induced. Both update the ENCODER only.
            one = torch.ones(1, 1, device=Mdev); zero = torch.zeros(1, 1, device=Mdev)
            for i in range(B):
                set_lora_enabled(base, False)
                Ma = model.compressor.encode(cids[i], qids[i])["memory"]            # re-encode (grad -> encoder)
                # Keep the reader state identical on both sides of D.  Otherwise
                # D can classify "read-LoRA on" versus "read-LoRA off" instead
                # of memory-induced versus full-context-induced hidden states.
                set_lora_enabled(base, False)
                _oM = base.model(inputs_embeds=torch.cat([Ma, embed(qids[i])], 1), output_hidden_states=True, use_cache=False)
                _hsM = _oM.hidden_states; _al = min(adv_layer, len(_hsM) - 1)  # GDN/linear-attn may return fewer hidden states
                hM = _hsM[_al][0, Ma.shape[1]:].mean(0, keepdim=True).float()
                with torch.no_grad():
                    set_lora_enabled(base, False)                                   # full path = exact base
                    _oF = base.model(inputs_embeds=torch.cat([embed(cids[i]), embed(qids[i])], 1), output_hidden_states=True, use_cache=False)
                    _hsF = _oF.hidden_states; _alF = min(adv_layer, len(_hsF) - 1)
                    hF = _hsF[_alF][0, cids[i].shape[1]:].mean(0, keepdim=True).float()
                for p in model.lora_params: p.requires_grad_(False)                 # ENCODER-only updates: freeze read-LoRA
                # adv + contrast SHARE hM; accumulate both into one scalar and do a SINGLE backward through hM->Ma
                # (separate backwards would free hM's graph and crash the second one).
                l_enc = torch.zeros((), device=Mdev)
                simultaneous_disc_step = False
                if lam_adv > 0:
                    d_opt.zero_grad()                                              # train D: M-induced=1, full-induced=0
                    d_loss = F.binary_cross_entropy_with_logits(disc(hM.detach()), one) + \
                             F.binary_cross_entropy_with_logits(disc(hF), zero)
                    d_loss.backward()                                              # hM.detach -> no Ma graph touched
                    if adv_mode == "alternating":
                        torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                        d_opt.step()                                                # G sees the freshly updated D
                    else:
                        simultaneous_disc_step = True                               # step D after G gradients use the old D
                    for p in disc.parameters(): p.requires_grad_(False)            # freeze D for the fool-D step
                    l_a = F.binary_cross_entropy_with_logits(disc(hM), zero)        # make M-induced look "full"
                    l_enc = l_enc + lam_adv * l_a
                    advs.append(float(l_a.detach()))
                    discs.append(float(d_loss.detach()))
                if lam_contrast > 0:                                               # InfoNCE: pull hM->its hF, push from bank
                    hMn = F.normalize(hM, dim=-1); hFn = F.normalize(hF, dim=-1)
                    pos = (hMn * hFn).sum(-1, keepdim=True) / contrast_temp         # [1,1] positive (own full hidden)
                    if hF_bank:
                        neg = F.normalize(torch.cat(hF_bank, 0), dim=-1)
                        logits_c = torch.cat([pos, (hMn @ neg.t()) / contrast_temp], 1)  # [1, 1+N]
                    else:
                        logits_c = pos
                    l_c = F.cross_entropy(logits_c, torch.zeros(1, dtype=torch.long, device=Mdev))  # positive at index 0
                    l_enc = l_enc + lam_contrast * l_c
                    cons.append(float(l_c.detach()))
                    hF_bank.append(hF.detach())
                    if len(hF_bank) > 256: hF_bank.pop(0)
                if l_enc.requires_grad:
                    (l_enc / grad_accum / B).backward()                            # SINGLE backward -> Ma (encoder) only
                if simultaneous_disc_step:
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
                    d_opt.step()
                for p in model.lora_params: p.requires_grad_(True)
                if lam_adv > 0:
                    for p in disc.parameters(): p.requires_grad_(True)
            if advs: l_adv = torch.tensor(sum(advs) / len(advs))
            if discs: l_disc = torch.tensor(sum(discs) / len(discs))
            if cons: l_con = torch.tensor(sum(cons) / len(cons))

        if (step + 1) % grad_accum == 0:
            _allreduce_grads(params, world_size)
            gtot = _gnorm(params)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            _curlr = lr * _lr_mult(step)                       # curriculum: re-warm per domain; else global warmup+cosine
            for _g in opt.param_groups: _g["lr"] = _curlr
            opt.step(); opt.zero_grad()
            if empty_cache_every > 0 and (step + 1) % empty_cache_every == 0:
                torch.cuda.empty_cache()
        else:
            gtot = None

        if rank == 0 and step % log_every == 0:
            with torch.no_grad():
                mnorm = float(Ms[0].detach().float().norm(dim=-1).mean())
            escale = float(model.compressor.embed_scale)
            _kl_v = float(l_kl) if (lam_kl > 0 and l_kl is not None) else 0.0
            log = {"loss/total": (l_task.item() + lam_distill * float(l_dist) + lam_kl * _kl_v + lam_dev * float(l_dev)
                                  + lam_recon * float(l_rec) + lam_repeat * float(l_rep) + lam_adv * float(l_adv) + lam_contrast * float(l_con)),
                   "loss/answer_ce": l_task.item(), "loss/distill_kl": float(l_dist),
                   "loss/recon_ce": float(l_rec), "loss/repeat_ce": float(l_rep), "loss/kl": float(l_kl) if l_kl is not None else 0.0, "loss/dev": float(l_dev),
                   "loss/contrast": float(l_con),
                   "loss/adv": float(l_adv),
                   "loss/discriminator": float(l_disc),
                   "opt/lr": opt.param_groups[0]["lr"], "perf/steps_per_sec": (step + 1) / max(1e-6, time.time() - t0),
                   "perf/items_per_sec": (step + 1) * B / max(1e-6, time.time() - t0),
                   "mem/M_norm": mnorm, "mem/embed_scale": escale, "mem/M_scale_ratio": mnorm / max(1e-6, escale),
                   "train/epoch": step / max(1, steps_per_epoch), "train/batch_size": B}
            if gtot is not None:
                log.update({"grad/total": gtot, "grad/mem_tokens": _gnorm(groups["mem"]),
                            "grad/proj_mlp": _gnorm(groups["proj"]), "grad/read_lora": _gnorm(groups["lora"])})
            _wlog(wandb_run, log, step)
            if step % 50 == 0:
                print(
                    f"[gcm] step {step}/{steps} loss={log['loss/total']:.4f} "
                    f"items_s={log['perf/items_per_sec']:.3f}",
                    flush=True,
                )

        if eval_items and step > 0 and step % val_every == 0:
            if rank == 0:
                vm = _val_metrics(model, eval_items); _wlog(wandb_run, vm, step)
                cur = vm.get("val/answer_ce", float("inf"))
                if cur < best_ce - 1e-3:
                    best_ce = cur; bad_checks = 0
                else:
                    bad_checks += 1
            if patience > 0:
                stop_t = torch.zeros(1, device=Mdev)
                if rank == 0 and bad_checks >= patience:
                    stop_t[0] = 1.0
                if world_size > 1:
                    dist.broadcast(stop_t, src=0)
                if stop_t.item() > 0:
                    if rank == 0 and wandb_run is not None:
                        wandb_run.summary["train/early_stopped_step"] = step
                    break

    if eval_items and rank == 0:
        _wlog(wandb_run, _val_metrics(model, eval_items), steps)
