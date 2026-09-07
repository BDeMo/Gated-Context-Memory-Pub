"""Nail the DEFAULT training recipe for the self-compressor: ONE experiment across all GPUs (data-parallel,
effective batch = world_size * grad_accum), trained on a MIX of benchmarks, with the full recipe in the WandB
config and all method signals + monitoring metrics on the dashboard.

Launch:  torchrun --nproc_per_node=4 run_recipe.py <model_path>
Recipe knobs are env vars (GCM_K, GCM_DEPTH, GCM_EPOCHS, GCM_LR, GCM_DISTILL, GCM_ACCUM, GCM_NTRAIN, GCM_NVAL,
GCM_PROJ, GCM_LORA, GCM_BENCHES) so the recipe can be re-tuned without code changes.
Needs the research data loaders (mem_embedding) on PYTHONPATH; the gcm lib stays standalone."""
import os
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from gcm import GCMModel, enable_torch_linear_attention, train_compressor
enable_torch_linear_attention()
from gcm.lora import checkpoint_lora_context, set_lora_enabled  # noqa: E402
from gcm.signals import auroc                                  # noqa: E402
from mem_embedding.gcm.runtime import Runtime                  # noqa: E402
from mem_embedding.gcm.data import load_items, score_gen       # noqa: E402


def env(k, d):
    return os.environ.get(k, d)


def main():
    rank = int(env("RANK", "0")); world = int(env("WORLD_SIZE", "1")); local = int(env("LOCAL_RANK", "0"))
    import datetime as _dt
    dist.init_process_group("nccl", timeout=_dt.timedelta(hours=24), device_id=torch.device(f"cuda:{local}")) if world > 1 else None
    torch.cuda.set_device(local); dev = f"cuda:{local}"
    SEED = int(env("GCM_SEED", "1234"))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    MODEL = sys.argv[1]; mname = MODEL.rstrip("/").split("/")[-1]
    K = int(env("GCM_K", "16")); PROJ = int(env("GCM_PROJ", "2")); LORA = int(env("GCM_LORA", "32"))
    EPOCHS = int(env("GCM_EPOCHS", "3")); LR = float(env("GCM_LR", "3e-4")); DISTILL = float(env("GCM_DISTILL", "0.5"))
    ACCUM = int(env("GCM_ACCUM", "2")); NTR = int(env("GCM_NTRAIN", "128")); NVAL = int(env("GCM_NVAL", "100000"))
    DEFB = "bfcl_live_multiple,toolace,squad_v2,narrativeqa"
    TRAIN_BENCHES = env("GCM_TRAIN", env("GCM_BENCHES", DEFB)).split(",")   # what we fit M on
    EVAL_BENCHES = env("GCM_EVAL", env("GCM_BENCHES", DEFB)).split(",")     # eval-on-all -> in/cross-task/cross-domain
    NTR_BY_BENCH = {
        key.strip(): int(value)
        for entry in env("GCM_NTRAIN_BY_BENCH", "").split(",")
        if entry.strip()
        for key, value in [entry.split("=", 1)]
    }
    TAG = env("GCM_TAG", "mix")                                            # wandb run-name prefix (the cell id)
    depth_cfg = env("GCM_DEPTH", "half")

    MAXCTX = int(env("GCM_MAXCTX", "1024"))                       # single-forward window (full/trunc baseline + teacher)
    ENC = max(MAXCTX, int(env("GCM_ENC_MAXCTX", str(MAXCTX))))    # v1.8.0 over-window: compress encodes up to ENC via recurrent chunking
    rt = Runtime(MODEL, device=dev, n_memory=K, max_ctx_tokens=ENC, max_new_tokens=int(env("GCM_GOLD_MAX", "32")))
    nlayers = int(rt.model.config.num_hidden_layers)
    depth = nlayers if depth_cfg == "full" else (nlayers // 2 if depth_cfg == "half" else int(depth_cfg))
    _nm = env("GCM_NORM", "1").lower()                          # 0/off, 1/hard, learn, manifold(=convex-hull)
    NORM_MODE = {"0":"off","off":"off","1":"hard","hard":"hard","learn":"learn","manifold":"manifold"}.get(_nm, "hard")
    NORM = NORM_MODE != "off"
    MANTEMP = float(env("GCM_MANIFOLD_TEMP", "1.0"))
    VAE = env("GCM_VAE", "0").lower() in ("1", "true", "on")   # VAE memory (proj->mu,logvar; reparam; KL prior)
    CHUNK = int(env("GCM_CHUNK", "0"))                          # >0 => adaptive (length-scaled) chunked memory
    XCHUNK = int(env("GCM_XCHUNK", "0"))                        # >0 => cross-chunk refine (mix chunk memories via N frozen base layers)
    RECUR = env("GCM_RECUR", "0").lower() in ("1", "true", "on")  # recurrent encode: carry prior-chunk summaries (AutoCompressor-style)
    STATE_CAP = int(env("GCM_STATE_CAP", "0"))                   # 0 => legacy S*K concat; >0 => bounded recurrent state/output
    ENC_LORA = int(env("GCM_ENC_LORA", "0"))                    # >0 => train the encoder via an encode-phase LoRA
    DEC_LAYERS = int(env("GCM_DEC_LAYERS", "0"))                 # >0 => separate trainable N-layer reconstruction decoder
    model = GCMModel(rt.model, rt.tok, n_memory=K, depth=depth, proj_mult=PROJ, lora_rank=LORA, normalize=NORM,
                     norm_mode=NORM_MODE, manifold_temp=MANTEMP, vae=VAE, chunk_size=CHUNK, enc_lora_rank=ENC_LORA,
                     dec_layers=DEC_LAYERS, xchunk=XCHUNK, recur=RECUR, state_cap=STATE_CAP).to(dev)
    if world > 1:                                              # identical adapter init across ranks
        for p in model.trainable_parameters():
            dist.broadcast(p.data, src=0)

    def prep(it):
        g = rt.tok(" " + str(it.gold).strip() + "\n", add_special_tokens=False).input_ids[:int(env("GCM_GOLD_MAX", "32"))]  # v1.7.5-FIX: teach termination
        return (rt.ctx_ids(it), rt.query_ids(it), torch.tensor([g], device=dev)) if g else None

    # train on TRAIN_BENCHES (mix or a single anchor), then give each rank an equal disjoint shard
    NCHUNKS = int(env("GCM_NCHUNKS", "8"))   # haystack size for synthetic long-ctx benches (~182 tok/chunk for NIAH) -> length sweep
    raw_pool = []
    for b in TRAIN_BENCHES:
        raw_pool += [
            (b, it)
            for it in load_items(b, NTR_BY_BENCH.get(b, NTR), NCHUNKS, 0, "train")
        ]
    # Tokenized contexts dominate host memory on full-split runs. Preparing the
    # complete pool independently on every distributed rank multiplies that
    # footprint by world_size and can trigger a node-level SIGKILL. Assign each
    # rank its deterministic strided shard before tokenization instead.
    per = len(raw_pool) // max(1, world)
    rank_rows = [raw_pool[rank + world * j] for j in range(per)]
    pool = [(x, b, it) for b, it in rank_rows if (x := prep(it))]
    # eval on EVAL_BENCHES (all) so per-bench cells give in-task / cross-task / cross-domain
    val_pool = [(x, b, it) for b in EVAL_BENCHES for it in load_items(b, NVAL, NCHUNKS, 1, "validation") if (x := prep(it))]
    # Belt-and-suspenders leakage guard. A native BFCL split contains one exact duplicate; remove any train item
    # whose tokenized (context, query, gold) tuple occurs in validation before sharding or training.
    def _item_sig(x):
        import hashlib
        h = hashlib.sha256()
        for tensor in x:
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.digest()
    if env("GCM_LEAKAGE_GUARD", "1").lower() not in ("0", "off", "false"):
        val_signatures = {_item_sig(x) for x, _, _ in val_pool}
        pool = [row for row in pool if _item_sig(row[0]) not in val_signatures]
    if world > 1:
        local_n = torch.tensor([len(pool)], device=dev, dtype=torch.long)
        dist.all_reduce(local_n, op=dist.ReduceOp.MIN)
        pool = pool[:int(local_n.item())]
    shard = [row[0] for row in pool]
    shard_domains = [row[1] for row in pool]   # per-item bench label (for curriculum)
    val_loss_items = [x for (x, _, _) in val_pool]
    val_small = [(x[0][:, :MAXCTX], x[1], x[2]) for x in val_loss_items[::max(1, len(val_loss_items) // 24)][:24]]  # in-loop val (truncated to window; over-window only matters for compress eval)

    run = None
    if rank == 0:
        import wandb
        regime = "mix" if len(TRAIN_BENCHES) > 1 else f"single:{TRAIN_BENCHES[0]}"
        run = wandb.init(entity=os.environ.get("WANDB_ENTITY"), project=env("WANDB_PROJECT", "gcm"),
                         name=f"{TAG}-{mname}-{world}gpu",
                         config=dict(design="self-compressor (merged, no duplicated weights)",
                                     M="nonlinear-MLP + embedding-manifold normalized", cell=TAG,
                                     model=mname, K=K, depth=f"{depth_cfg}={depth}/{nlayers}", proj_mult=PROJ,
                                     read_lora_rank=LORA, epochs=EPOCHS, lr=LR, optimizer="AdamW",
                                     lr_sched="cosine+warmup", loss="answer_ce + distill_weight*distill_kl",
                                     distill_weight=DISTILL, distill_topk=64,
                                     seed=SEED, chunk=CHUNK, recur=RECUR, state_cap=STATE_CAP,
                                     world_size=world, per_gpu_batch=int(env("GCM_BATCH","1")), grad_accum=ACCUM, effective_batch=world * ACCUM,
                                     n_train=len(raw_pool), n_train_per_rank=len(shard), n_val=len(val_pool),
                                     train_benches=",".join(TRAIN_BENCHES), eval_benches=",".join(EVAL_BENCHES),
                                     regime=regime))
        run.summary["trainable_params_M"] = sum(p.numel() for p in model.trainable_parameters()) / 1e6

    MAXSTEPS = int(env("GCM_MAX_STEPS", "600")); PAT = int(env("GCM_PATIENCE", "0")); BATCH = int(env("GCM_BATCH", "1")); ENCODE = env("GCM_ENCODE", "parallel")
    if ENCODE == "ar": BATCH = 1
    RECON = float(env("GCM_RECON", "0")); RECON_MONLY = env("GCM_RECON_MONLY", "0").lower() in ("1", "true", "on")
    REPEAT = float(env("GCM_REPEAT", "0"))                        # kvzip repeat-prompt reconstruction loss (composes with slot-recon+VAE)
    KL = float(env("GCM_KL", "0")); DEV = float(env("GCM_DEV", "0"))
    ADV = float(env("GCM_ADV", "0")); ADV_LAYER = int(env("GCM_ADV_LAYER", "18"))
    ADV_MODE = env("GCM_ADV_MODE", "alternating").lower()
    CONTRAST = float(env("GCM_CONTRAST", "0")); CONTRAST_TEMP = float(env("GCM_CONTRAST_TEMP", "0.2"))
    JOINT = env("GCM_JOINT", "1").lower() not in ("0", "off", "false")   # v1.7.5 recipe: answer-CE trains the encoder too
    LOAD_ADAPTER = env("GCM_LOAD_ADAPTER", "")
    if LOAD_ADAPTER:
        model.load_adapters(LOAD_ADAPTER, map_location=dev)
    else:
        # Runtime loads the frozen base in eval mode. HF gradient checkpointing is active only when
        # ``module.training`` is true, so merely calling gradient_checkpointing_enable() is not enough.
        # Put the complete wrapper in train mode while fitting adapters, then restore eval mode below.
        if env("GCM_GRAD_CKPT", "0").lower() not in ("0", "off", "false"):
            try:
                model.base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": False,
                        "context_fn": checkpoint_lora_context(model.base),
                    }
                )
            except TypeError:
                model.base.gradient_checkpointing_enable()
        model.train()
        train_compressor(model, shard, eval_items=val_small, epochs=EPOCHS, lr=LR, grad_accum=ACCUM,
                         lam_distill=DISTILL, lam_recon=RECON, recon_m_only=RECON_MONLY, lam_repeat=REPEAT, lam_kl=KL, lam_dev=DEV,
                         lam_adv=ADV, adv_layer=ADV_LAYER, adv_mode=ADV_MODE,
                         lam_contrast=CONTRAST, contrast_temp=CONTRAST_TEMP, joint_task=JOINT,
                         max_steps=MAXSTEPS, patience=PAT, batch_size=BATCH, encode_mode=ENCODE,
                         wandb_run=run, rank=rank, world_size=world,
                         shuffle=env("GCM_SHUFFLE", "1").lower() not in ("0", "off", "false"),
                         domains=shard_domains,
                         curriculum=env("GCM_CURRICULUM", "0").lower() not in ("0", "off", "false"),
                         domain_warmup_frac=float(env("GCM_DOMAIN_WARMUP", "0.1")),
                         full_maxctx=(MAXCTX if ENC > MAXCTX else 0),   # over-window: truncate teacher/recon to window, encode sees ENC
                         enc_varlen=env("GCM_ENC_VARLEN", "0").lower() not in ("0", "off", "false"),
                         seed=SEED)   # robustness to encode length

    model.eval()
    if world > 1:
        dist.barrier()
    torch.cuda.empty_cache()  # free training mem before eval
    if rank == 0:
        outdir = str(
            Path(env("GCM_OUT", Path(__file__).resolve().parents[1] / "out"))
            / mname
        )
        os.makedirs(outdir, exist_ok=True)
        # Persist the trained adapter before the long full-split evaluation so a scorer/OOM failure
        # never discards the expensive training phase.
        model.save_adapters(f"{outdir}/{TAG}_adapters.pt")
        torch.set_grad_enabled(False)        # eval is inference-only; avoids autograd-graph OOM over full split
        from mem_embedding.gcm.data import MC_BENCHES, options_for
        import string
        per_bench = {}
        records = []                                             # paper-grade per-item dump; gate calibration never uses test aggregates
        def _document_id(item):
            import hashlib
            text = "\n".join(str(value) for value in (getattr(item, "chunks", []) or []))
            return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        def _source_token_count(item):
            text = "\n".join(str(value) for value in (getattr(item, "chunks", []) or []))
            return len(rt.tok(text, add_special_tokens=False).input_ids)
        SIGNALS = ["margin", "conf", "neg_entropy", "neg_recon", "dlogit", "targ",   # +targ = TARG base-uncertainty baseline
                   "neg_recon_repeat", "neg_maxr", "neg_recon_repeat_rand"]   # kvzip repeat-recon fidelity + random-M leak control
        sig_vals = {s: [] for s in SIGNALS}; lab = []; fl = []; nl = []      # signals, compress-correct, full-correct, no_ctx-correct
        noconf = []                                                          # base no-ctx confidence (for 3-way: skip ctx when base already sure)
        GEN_MAX = int(env("GCM_GEN_MAX", "64")); GEN_BS = int(env("GCM_GEN_BS", "16"))
        from collections import defaultdict
        by_bench = defaultdict(list)
        for (x, b, it) in val_pool:                          # FULL split per bench (no subsample)
            by_bench[b].append((x, it))
        for b, rows in by_bench.items():
            d = per_bench.setdefault(b, {"no_ctx": [], "full": [], "compress": []})
            is_mc = b in MC_BENCHES                           # MC benches are scored by loglik-over-options, NOT generation
            for s0 in range(0, len(rows), GEN_BS):            # stream batches to bound GPU memory
                chunk = rows[s0:s0 + GEN_BS]
                qids = [xx[1] for (xx, it) in chunk]; cids = [xx[0] for (xx, it) in chunk]; its = [it for (xx, it) in chunk]
                try:
                    Ms = [model.encode(c, q) for c, q in zip(cids, qids)]
                except Exception as e:                        # robustness: skip a bad batch, keep the cell alive
                    print(f"[eval] encode failed ({b}): {repr(e)[:120]}", flush=True); continue
                if is_mc:
                    for j, it in enumerate(its):
                        try:
                            opts = options_for(it)
                            if not opts:
                                continue
                            letters = list(string.ascii_uppercase)[:len(opts)]
                            def _mc(prefix, um):
                                # the prompt asks for a LETTER ("respond with exactly one letter") -> score the letter
                                # tokens, NOT the option text (text-loglik after a letter-prompt is mismatched/uninformative).
                                sc = model.mc_loglik(prefix, qids[j], letters, um)
                                pred = letters[max(range(len(sc)), key=lambda k: sc[k])]
                                return {
                                    "score": float(pred == str(it.gold).strip()),
                                    "prediction": pred,
                                    "option_loglik": [float(value) for value in sc],
                                }
                            no_result = _mc(None, False)
                            full_result = _mc(model._embed(cids[j][:, :MAXCTX]), False)
                            memory_result = _mc(Ms[j], True)
                            no, fu, cm = no_result["score"], full_result["score"], memory_result["score"]
                            source_tokens = _source_token_count(it)
                            d["no_ctx"].append(no); d["full"].append(fu); d["compress"].append(cm)
                            sg = model.gate_signal(Ms[j], cids[j][:, :MAXCTX], qids[j])
                            pf = model.gate_probe_features(Ms[j], qids[j], int(cids[j].shape[1]))
                            for s in SIGNALS: sig_vals[s].append(sg.get(s, 0.0))
                            lab.append(int(cm >= 0.5)); fl.append(int(fu >= 0.5)); nl.append(int(no >= 0.5)); noconf.append(sg.get("no_conf", 0.0))
                            records.append({
                                "bench": b,
                                "item_id": str(getattr(it, "item_id", s0 + j)),
                                "document_id": _document_id(it),
                                "gold": str(getattr(it, "gold", "")),
                                "seed": SEED,
                                "is_mc": True,
                                "source_tokens": source_tokens,
                                "ctx_tokens_encoder": int(cids[j].shape[1]),
                                "ctx_tokens_feasible_raw": min(int(cids[j].shape[1]), MAXCTX),
                                "query_tokens": int(qids[j].shape[1]),
                                "memory_tokens": int(Ms[j].shape[1]),
                                "truncation": {
                                    "encoder": source_tokens > int(cids[j].shape[1]),
                                    "feasible_raw": source_tokens > min(int(cids[j].shape[1]), MAXCTX),
                                    "side": "right",
                                },
                                "scores": {"no_ctx": no, "feasible_raw": fu, "compress": cm},
                                "predictions": {
                                    "no_ctx": no_result["prediction"],
                                    "feasible_raw": full_result["prediction"],
                                    "compress": memory_result["prediction"],
                                },
                                "option_loglik": {
                                    "labels": letters,
                                    "no_ctx": no_result["option_loglik"],
                                    "feasible_raw": full_result["option_loglik"],
                                    "compress": memory_result["option_loglik"],
                                },
                                "signals": {s: float(sg.get(s, 0.0)) for s in SIGNALS},
                                "probe_features": pf,
                            })
                        except Exception as e:
                            print(f"[eval] MC item failed ({b}): {repr(e)[:120]}", flush=True)
                else:
                    try:
                        t_no = model._gen_batch([None] * len(chunk), qids, False, GEN_MAX, GEN_BS)
                        t_full = model._gen_batch([model._embed(c[:, :MAXCTX]) for c in cids], qids, False, GEN_MAX, GEN_BS)
                        t_cmp = model._gen_batch(Ms, qids, True, GEN_MAX, GEN_BS)
                    except Exception as e:
                        print(f"[eval] gen failed ({b}): {repr(e)[:120]}", flush=True); continue
                    for j, it in enumerate(its):
                        try:
                            no = score_gen(b, t_no[j], it)[0]; fu = score_gen(b, t_full[j], it)[0]; cm = score_gen(b, t_cmp[j], it)[0]
                            source_tokens = _source_token_count(it)
                            d["no_ctx"].append(no); d["full"].append(fu); d["compress"].append(cm)
                            sg = model.gate_signal(Ms[j], cids[j][:, :MAXCTX], qids[j])
                            pf = model.gate_probe_features(Ms[j], qids[j], int(cids[j].shape[1]))
                            for s in SIGNALS: sig_vals[s].append(sg.get(s, 0.0))
                            lab.append(int(cm >= 0.5)); fl.append(int(fu >= 0.5)); nl.append(int(no >= 0.5)); noconf.append(sg.get("no_conf", 0.0))
                            records.append({
                                "bench": b,
                                "item_id": str(getattr(it, "item_id", s0 + j)),
                                "document_id": _document_id(it),
                                "gold": str(getattr(it, "gold", "")),
                                "seed": SEED,
                                "is_mc": False,
                                "source_tokens": source_tokens,
                                "ctx_tokens_encoder": int(cids[j].shape[1]),
                                "ctx_tokens_feasible_raw": min(int(cids[j].shape[1]), MAXCTX),
                                "query_tokens": int(qids[j].shape[1]),
                                "memory_tokens": int(Ms[j].shape[1]),
                                "truncation": {
                                    "encoder": source_tokens > int(cids[j].shape[1]),
                                    "feasible_raw": source_tokens > min(int(cids[j].shape[1]), MAXCTX),
                                    "side": "right",
                                },
                                "scores": {"no_ctx": no, "feasible_raw": fu, "compress": cm},
                                "predictions": {
                                    "no_ctx": t_no[j],
                                    "feasible_raw": t_full[j],
                                    "compress": t_cmp[j],
                                },
                                "signals": {s: float(sg.get(s, 0.0)) for s in SIGNALS},
                                "probe_features": pf,
                            })
                        except Exception as e:
                            print(f"[eval] gen item failed ({b}): {repr(e)[:120]}", flush=True)
                torch.cuda.empty_cache()
        agg = {m: float(sum(sum(d[m]) for d in per_bench.values()) / max(1, sum(len(d[m]) for d in per_bench.values())))
               for m in ("no_ctx", "full", "compress")}
        run.log({f"eval/acc_{m}": agg[m] for m in agg}, step=10**9)
        def _prf(pred, y):                                                 # POSITIVE = gate fires (keep M); y = compress-correct
            tp = sum(1 for p, t in zip(pred, y) if p and t); fp = sum(1 for p, t in zip(pred, y) if p and not t)
            fn = sum(1 for p, t in zip(pred, y) if not p and t)
            prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
            return 2 * prec * rec / max(1e-9, prec + rec), prec, rec
        # ---- GATING ABLATION: for EACH candidate signal, sweep tau (compress iff signal>=tau else fall back to full);
        # report gated-acc + best-F1 + AUROC; pick the best signal by (gated-acc, F1). ----
        def _sweep(sv):
            best_acc, best_tau = agg["full"], float("inf")
            for tau in [float("-inf")] + sorted(set(sv)):
                fire = [v >= tau for v in sv]
                ga = sum(lab[i] if fire[i] else fl[i] for i in range(len(sv))) / max(1, len(sv))
                if ga > best_acc: best_acc, best_tau = ga, tau
            fire = [v >= best_tau for v in sv] if best_tau != float("inf") else [False] * len(sv)
            f1, pr, rc = _prf(fire, lab)
            return {"gated_acc": best_acc, "tau": best_tau, "f1": f1, "precision": pr, "recall": rc,
                    "auroc": (auroc(sv, lab) if len(set(lab)) > 1 else 0.5), "fire_rate": float(sum(fire) / max(1, len(fire)))}
        gate = {s: _sweep(sig_vals[s]) for s in SIGNALS} if lab else {}
        best_sig = max(gate, key=lambda s: (gate[s]["gated_acc"], gate[s]["f1"])) if gate else "margin"
        # ---- HONEST GATE: K-fold CV. Pick (signal, tau) on TRAIN folds; apply to the HELD-OUT fold. The in-sample
        # _sweep above is >= full BY CONSTRUCTION (it keeps full unless a tau beats it on the SAME items) -> optimistic.
        # The CV number can drop BELOW full if the threshold doesn't generalize; that is the defensible do-no-harm result. ----
        def _gate_cv(k=5, seed=0):
            n = len(lab)
            if n < 2 * k or len(set(lab)) < 2:
                return {}
            import random as _r
            order = list(range(n)); _r.Random(seed).shuffle(order)
            folds = [order[i::k] for i in range(k)]
            held = [None] * n; pick = []
            for f in range(k):
                te = set(folds[f]); tr = [i for i in order if i not in te]
                best = (-1.0, "margin", float("inf"))
                for s in SIGNALS:
                    sv = sig_vals[s]
                    for tau in [float("-inf")] + sorted({sv[i] for i in tr}):
                        ga = sum((lab[i] if sv[i] >= tau else fl[i]) for i in tr) / len(tr)
                        if ga > best[0]: best = (ga, s, tau)
                _, bs, bt = best; pick.append(bs); sv = sig_vals[bs]
                for i in folds[f]: held[i] = (lab[i] if sv[i] >= bt else fl[i], (sv[i] >= bt))
                f1c = _prf([held[i][1] for i in folds[f]], [lab[i] for i in folds[f]])[0]
            ga_cv = sum(h[0] for h in held) / n; fire_cv = sum(1 for h in held if h[1]) / n
            f1_cv, prec_cv, rec_cv = _prf([h[1] for h in held], lab)
            from collections import Counter
            return {"gated_acc_cv": ga_cv, "delta_vs_full": ga_cv - agg["full"], "fire_cv": fire_cv,
                    "fallback_cv": 1.0 - fire_cv, "f1_cv": f1_cv, "precision_cv": prec_cv, "recall_cv": rec_cv,
                    "signals_picked": dict(Counter(pick))}
        gate_cv = _gate_cv() if lab else {}
        # ---- 3-WAY ADAPTIVE GATE (cost-ascending: base/no_ctx -> compress(M) -> full): pick the CHEAPEST path the gate trusts.
        # tau0 over base no-ctx conf (skip context if base already sure); tau1 over the best compress signal; else full. ----
        def _sweep3():
            if not lab: return {}
            n3 = len(lab); best = {"acc": agg["full"], "cost": 1.0, "tau0": float("inf"), "tau1": float("inf")}
            sv1 = sig_vals[best_sig] if best_sig in sig_vals else sig_vals["conf"]
            cand0 = [float("inf")] + sorted(set(noconf))[::max(1, len(set(noconf)) // 20 or 1)]
            cand1 = [float("inf")] + sorted(set(sv1))[::max(1, len(set(sv1)) // 20 or 1)]
            for t0 in cand0:
                for t1 in cand1:
                    acc = cost = 0.0
                    for i in range(n3):
                        if noconf[i] >= t0: acc += nl[i]; cost += 0.0            # base path: 0 ctx tokens
                        elif sv1[i] >= t1: acc += lab[i]; cost += 0.05           # compress path: ~K/L tokens
                        else: acc += fl[i]; cost += 1.0                          # full path: L tokens
                    acc /= n3; cost /= n3
                    if acc > best["acc"] + 1e-9 or (abs(acc - best["acc"]) <= 1e-9 and cost < best["cost"]):
                        best = {"acc": acc, "cost": cost, "tau0": t0, "tau1": t1}
            return best
        gate3 = _sweep3()
        g = gate.get(best_sig, {"gated_acc": agg["full"], "tau": float("inf"), "f1": 0.0, "precision": 0.0, "recall": 0.0, "auroc": 0.5, "fire_rate": 0.0})
        for s, gv in gate.items():
            run.summary.update({f"gate/{s}_gated_acc": gv["gated_acc"], f"gate/{s}_f1": gv["f1"], f"gate/{s}_auroc": gv["auroc"]})
        run.summary.update({"eval/acc_no_ctx": agg["no_ctx"], "eval/acc_full": agg["full"], "eval/acc_compress": agg["compress"],
                            "gate/best_signal": best_sig, "gate/gated_acc": g["gated_acc"], "gate/f1": g["f1"],
                            "gate/auroc": g["auroc"], "gate/precision": g["precision"], "gate/recall": g["recall"], "gate/fire_rate": g["fire_rate"]})
        agg.update({"gated_acc": g["gated_acc"], "gate_best_signal": best_sig, "gate_f1": g["f1"], "gate_auroc": g["auroc"],
                    "gate_precision": g["precision"], "gate_recall": g["recall"], "fire_rate": g["fire_rate"]})
        if gate_cv:                                    # honest held-out gate (the defensible do-no-harm number)
            agg.update(gate_cv)
            run.summary.update({"gate/gated_acc_cv": gate_cv["gated_acc_cv"], "gate/cv_delta_vs_full": gate_cv["delta_vs_full"],
                                "gate/f1_cv": gate_cv["f1_cv"], "gate/fire_cv": gate_cv["fire_cv"]})
        if gate3:                                       # 3-way adaptive gate (selling point): acc at min cost vs always-full
            agg.update({"gate3_acc": gate3["acc"], "gate3_cost": gate3["cost"]})
            run.summary.update({"gate3/acc": gate3["acc"], "gate3/avg_cost": gate3["cost"],
                                "gate3/targ_gated_acc": gate.get("targ", {}).get("gated_acc", 0.0)})
        for b, d in per_bench.items():
            run.summary.update({f"eval_by_bench/{b}_compress": float(sum(d["compress"]) / max(1, len(d["compress"]))),
                                f"eval_by_bench/{b}_full": float(sum(d["full"]) / max(1, len(d["full"]))),
                                f"eval_by_bench/{b}_no_ctx": float(sum(d["no_ctx"]) / max(1, len(d["no_ctx"])))})
        per_bench_acc = {b: {k: float(sum(d[k]) / max(1, len(d[k]))) for k in d} for b, d in per_bench.items()}
        json.dump({"model": mname, "cell": TAG, "seed": SEED,
                   "config": {"K": K, "maxctx": MAXCTX, "enc_maxctx": ENC, "depth": depth_cfg,
                              "lora": LORA, "distill": DISTILL, "recon": RECON, "joint": JOINT,
                              "adv": ADV, "adv_layer": ADV_LAYER, "adv_mode": ADV_MODE,
                              "chunk": CHUNK, "recur": RECUR, "state_cap": STATE_CAP,
                              "n_train": len(pool), "n_val": len(val_pool),
                              "trainable_params": sum(p.numel() for p in model.trainable_parameters())},
                   "agg": agg, "per_bench": per_bench_acc, "gate": gate, "records": records},
                  open(f"{outdir}/{TAG}.json", "w"), indent=2)
        try:
            model.save_adapters(f"{outdir}/{TAG}_adapters.pt")
        except Exception as e:
            print(f"[eval] save_adapters failed: {repr(e)[:120]}", flush=True)
        print("RECIPE_EVAL", TAG, agg); run.finish()

    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
