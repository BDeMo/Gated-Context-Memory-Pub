"""v1.7 benchmark testing.

Train the trainable baselines on the training corpus, then evaluate ALL selected methods on each eval
bench, writing one JSONL record per item with every method's score. Each record therefore carries
n_0 (no_ctx), n_full (full_ctx), and n_w (each compressor) together, plus token costs and the transfer
relation, which is exactly what the v1.7 compress-decision metric needs downstream.

Usage:
  python -m mem_embedding.gcm.harness --base Qwen/Qwen3-8B --out DIR \
      --methods no_ctx,full_ctx,cartridge,gist --train-dataset bfcl_simple \
      --eval-benches bfcl_simple,apibank,toolace,trivia_qa
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from .data import DOMAIN, MC_BENCHES, PRIMARY, load_items, options_for, relation, score_gen
from .methods import REGISTRY, Method, TrainCfg
from .runtime import Runtime

try:  # GCM (our method, in svc) plugs into the same harness as a Method; svc is optional.
    from svc.method import GCM

    REGISTRY["gcm"] = GCM
except Exception:  # noqa: BLE001
    pass


def build_methods(names: list[str]) -> list[Method]:
    bad = [n for n in names if n not in REGISTRY]
    if bad:
        raise KeyError(f"unknown methods {bad}; known: {sorted(REGISTRY)}")
    return [REGISTRY[n]() for n in names]


def _set_base_lora(rt: Runtime, on: bool) -> None:
    """Toggle any non-merged base LoRA (svc.lora.LoRALinear). Compressed path => on (base learns to read M);
    full_ctx / no_ctx / fallback => off, recovering the EXACT original base (the do-no-harm guarantee).
    Duck-typed so the harness keeps no hard svc dependency."""
    for mod in rt.model.modules():
        if mod.__class__.__name__ == "LoRALinear" and hasattr(mod, "enabled"):
            mod.enabled = bool(on)


def agnostic_confidence(rt: Runtime, method: Method, item: Any, qids: torch.Tensor) -> dict:
    """Compressor-AGNOSTIC gate signal: the frozen base's first-answer-token distribution when it reads THIS
    method's compressed memory + the query. Works for any compressor (kv-prefix like Cartridge, soft-prefix like
    Gist) — that is the point: the robustness layer (detect bad compression -> fall back) is not tied to our
    encoder. Returns confidence (top-1 prob), margin (top1-top2), neg_entropy. Lower confidence => likelier the
    compression dropped what the query needs => fall back."""
    qe = rt.embed(qids)
    is_kv = hasattr(method, "kv_prefix")
    with torch.no_grad():
        if is_kv:
            zk, zv = method.kv_prefix(rt, item)
            lg = rt.kv_logits(zk, zv, qe)[0, -1].float()
        else:
            pre, ctx_len = method.prefix(rt, item)
            if pre is None:
                seq, am = qe, None
            else:
                seq = torch.cat([pre, qe], dim=1)
                am = rt.gist_mask(seq.shape[1], pre.shape[1], ctx_len) if ctx_len > 0 else None
            lg = rt.model(inputs_embeds=seq, attention_mask=am, use_cache=False).logits[0, -1].float()
    p = lg.softmax(-1)
    top2 = torch.topk(p, 2).values
    eps = 1e-9
    ent = float(-(p * (p + eps).log()).sum())
    out = {"conf": float(top2[0]), "margin": float(top2[0] - top2[1]), "neg_entropy": -ent}
    # LLM-judge baseline (D-Mem style): same memory M, ask the base whether it can answer -> P(Yes)-P(No).
    try:
        jids = rt.tok(" Can you answer the question accurately using only the provided context? Answer Yes or No.\nAnswer:",
                      return_tensors="pt", add_special_tokens=False).input_ids.to(qe.device)
        je = rt.embed(jids)
        with torch.no_grad():
            if is_kv:
                jlg = rt.kv_logits(zk, zv, torch.cat([qe, je], dim=1))[0, -1].float()
            else:
                if pre is None:
                    jseq, jam = torch.cat([qe, je], dim=1), None
                else:
                    jseq = torch.cat([pre, qe, je], dim=1)
                    jam = rt.gist_mask(jseq.shape[1], pre.shape[1], ctx_len) if ctx_len > 0 else None
                jlg = rt.model(inputs_embeds=jseq, attention_mask=jam, use_cache=False).logits[0, -1].float()
        jp = jlg.softmax(-1)
        yes = rt.tok.encode(" Yes", add_special_tokens=False)[0]
        no = rt.tok.encode(" No", add_special_tokens=False)[0]
        out["judge"] = float(jp[yes] - jp[no])
    except Exception:
        out["judge"] = 0.0
    return out


def evaluate_item(rt: Runtime, method: Method, bench: str, item: Any, qids: torch.Tensor) -> tuple[float, str]:
    """Score one (method, item): MC accuracy for MC benches, else primary generation metric."""
    _set_base_lora(rt, bool(getattr(method, "use_base_lora", False)))   # GCM+LoRA reads M with LoRA; full/no_ctx without
    kv = method.kv_prefix(rt, item) if hasattr(method, "kv_prefix") else None   # faithful Cartridge: KV-cache path
    if bench in MC_BENCHES:
        opts = options_for(item)
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"][: len(opts)]
        # The benchmark prompt explicitly asks for a letter. Scoring option text after a
        # letter-answer prompt was the pre-v1.8 MC bug; score the answer letters here too.
        sc = rt.mc_loglik_kv(kv[0], kv[1], qids, letters) if kv else rt.mc_loglik(*method.prefix(rt, item), qids, letters)
        pred = letters[max(range(len(sc)), key=lambda i: sc[i])] if sc else ""
        correct = float(pred == item.gold and item.gold in letters)
        return correct, pred
    if kv:
        pred = rt.generate_kv(kv[0], kv[1], qids)
    else:
        pre, ctx_len = method.prefix(rt, item)
        pred = rt.generate(pre, ctx_len, qids)
    val, _ = score_gen(bench, pred, item)
    return val, pred


def deployed_cost(name: str, K: int, n_query: int, n_ctx: int) -> int:
    """Token budget the method consumes at deploy time (gist drops raw ctx via KV caching)."""
    if name == "no_ctx":
        return n_query
    if name == "full_ctx":
        return n_ctx + n_query
    return K + n_query  # cartridge / gist / future compressors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="no_ctx,full_ctx,cartridge,gist")
    ap.add_argument("--train-dataset", default="trivia_qa")
    ap.add_argument("--eval-benches", default="trivia_qa")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--n-memory", type=int, default=64)
    ap.add_argument("--n-items", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=150)
    ap.add_argument("--n-chunks", type=int, default=6)
    ap.add_argument("--eval-n-chunks", type=int, default=8)
    ap.add_argument("--max-ctx-tokens", type=int, default=1024)
    ap.add_argument("--max-input-tokens", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--epochs", type=int, default=0, help="v1.7.6: >0 => train for N epochs over the (full) train set (1 epoch = every item seen once); overrides --steps")
    ap.add_argument("--patience", type=int, default=0, help="v1.7.6: >0 => early-stop after N consecutive val-loss checks without improvement (converge)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--distill-tokens", type=int, default=16)
    ap.add_argument("--distill-topk", type=int, default=64)
    ap.add_argument("--distill-temp", type=float, default=1.0)
    # OURS (svc) knobs
    ap.add_argument("--enc-layers", default="4", help="GCM encoder depth: an int, or 'full'/'half' relative to the base num_hidden_layers (v1.7.6)")
    ap.add_argument("--n-dec-layers", type=int, default=2)
    ap.add_argument("--lam-rec", type=float, default=1.0)
    ap.add_argument("--lam-dev", type=float, default=0.05)
    ap.add_argument("--lam-task", type=float, default=1.0)
    ap.add_argument("--lam-distill", type=float, default=0.0)
    ap.add_argument("--train-fp32", action="store_true", help="GCM: train the compressor in fp32 (default: bf16, matches deployment + baselines)")
    ap.add_argument("--no-train-fp32", action="store_true", help="(deprecated alias; bf16 is now the default)")
    ap.add_argument("--enc-init", default="copy", choices=["copy", "random"], help="GCM encoder init (ablation)")
    ap.add_argument("--gcm-agnostic", action="store_true", help="GCM: train+eval the query-agnostic memory M0 (ablation)")
    ap.add_argument("--lam-adv", type=float, default=0.0, help="GCM: adversarial losslessness weight (0=off)")
    ap.add_argument("--adv-layer", type=int, default=18, help="base layer for the adversarial discriminator")
    ap.add_argument("--lam-contrast", type=float, default=0.0, help="GCM v1.8.0: InfoNCE semantic-alignment weight (alt. to adv; 0=off)")
    ap.add_argument("--lam-align", type=float, default=0.0, help="GCM v1.8.0: PURE positive alignment weight, no negatives (0=off)")
    ap.add_argument("--align-layer", type=int, default=18, help="base layer for adv/contrast semantic alignment")
    ap.add_argument("--contrast-temp", type=float, default=0.2, help="InfoNCE temperature for contrastive alignment")
    ap.add_argument("--contrast-bank", type=int, default=256, help="FIFO negative-bank size for contrastive (batch-1)")
    ap.add_argument("--m-norm-match", default="off", choices=["off", "hard", "learn"], help="A1: L2-normalize M to the embedding scale (fix OOD)")
    ap.add_argument("--m-manifold-temp", type=float, default=0.0, help="A2: >0 => project M onto the token-embedding hull at this softmax temp")
    ap.add_argument("--base-lora-rank", type=int, default=0, help="A3: >0 => add a LoRA on the frozen base (learns to read M)")
    ap.add_argument("--inject", default="prefix", choices=["prefix", "kv"], help="A4: prefix(input-embed) | kv(per-layer KV-cache)")
    ap.add_argument("--recon-m-only", action="store_true", help="C1: decoder reconstructs ctx from M only (true lossless signal)")
    ap.add_argument("--chunk-size", type=int, default=0, help="v1.7.4: >0 => length-adaptive chunked memory (K slots PER chunk of this many ctx tokens; budget scales with length at fixed ratio)")
    ap.add_argument("--chunk-keep", type=int, default=0, help="v1.7.4: >0 => query-relevance routing, keep top-K chunk-memories (soft retrieval; bounds budget at keep*K)")
    ap.add_argument("--grad-accum", type=int, default=8, help="v1.7.5 STABILITY: effective batch via gradient accumulation (1=old batch-1 noisy training)")
    ap.add_argument("--warmup-frac", type=float, default=0.05, help="v1.7.5: LR warmup fraction")
    ap.add_argument("--no-lr-cosine", action="store_true", help="v1.7.5: disable cosine LR decay (use constant after warmup)")
    ap.add_argument("--signals", action="store_true", help="log per-item gate signals for methods that support it (GCM)")
    ap.add_argument("--bfcl-full-call", action="store_true", help="R3: BFCL asks for the full call (name+args) and scores args-aware (tool_call_acc)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    # PROVENANCE (v1.7.5): records_*.jsonl do NOT carry the training recipe/env, so write a per-run meta.json.
    # This is the fix for "is this cell actually the v1.7.5 setting?" — every future run self-documents.
    try:
        import os
        import sys
        import transformers as _tf
        meta = {**vars(a), "ts": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
                "torch": torch.__version__, "transformers": _tf.__version__,
                "host": os.uname().nodename, "python": sys.version.split()[0], "argv": sys.argv}
        (a.out / "meta.json").write_text(json.dumps(meta, default=str, indent=2))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"meta.json write failed: {e}")
    if a.bfcl_full_call:
        import os
        os.environ["MEM_BFCL_FULL_CALL"] = "1"
    torch.manual_seed(a.seed)

    # v1.7.6: Qwen3.5 (linear-attn) — force the PURE-TORCH gated-delta-rule path. The fla Triton chunk-backward is
    # buggy on Hopper + Triton>=3.4 (fla #640) and tilelang (the suggested fix) crashes on import here. Nulling the
    # fla symbols makes the modeling fall back to its built-in torch_{chunk,recurrent}_gated_delta_rule (correct, slower).
    for _mod in ("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"):
        try:
            import importlib
            _m = importlib.import_module(_mod)
            _m.chunk_gated_delta_rule = None
            _m.fused_recurrent_gated_delta_rule = None
        except Exception:  # noqa: BLE001
            pass

    names = [m.strip() for m in a.methods.split(",") if m.strip()]
    if "gist" in names:
        # SDPA accepts the 4D gist mask and avoids eager attention's O(L²) materialization.
        # cuDNN's masked-row backward is disabled because it produced NaNs on Hopper.
        torch.backends.cuda.enable_cudnn_sdp(False)
    gist_eager = os.environ.get("GCM_GIST_EAGER", "0") == "1"
    rt = Runtime(
        a.base, device=a.device, dtype=a.dtype, n_memory=a.n_memory,
        max_ctx_tokens=a.max_ctx_tokens, max_input_tokens=a.max_input_tokens,
        max_new_tokens=a.max_new_tokens, eager=("gist" in names and gist_eager),
    )
    methods = build_methods(names)
    # v1.7.6: resolve encoder depth 'full'/'half' against the base model's layer count
    _baseL = getattr(rt.model.config, "num_hidden_layers", None) or len(rt.model.model.layers)
    _el = str(a.enc_layers).strip().lower()
    enc_layers = _baseL if _el == "full" else (max(1, _baseL // 2) if _el == "half" else int(a.enc_layers))
    # v1.7.6: load the (full) train set, then convert epochs -> steps (1 epoch = every item seen once)
    train_items = load_items(a.train_dataset, a.n_items, a.n_chunks, a.seed, a.train_split)
    eval_benches = [b.strip() for b in a.eval_benches.split(",") if b.strip()]
    eval_cache = {
        bench: load_items(bench, a.n_eval, a.eval_n_chunks, a.seed + 1, a.eval_split)
        for bench in eval_benches
    }
    # Native public splits can still contain exact duplicate rows (BFCL has one). Remove any training
    # item whose content/query/gold fingerprint appears in the paper evaluation pool.
    def _fingerprint(item: Any) -> str:
        import hashlib
        text = "\u241f".join((
            "\n".join(str(c) for c in (getattr(item, "chunks", None) or [])),
            str(getattr(item, "query", "")),
            str(getattr(item, "gold", "")),
        ))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    eval_fingerprints = {
        _fingerprint(item)
        for items in eval_cache.values()
        for item in items
    }
    train_items = [item for item in train_items if _fingerprint(item) not in eval_fingerprints]
    n_steps = a.epochs * len(train_items) if a.epochs > 0 else a.steps
    logger.info(f"v1.7.6 recipe: enc_layers={enc_layers} (base={_baseL}), steps={n_steps} "
                f"(epochs={a.epochs} x {len(train_items)} items), patience={a.patience}")
    cfg = TrainCfg(
        steps=n_steps, lr=a.lr, distill_tokens=a.distill_tokens,
        distill_topk=a.distill_topk, distill_temp=a.distill_temp,
        enc_layers=enc_layers, n_dec_layers=a.n_dec_layers,
        lam_rec=a.lam_rec, lam_dev=a.lam_dev, train_fp32=(a.train_fp32 and not a.no_train_fp32),
        lam_task=a.lam_task, lam_distill=a.lam_distill,
        enc_init=a.enc_init, gcm_agnostic=a.gcm_agnostic,
        lam_adv=a.lam_adv, adv_layer=a.adv_layer,
        lam_contrast=a.lam_contrast, lam_align=a.lam_align, align_layer=a.align_layer, contrast_temp=a.contrast_temp, contrast_bank=a.contrast_bank,
        m_norm_match=a.m_norm_match,
        m_manifold_temp=a.m_manifold_temp, base_lora_rank=a.base_lora_rank, inject=a.inject,
        recon_m_only=a.recon_m_only, chunk_size=a.chunk_size, chunk_keep=a.chunk_keep,
        grad_accum=a.grad_accum, warmup_frac=a.warmup_frac, lr_cosine=(not a.no_lr_cosine),
        patience=a.patience,
    )

    for m in methods:
        if getattr(m, "trainable", False):
            logger.info(f"training {m.name} on corpus={a.train_dataset} ({len(train_items)} items)")
            m.train(rt, train_items, cfg)
    base_trainable = sum(p.numel() for p in rt.model.parameters() if p.requires_grad)
    method_params = {}
    for m in methods:
        own = 0
        for attr in ("gist_p", "zk", "zv"):
            value = getattr(m, attr, None)
            if value is not None and hasattr(value, "numel"):
                own += int(value.numel())
        method_params[m.name] = own + (base_trainable if m.name == "gist" else 0)
    (a.out / "method_params.json").write_text(json.dumps(method_params, indent=2))

    for bench in eval_benches:
        items = eval_cache[bench]
        rel = relation(a.train_dataset, bench)
        path = a.out / f"records_{bench}.jsonl"
        logger.info(f"=== eval {bench} rel={rel}: {len(items)} items x {len(methods)} methods ===")
        with open(path, "w") as f:
            for ii, it in enumerate(items):
                if ii % 8 == 0:
                    torch.cuda.empty_cache()
                qids = rt.query_ids(it)
                n_query = int(qids.shape[1])
                n_ctx = int(rt.ctx_ids(it).shape[1])
                rec: dict[str, Any] = {
                    "bench": bench, "item_id": it.item_id, "gold": it.gold,
                    "train_dataset": a.train_dataset, "relation": rel,
                    "train_domain": DOMAIN.get(a.train_dataset), "eval_domain": DOMAIN.get(bench),
                    "mc": bench in MC_BENCHES, "primary": PRIMARY.get(bench),
                    "n_query_tokens": n_query, "n_ctx_tokens": n_ctx,
                    "trainable_params": method_params,
                    "scores": {}, "preds": {}, "cost": {},
                }
                for m in methods:
                    try:
                        val, pred = evaluate_item(rt, m, bench, it, qids)
                        rec["scores"][m.name] = val
                        rec["preds"][m.name] = pred[:200]
                    except Exception as e:  # keep the run alive; record the failure per cell
                        rec["scores"][m.name] = None
                        rec["preds"][m.name] = f"ERR {repr(e)[:140]}"
                    k_eff = m.mem_len(n_ctx) if hasattr(m, "mem_len") else rt.K  # v1.7.4: chunked M scales with len
                    rec["cost"][m.name] = deployed_cost(m.name, k_eff, n_query, n_ctx)
                    if a.signals and m.name not in ("no_ctx", "full_ctx"):  # per-item gate signals (GCM + agnostic conf for any compressor)
                        try:
                            rec["signals"] = m.signals(rt, it) if hasattr(m, "signals") else agnostic_confidence(rt, m, it, qids)
                        except Exception as e:  # noqa: BLE001
                            rec["signals"] = {"err": repr(e)[:100]}
                f.write(json.dumps(rec) + "\n")
        logger.info(f"wrote {path}")
    print("DONE gcm benchmark testing")


if __name__ == "__main__":
    main()
