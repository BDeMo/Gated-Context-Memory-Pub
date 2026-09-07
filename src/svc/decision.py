"""Decision-metric reporter: the compress-vs-fallback confusion matrix.

Reads per-item JSONL records (the gcm harness schema: per-method scores + cost + relation) and computes the
v1.7 headline metric, framing "use the compressed memory vs fall back" as a binary classifier:

  precision = of the inputs we compressed, the fraction that did no harm  (compress effectively / do-no-harm)
  recall    = of the genuinely-compressible inputs, the fraction we compressed  (compress as much as possible)
  F1        = the combined judgment number
  AUROC     = how well a gate SIGNAL ranks safe-to-compress vs not (if a signal is logged)
  AURC / cost-coverage = accuracy and token-cost as we vary the operating threshold

Two tracks:
  B (compression): label y = 1[n_w >= n_full - eps], fallback = full context.
  A (do-no-harm):  label y = 1[n_w >= n_0],          fallback = no-context base.

Baselines (no gate signal) report the base rate + oracle + trivial gates + cost; OURS adds a `--signal-key`
to get the gate's precision/recall/F1/AUROC. Pure analysis (numpy only); runs wherever the JSONL is.

Usage:
  python -m svc.decision --records DIR --compressor cartridge --track B [--signal-key gate] [--eps 0.0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import numpy as np

REL_ORDER = ["in_task", "cross_task_in_domain", "cross_task_cross_domain"]
_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)  # numpy>=2 renamed trapz->trapezoid


def load_records(root: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(root, "records_*.jsonl"))) if os.path.isdir(root) else sorted(glob.glob(root))
    recs: list[dict] = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _signal_of(rec: dict, key: str | None) -> float | None:
    if not key:
        return None
    if key in rec and isinstance(rec[key], (int, float)):
        return float(rec[key])
    sig = rec.get("signals") or {}
    v = sig.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks; nan if a class is empty."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _arrays(recs: list[dict], comp: str, track: str, eps: float, signal_key: str | None):
    n0, nw, nfull, sig, cw, cf, rel, bench = [], [], [], [], [], [], [], []
    for r in recs:
        sc = r.get("scores") or {}
        a, b, f = sc.get("no_ctx"), sc.get(comp), sc.get("full_ctx")
        if b is None or f is None or a is None:
            continue
        n0.append(float(a)); nw.append(float(b)); nfull.append(float(f))
        cost = r.get("cost") or {}
        cw.append(float(cost.get(comp, r.get("n_query_tokens", 0) or 0)))
        cf.append(float(cost.get("full_ctx", (r.get("n_ctx_tokens", 0) or 0) + (r.get("n_query_tokens", 0) or 0))))
        rel.append(r.get("relation", "?")); bench.append(r.get("bench", "?"))
        sig.append(_signal_of(r, signal_key))
    d = dict(n0=np.array(n0), nw=np.array(nw), nfull=np.array(nfull),
             cw=np.array(cw), cf=np.array(cf), rel=np.array(rel), bench=np.array(bench))
    d["y"] = (d["nw"] >= (d["nfull"] - eps)).astype(float) if track == "B" else (d["nw"] >= d["n0"]).astype(float)
    d["sig"] = np.array([np.nan if v is None else v for v in sig], float)
    return d


def report_block(d: dict, signal_key: str | None) -> dict:
    n = len(d["nw"])
    if n == 0:
        return {"n": 0}
    y = d["y"]
    fb = d["nfull"]  # track-B fallback (full); track-A would pass n0 as fb upstream
    out = {
        "n": n, "base_rate": float(y.mean()),
        "acc_no_ctx": float(d["n0"].mean()), "acc_compressor": float(d["nw"].mean()),
        "acc_full": float(fb.mean()),
        "acc_best": float(np.where(y == 1, d["nw"], fb).mean()),  # per-input best of {compress, fallback}
        "cost_compressor": float(d["cw"].mean()), "cost_full": float(d["cf"].mean()),
    }
    s = d["sig"]
    if signal_key and np.isfinite(s).any():
        m = np.isfinite(s)
        ys, ss, nw_s, fb_s, cw_s, cf_s = y[m], s[m], d["nw"][m], fb[m], d["cw"][m], d["cf"][m]
        out["auroc"] = _auroc(ys, ss)
        # best-F1 over thresholds + recall at precision>=0.95
        taus = np.concatenate([np.unique(ss), [np.inf]])   # +inf => "compress nothing" (do-no-harm floor) reachable
        best = {"f1": -1.0}
        rec_at_p95 = 0.0
        for t in taus:
            pred = ss >= t
            tp = float(((pred == 1) & (ys == 1)).sum()); fp = float(((pred == 1) & (ys == 0)).sum())
            fn = float(((pred == 0) & (ys == 1)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            recl = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * recl / (prec + recl) if prec + recl else 0.0
            if f1 > best["f1"]:
                best = {"f1": f1, "precision": prec, "recall": recl, "tau": float(t),
                        "coverage": float(pred.mean())}
            if prec >= 0.95 and recl > rec_at_p95:
                rec_at_p95 = recl
        out.update({f"gate_{k}": v for k, v in best.items()})
        out["recall_at_p95"] = rec_at_p95
        # cost-coverage frontier by signal ordering: compress the highest-signal items first
        order = np.argsort(-ss)
        accs, costs, covs = [], [], []
        for c in np.linspace(0.0, 1.0, 11):
            k = int(round(c * len(ss)))
            comp_mask = np.zeros(len(ss), bool); comp_mask[order[:k]] = True
            acc = np.where(comp_mask, nw_s, fb_s).mean()
            cost = np.where(comp_mask, cw_s, cf_s).mean()
            accs.append(float(acc)); costs.append(float(cost)); covs.append(float(c))
        out["cost_coverage"] = {"coverage": covs, "accuracy": accs, "cost": costs}
        out["aurc_acc"] = float(_TRAPZ(accs, covs))  # area under accuracy-coverage (higher better)
    return out


def _fmt(b: dict) -> str:
    if not b.get("n"):
        return "n=0"
    s = (f"n={b['n']:>4}  base_rate={b['base_rate']:.3f}  "
         f"acc[no_ctx/comp/full/best]={b['acc_no_ctx']:.3f}/{b['acc_compressor']:.3f}/"
         f"{b['acc_full']:.3f}/{b['acc_best']:.3f}  cost[comp/full]={b['cost_compressor']:.0f}/{b['cost_full']:.0f}")
    if "auroc" in b:
        s += (f"\n      gate: AUROC={b['auroc']:.3f}  F1={b.get('gate_f1', 0):.3f} "
              f"(P={b.get('gate_precision', 0):.3f} R={b.get('gate_recall', 0):.3f} "
              f"cov={b.get('gate_coverage', 0):.3f})  recall@P95={b.get('recall_at_p95', 0):.3f}  "
              f"AURC_acc={b.get('aurc_acc', 0):.3f}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="dir with records_*.jsonl, or a glob")
    ap.add_argument("--compressor", default="cartridge", help="which method is OURS/the compressor row")
    ap.add_argument("--track", choices=["A", "B"], default="B")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--signal-key", default=None, help="record key holding the gate signal (OURS only)")
    a = ap.parse_args()

    recs = load_records(a.records)
    d = _arrays(recs, a.compressor, a.track, a.eps, a.signal_key)
    if a.track == "A":  # fallback is no-ctx for track A
        d["nfull"] = d["n0"]; d["cf"] = np.minimum(d["cf"], d["cw"])  # no-ctx cost ~ query only
    print(f"=== compressor={a.compressor}  track={a.track}  eps={a.eps}  signal={a.signal_key}  N={len(d['nw'])} ===")
    print("[overall] " + _fmt(report_block(d, a.signal_key)))
    print("\n[by relation]")
    for rel in REL_ORDER:
        mask = d["rel"] == rel
        if mask.any():
            sub = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
            print(f"  {rel:24} " + _fmt(report_block(sub, a.signal_key)))
    print("\n[by bench]")
    for bench in sorted(set(d["bench"].tolist())):
        mask = d["bench"] == bench
        sub = {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        print(f"  {bench:22} " + _fmt(report_block(sub, a.signal_key)))


if __name__ == "__main__":
    main()
