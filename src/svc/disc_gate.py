"""Rank discriminator-gate (and intrinsic) strategies on one records dir.

The adversarial discriminator D scores p = P(D thinks [M;q] is "compressed") at base layer-ell. The compressor was
trained to push p->0 (look like full) wherever it could, so we read p as a LEARNED gate. Each "strategy" below is a
single-threshold gate on a derived signal (the reporter sweeps tau and reports the best-F1 operating point):

  S_low   (recommended): compress iff p < tau          signal=disc_negp  (D fooled => looks like full => lossless)
  S_conf  : compress iff |p-0.5| > tau                  signal=disc_conf  (D confident EITHER way => in-distribution)
  S_ent   : compress iff binary-entropy(p) low          signal=disc_negent (same idea as S_conf, entropy form)
  S_high  : compress iff p > tau                         signal=disc_p     (D sure "compressed" => a recognized M)

vs INTRINSIC signals (neg_recon / dlogit / dcode / conf / margin / neg_entropy / mnorm) and the trivial
always-compress / always-full / per-item-best references. Track B = do-no-harm-vs-full (y=1 iff comp >= full-eps).

Usage:
  python -m svc.disc_gate --records DIR [--compressor gcm] [--track B] [--eps 0.0] [--rel in_task]
"""
from __future__ import annotations

import argparse

import numpy as np

from svc.decision import _arrays, _auroc, load_records, report_block


def gated_acc(d: dict) -> tuple[float, float, float]:
    """IN-SAMPLE CEILING of the GCM+fallback accuracy (NOT realizable): pick the single global threshold on this
    signal that maximises mean( comp if signal>=tau else full ) ON THE SAME DATA. Returns (acc, coverage, tau).
    Optimistic (the threshold is tuned on the eval set) — report `gated_acc_cv` (held-out) as the honest number
    and treat this only as an upper bound."""
    s = d["sig"]
    m = np.isfinite(s)
    nw, nf, ss = d["nw"][m], d["nfull"][m], s[m]
    if len(ss) == 0:
        return float("nan"), float("nan"), float("nan")
    best, bcov, btau = -1.0, 0.0, float("inf")
    for t in np.concatenate([[-np.inf], np.unique(ss), [np.inf]]):   # +inf => compress-nothing (do-no-harm floor)
        comp = ss >= t
        acc = float(np.where(comp, nw, nf).mean())
        if acc > best:
            best, bcov, btau = acc, float(comp.mean()), float(t)
    return best, bcov, btau


def gated_acc_cv(d: dict, folds: int = 5, seed: int = 0) -> tuple[float, float, float]:
    """HONEST realizable GCM+fallback accuracy: K-fold CROSS-VALIDATED — the threshold is fit on the training
    folds and applied to the held-out fold (never tuned on the items it scores). Returns (acc, coverage, nan).
    This is the number to report; `gated_acc` (in-sample) is only the optimistic ceiling. Where compression
    rarely beats full, the fitted threshold becomes 'compress almost nothing' ⇒ acc≈full, cov≈0 (honest do-no-harm)."""
    s = d["sig"]
    m = np.isfinite(s)
    nw, nf, ss = d["nw"][m], d["nfull"][m], s[m]
    n = len(ss)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    if n < 2 * folds:  # too few items to CV reliably -> leave-one-out
        folds = max(2, n // 2)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    realized = np.empty(n)
    comp_flag = np.zeros(n, bool)
    for k in range(folds):
        val = idx[k::folds]
        tr = np.setdiff1d(idx, val)
        if len(tr) == 0 or len(val) == 0:
            continue
        best, btau = -1.0, float("inf")  # fit tau on the TRAIN folds only
        for t in np.concatenate([[-np.inf], np.unique(ss[tr]), [np.inf]]):   # +inf => compress-nothing reachable
            comp = ss[tr] >= t
            acc = float(np.where(comp, nw[tr], nf[tr]).mean())
            if acc > best:
                best, btau = acc, float(t)
        cv_comp = ss[val] >= btau
        realized[val] = np.where(cv_comp, nw[val], nf[val])
        comp_flag[val] = cv_comp
    return float(realized.mean()), float(comp_flag.mean()), float("nan")

STRATEGIES = {  # human label -> signal key fed to the reporter (orientation: higher signal => compress)
    "S_low  (p<tau, D=full)": "disc_negp",
    "S_conf (|p-.5|>tau)": "disc_conf",
    "S_ent  (low entropy)": "disc_negent",
    "S_high (p>tau, D=comp)": "disc_p",
    "intr:neg_recon": "neg_recon",
    "intr:dlogit": "dlogit",
    "intr:neg_dlogit": "neg_dlogit",
    "intr:dcode": "dcode",
    "intr:conf": "conf",
    "intr:margin": "margin",
    "intr:neg_entropy": "neg_entropy",
    "intr:mnorm": "mnorm",
}


def ablate(d: dict, npts: int = 11) -> list[tuple]:
    """Threshold (coverage) ablation for one signal: compress the top-c fraction by signal, report the gate's
    precision/recall/F1 (vs label y=safe-to-compress) and the realised GCM+fallback accuracy at each c."""
    s = d["sig"]
    m = np.isfinite(s)
    s, y, nw, nf = s[m], d["y"][m], d["nw"][m], d["nfull"][m]
    N = len(s)
    order = np.argsort(-s)  # highest signal = compress first
    out = []
    for c in np.linspace(0, 1, npts):
        k = int(round(c * N))
        comp = np.zeros(N, bool)
        comp[order[:k]] = True
        tp = float(((comp) & (y == 1)).sum()); fp = float(((comp) & (y == 0)).sum()); fn = float(((~comp) & (y == 1)).sum())
        P = tp / (tp + fp) if tp + fp else float("nan")
        R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if (P == P and P + R) else 0.0
        acc = float(np.where(comp, nw, nf).mean())
        out.append((c, P, R, F1, acc))
    return out


_GATE_KEYS = ["dlogit", "neg_dlogit", "dcode", "neg_recon", "conf", "margin", "neg_entropy", "mnorm", "disc_p"]


def _logreg_cv(X, y, folds: int = 5, l2: float = 1.0, iters: int = 800, lr: float = 0.3):
    """Held-out logit per item from K-fold logistic regression (numpy; standardized features, L2)."""
    n, f = X.shape
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    idx = np.arange(n); np.random.default_rng(0).shuffle(idx)
    out = np.zeros(n)
    for k in range(folds):
        val = idx[k::folds]; tr = np.setdiff1d(idx, val)
        if len(tr) == 0 or len(val) == 0:
            continue
        w = np.zeros(f); b = 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Xs[tr] @ w + b))); g = p - y[tr]
            w -= lr * (Xs[tr].T @ g / len(tr) + l2 * w / len(tr)); b -= lr * g.mean()
        out[val] = Xs[val] @ w + b
    return out


def fit_gate(recs, compressor, track, eps, rel):
    """B1 (v1.7.2.1): supervised gate = K-fold logistic regression over ALL signals; report HELD-OUT AUROC/F1/gAcc
    vs the best single signal. Builds X from each record's signals dict."""
    keys = [k for k in _GATE_KEYS if any(k in (r.get("signals") or {}) for r in recs)]
    d0 = _arrays(recs, compressor, track, eps, keys[0] if keys else None)
    if track == "A":
        d0["nfull"] = d0["n0"]
    mask = np.ones(len(d0["y"]), bool)
    if rel:
        mask = d0["rel"] == rel
    feats = []
    for r in recs:
        sig = r.get("signals") or {}
        feats.append([float(sig.get(k, 0.0)) for k in keys])
    X = np.array(feats, float)
    # align X to the (filtered) records used by _arrays (it drops items missing scores); reuse d0 ordering via mask
    keep = [i for i, r in enumerate(recs) if (r.get("scores") or {}).get(compressor) is not None
            and (r.get("scores") or {}).get("full_ctx") is not None and (r.get("scores") or {}).get("no_ctx") is not None]
    X = X[keep]
    y, nw, nf = d0["y"], d0["nw"], d0["nfull"]
    if rel:
        rm = d0["rel"] == rel
        X, y, nw, nf = X[rm], y[rm], nw[rm], nf[rm]
    s = _logreg_cv(X, y.astype(float))
    auc = _auroc(y, s)
    # best-F1 + gAcc on the held-out combined score
    d = {"sig": s, "y": y, "nw": nw, "nfull": nf, "n0": nf}
    best = {"f1": -1.0}
    for t in np.unique(s):
        pred = s >= t
        tp = float(((pred) & (y == 1)).sum()); fp = float(((pred) & (y == 0)).sum()); fn = float(((~pred) & (y == 1)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
        F1 = 2 * P * R / (P + R) if P + R else 0.0
        if F1 > best["f1"]:
            best = {"f1": F1, "P": P, "R": R}
    ga, gcov, _ = gated_acc(d)
    return keys, auc, best, ga, gcov, len(y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--compressor", default="gcm")
    ap.add_argument("--track", choices=["A", "B"], default="B")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--rel", default=None, help="restrict to one relation (in_task / cross_task_in_domain / ...)")
    ap.add_argument("--ablate", default=None, help="signal key: print a threshold/coverage ablation for it")
    ap.add_argument("--fit", action="store_true", help="B1: fit a supervised CV logistic-regression gate over all signals")
    a = ap.parse_args()

    recs = load_records(a.records)
    if a.fit:
        keys, auc, best, ga, gcov, n = fit_gate(recs, a.compressor, a.track, a.eps, a.rel)
        print(f"=== B1 supervised gate (5-fold logreg)  records={a.records}  N={n}  rel={a.rel or 'ALL'} ===")
        print(f"features ({len(keys)}): {','.join(keys)}")
        print(f"HELD-OUT  AUROC={auc:.3f}  bestF1={best['f1']:.3f} (P={best.get('P', 0):.3f} R={best.get('R', 0):.3f})  "
              f"gAcc={ga:.3f} (cov={gcov:.3f})")
        return
    if a.ablate:
        d = _arrays(recs, a.compressor, a.track, a.eps, a.ablate)
        if a.track == "A":
            d["nfull"] = d["n0"]
        if a.rel:
            m = d["rel"] == a.rel
            d = {k: (v[m] if hasattr(v, "shape") else v) for k, v in d.items()}
        b = report_block(d, a.ablate)
        print(f"=== threshold ablation  signal={a.ablate}  compressor={a.compressor}  N={b.get('n')}  "
              f"always-full={b.get('acc_full', 0):.3f}  always-compress={b.get('acc_compressor', 0):.3f}  "
              f"best/oracle={b.get('acc_best', 0):.3f}  AUROC={b.get('auroc', float('nan')):.3f} ===")
        print(f"{'compress%':>9} {'fallback%':>9} {'prec':>6} {'recall':>7} {'F1':>6} {'gAcc':>6}")
        for c, P, R, F1, acc in ablate(d):
            print(f"{c * 100:9.0f} {(1 - c) * 100:9.0f} {P:6.3f} {R:7.3f} {F1:6.3f} {acc:6.3f}")
        return
    rows = []
    for label, key in STRATEGIES.items():
        d = _arrays(recs, a.compressor, a.track, a.eps, key)
        if a.track == "A":
            d["nfull"] = d["n0"]
        if a.rel:
            m = d["rel"] == a.rel
            d = {k: (v[m] if hasattr(v, "shape") else v) for k, v in d.items()}
        b = report_block(d, key)
        if not b.get("n"):
            continue
        ga, gcov, _ = gated_acc(d)            # in-sample ceiling
        gacv, gcovcv, _ = gated_acc_cv(d)     # HONEST held-out (5-fold CV)
        b["gated_acc"], b["gated_cov"] = ga, gcov
        b["gacc_cv"], b["gcov_cv"] = gacv, gcovcv
        rows.append((label, b))

    if not rows:
        print("no records / no signals found")
        return
    ref = rows[0][1]
    print(f"=== disc-gate strategies  records={a.records}  compressor={a.compressor}  track={a.track}  "
          f"rel={a.rel or 'ALL'}  N={ref['n']} ===")
    print(f"references: always-full={ref['acc_full']:.3f}  always-compress={ref['acc_compressor']:.3f}  "
          f"no_ctx={ref['acc_no_ctx']:.3f}  best(compress,full)={ref['acc_best']:.3f}  "
          f"base_rate(compressible)={ref['base_rate']:.3f}")
    print(f"  [GCM+fallback gAcc vs always-full {ref['acc_full']:.3f} vs best/oracle {ref['acc_best']:.3f}]")
    print(f"  [gAcc_cv = HELD-OUT 5-fold (honest, realizable); gAcc = in-sample ceiling (optimistic)]")
    print(f"\n{'strategy':24} {'gAcc_cv':>8} {'cov_cv':>7} {'gAcc_is':>8} {'F1':>6} {'P':>6} {'R':>6} {'AUROC':>6}")
    for label, b in sorted(rows, key=lambda r: -(r[1].get("gacc_cv") if r[1].get("gacc_cv") == r[1].get("gacc_cv") else -1)):
        print(f"{label:24} {b.get('gacc_cv', 0):8.3f} {b.get('gcov_cv', 0):7.3f} {b.get('gated_acc', 0):8.3f} "
              f"{b.get('gate_f1', 0):6.3f} {b.get('gate_precision', 0):6.3f} {b.get('gate_recall', 0):6.3f} "
              f"{b.get('auroc', float('nan')):6.3f}")


if __name__ == "__main__":
    main()
