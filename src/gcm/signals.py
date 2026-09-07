"""Label-free method signals for the do-no-harm gate (the core of the approach).

For a query we read the first answer-token distribution under two paths: the compressed path (read-LoRA ON, the base
reads M) and the full-context path (read-LoRA OFF, the exact base). The gate decides compress-vs-full from cheap
statistics of these distributions (confidence, top-1/top-2 margin, predictive entropy). These are exactly the
quantities we want on the dashboard, plus the AUROC of the gate signal for predicting whether the compressed path
is actually right.
"""
from __future__ import annotations

import numpy as np
import torch

from .lora import set_lora_enabled


@torch.no_grad()
def first_token_signal(model, prefix: torch.Tensor | None, query_ids: torch.Tensor, use_memory: bool) -> dict:
    """First answer-token stats when the base (read-LoRA on/off) conditions on ``prefix`` then the query."""
    set_lora_enabled(model.base, bool(use_memory))
    qe = model._embed(query_ids)
    seq = torch.cat([prefix, qe], 1) if prefix is not None else qe
    lg = model.base(inputs_embeds=seq, use_cache=False).logits[0, -1].float()
    p = lg.softmax(-1)
    t2 = p.topk(2).values
    return {"conf": float(t2[0]), "margin": float(t2[0] - t2[1]),
            "entropy": float(-(p * (p + 1e-9).log()).sum()), "argmax": int(lg.argmax())}


def gate_metrics(comp, full, signal, eps: float = 0.0) -> dict:
    """Do-no-harm gate metrics from per-item compress/full task scores + a gate signal (higher = trust compression).
    Reports the DEPLOYED point (gcm+fallback = threshold maximising mean(comp if keep else full) ⇒ gАcc ≥ full, with its
    fallback rate) AND the best-F1 DETECTION point (F1/precision/recall + confusion TP/FP/FN/TN). y=1 = compression safe
    (comp ≥ full); POSITIVE = gate keeps compressed. FP = harm, FN = wasted fallback."""
    comp, full, s = (np.asarray(a, float) for a in (comp, full, signal))
    y = (comp >= full - eps).astype(int)
    if len(y) == 0:
        return {}
    taus = np.concatenate([[-np.inf], np.unique(s), [np.inf]])
    accs = [float(np.where(s >= t, comp, full).mean()) for t in taus]      # deployed: max gcm+fallback
    keep = s >= taus[int(np.argmax(accs))]
    best_f1, bt = -1.0, np.inf                                             # detection: max F1
    for t in np.unique(s):
        p = s >= t
        tp, fp, fn = ((p) & (y == 1)).sum(), ((p) & (y == 0)).sum(), ((~p) & (y == 1)).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        if f1 > best_f1:
            best_f1, bt = f1, t
    p = s >= bt
    TP = int(((p) & (y == 1)).sum()); FP = int(((p) & (y == 0)).sum())
    FN = int(((~p) & (y == 1)).sum()); TN = int(((~p) & (y == 0)).sum())
    P = TP / (TP + FP) if TP + FP else 0.0; R = TP / (TP + FN) if TP + FN else 0.0
    return {"gcm_fallback": float(np.where(keep, comp, full).mean()), "fallback_rate": float((~keep).mean()),
            "full": float(full.mean()), "compress": float(comp.mean()),
            "F1": 2 * P * R / (P + R) if P + R else 0.0, "precision": P, "recall": R,
            "TP": TP, "FP": FP, "FN": FN, "TN": TN, "auroc": auroc(s, y)}


def auroc(scores, labels) -> float:
    """AUROC via the Mann-Whitney statistic. labels: 1 = compressed path correct. Higher score = more trusted."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    npos, n = int(y.sum()), len(y)
    if npos == 0 or npos == n:
        return float("nan")
    order = s.argsort(kind="mergesort")
    so = s[order]
    r = np.empty(n, float)
    i = 0
    while i < n:                                   # average ranks within tie groups
        j = i
        while j + 1 < n and so[j + 1] == so[i]:
            j += 1
        r[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(n, float); ranks[order] = r
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * (n - npos)))
