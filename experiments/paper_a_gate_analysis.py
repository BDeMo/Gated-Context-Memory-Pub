"""Held-out, fixed-signal gate analysis for Paper A.

The legacy v1.8 evaluator selected both the signal and threshold on each
evaluation fold.  This analyzer instead pre-registers ``conf`` as the primary
signal, uses a calibration subset only to choose a threshold, and evaluates
that frozen threshold on a disjoint test subset.  TARG and margin are reported
as baselines; they never influence the primary threshold.

Input files are the JSON artifacts emitted by the updated ``run_recipe.py``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


SIGNALS = ("conf", "targ", "margin")
LEARNED_SIGNALS = ("joint_probe", "performance_predictor")
# Fixed before reading calibration scores or labels. ``inf`` is always-raw.
FORMAL_CONF_THRESHOLDS = (float("inf"),) + tuple(i / 20 for i in range(20, -1, -1))
EMPIRICAL_RISK_MODES = ("accepted_positive_harm", "signed_excess")


def _binary_auroc(scores: list[float], labels: list[bool]) -> float:
    """Tie-aware ROC-AUC; returns 0.5 when only one class is present."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    if not xs:
        return float("inf")
    pos = min(len(xs) - 1, max(0.0, q * (len(xs) - 1)))
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _candidate_thresholds(values: list[float], n: int = 101) -> list[float]:
    return [float("inf")] + sorted({_quantile(values, i / (n - 1)) for i in range(n)})


def _score(records: list[dict[str, Any]], signal: str, tau: float) -> dict[str, Any]:
    full = [float(row["scores"]["feasible_raw"]) for row in records]
    comp = [float(row["scores"]["compress"]) for row in records]
    fire = [float(row["signals"].get(signal, 0.0)) >= tau for row in records]
    gated = [c if f else b for b, c, f in zip(full, comp, fire)]
    oracle = [max(b, c) for b, c in zip(full, comp)]
    n = max(1, len(records))
    accepted = [(b, c) for b, c, f in zip(full, comp, fire) if f]
    signed_harm = [b - c for b, c in accepted]
    positive_harm = [max(b - c, 0.0) for b, c in accepted]
    harm_count = sum(b > c + 1e-12 for b, c in accepted)
    benefit_count = sum(c > b + 1e-12 for b, c in accepted)
    fallback_labels = [b > c + 1e-12 for b, c in zip(full, comp)]
    fallback_scores = [
        -float(row["signals"].get(signal, 0.0))
        for row in records
    ]
    binary_outcomes = {
        "both_correct": sum(b >= 0.5 and c >= 0.5 for b, c in zip(full, comp)),
        "raw_correct_memory_wrong": sum(b >= 0.5 and c < 0.5 for b, c in zip(full, comp)),
        "memory_correct_raw_wrong": sum(c >= 0.5 and b < 0.5 for b, c in zip(full, comp)),
        "both_wrong": sum(b < 0.5 and c < 0.5 for b, c in zip(full, comp)),
    }
    return {
        "n": len(records),
        "full": mean(full) if full else 0.0,
        "compress": mean(comp) if comp else 0.0,
        "gated": mean(gated) if gated else 0.0,
        "oracle": mean(oracle) if oracle else 0.0,
        "delta_vs_full": (mean(gated) - mean(full)) if full else 0.0,
        "excess_risk": (mean(full) - mean(gated)) if full else 0.0,
        "coverage": sum(fire) / n,
        "fallback_rate": 1.0 - sum(fire) / n,
        "fallback_auroc": _binary_auroc(fallback_scores, fallback_labels),
        "accepted_n": len(accepted),
        "accepted_signed_excess": mean(signed_harm) if signed_harm else None,
        "accepted_positive_harm": mean(positive_harm) if positive_harm else None,
        "accepted_harm_rate": harm_count / len(accepted) if accepted else None,
        "accepted_benefit_rate": benefit_count / len(accepted) if accepted else None,
        "pairwise_harm_rate_all": sum(b > c + 1e-12 for b, c in zip(full, comp)) / n,
        "pairwise_benefit_rate_all": sum(c > b + 1e-12 for b, c in zip(full, comp)) / n,
        "binary_outcomes_at_0.5": binary_outcomes,
    }


def _choose_threshold(
    calibration: list[dict[str, Any]],
    signal: str,
    epsilon: float,
    risk_mode: str = "accepted_positive_harm",
) -> tuple[float, dict[str, Any]]:
    if risk_mode not in EMPIRICAL_RISK_MODES:
        raise ValueError(f"unknown empirical risk mode: {risk_mode}")
    values = [float(row["signals"].get(signal, 0.0)) for row in calibration]
    feasible: list[tuple[float, float, float, dict[str, Any]]] = []
    for tau in _candidate_thresholds(values):
        metrics = _score(calibration, signal, tau)
        risk = (
            metrics["accepted_positive_harm"]
            if risk_mode == "accepted_positive_harm"
            else metrics["excess_risk"]
        )
        # Conditional accepted harm is undefined at zero coverage, so "never"
        # remains a safe fallback but is not an empirically feasible candidate.
        if risk is not None and risk <= epsilon:
            # Maximize useful compression coverage, then calibration score.
            feasible.append((metrics["coverage"], metrics["gated"], tau, metrics))
    if not feasible:
        tau = float("inf")
        return tau, _score(calibration, signal, tau)
    _, _, tau, metrics = max(feasible, key=lambda row: (row[0], row[1]))
    return tau, metrics


def _choose_ltt_threshold(
    calibration: list[dict[str, Any]],
    signal: str,
    epsilon: float,
    delta: float,
    max_items_per_document: int,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Cluster-valid Bonferroni LTT over a fixed confidence-threshold family.

    For document d, S_d is summed accepted positive harm and C_d is the
    accepted-item count. The target E[S_d] / E[C_d] <= epsilon is equivalent
    to E[S_d - epsilon*C_d] <= 0 when coverage is nonzero. Documents are the
    independent units; rows inside one document never increase test power.
    Dividing by a pre-registered cluster-size cap maps the statistic to an
    interval of width one for Hoeffding's inequality.
    """
    candidates = FORMAL_CONF_THRESHOLDS
    alpha = delta / max(1, len(candidates))
    certified: list[tuple[float, float, float, dict[str, Any], float]] = []
    diagnostics = []
    for tau in candidates:
        metrics = _score(calibration, signal, tau)
        clusters: dict[str, list[dict[str, Any]]] = {}
        for row in calibration:
            document_id = str(row["document_id"])
            clusters.setdefault(document_id, []).append(row)
        if any(len(rows) > max_items_per_document for rows in clusters.values()):
            raise ValueError(
                f"document exceeds pre-registered cap {max_items_per_document}"
            )
        cluster_stats = []
        accepted_n = 0
        harm_sum = 0.0
        for rows in clusters.values():
            accepted_harm = []
            for row in rows:
                if float(row["signals"][signal]) >= tau:
                    raw = float(row["scores"]["feasible_raw"])
                    memory = float(row["scores"]["compress"])
                    accepted_harm.append(max(raw - memory, 0.0))
            accepted_n += len(accepted_harm)
            harm_sum += sum(accepted_harm)
            cluster_stats.append(
                (sum(accepted_harm) - epsilon * len(accepted_harm))
                / max_items_per_document
            )
        empirical = harm_sum / accepted_n if accepted_n else None
        if not accepted_n or not cluster_stats:
            p_value = None
            passes = False
        else:
            observed = mean(cluster_stats)
            gap = -observed
            p_value = (
                math.exp(-2.0 * len(cluster_stats) * gap * gap)
                if gap > 0
                else 1.0
            )
            passes = p_value <= alpha
        diagnostics.append({
            "tau": tau if math.isfinite(tau) else ("always" if tau < 0 else "never"),
            "coverage": metrics["coverage"],
            "accepted_n": accepted_n,
            "independent_documents": len(cluster_stats),
            "max_items_per_document": max_items_per_document,
            "empirical_positive_harm": empirical,
            "p_value": p_value,
            "bonferroni_alpha": alpha,
            "certified": passes,
        })
        if passes:
            certified.append((metrics["coverage"], metrics["gated"], tau, metrics, float(p_value)))
    if not certified:
        tau = float("inf")
        return tau, _score(calibration, signal, tau), {
            "certified": False,
            "p_value": None,
            "delta": delta,
            "family_size": len(candidates),
            "threshold_family": [
                value if math.isfinite(value) else "never" for value in candidates
            ],
            "diagnostics": diagnostics,
        }
    _, _, tau, metrics, p_value = max(certified, key=lambda row: (row[0], row[1]))
    return tau, metrics, {
        "certified": True,
        "p_value": p_value,
        "delta": delta,
        "family_size": len(candidates),
        "threshold_family": [
            value if math.isfinite(value) else "never" for value in candidates
        ],
        "diagnostics": diagnostics,
    }


def _held_out_curve(
    calibration: list[dict[str, Any]],
    test: list[dict[str, Any]],
    signal: str,
) -> list[dict[str, Any]]:
    values = [float(row["signals"].get(signal, 0.0)) for row in calibration]
    points = []
    for tau in _candidate_thresholds(values, n=21):
        points.append(
            {
                "tau": tau if math.isfinite(tau) else ("always" if tau < 0 else "never"),
                "calibration": _score(calibration, signal, tau),
                "test": _score(test, signal, tau),
            }
        )
    return points


def _random_gate(records: list[dict[str, Any]], coverage: float, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    full = [float(row["scores"]["feasible_raw"]) for row in records]
    comp = [float(row["scores"]["compress"]) for row in records]
    n_fire = round(coverage * len(records))
    selected = set(rng.sample(range(len(records)), min(n_fire, len(records))))
    gated = [comp[i] if i in selected else full[i] for i in range(len(records))]
    return {
        "gated": mean(gated) if gated else 0.0,
        "delta_vs_full": (mean(gated) - mean(full)) if full else 0.0,
        "coverage": len(selected) / max(1, len(records)),
    }


def _oracle(records: list[dict[str, Any]]) -> dict[str, float]:
    full = [float(row["scores"]["feasible_raw"]) for row in records]
    comp = [float(row["scores"]["compress"]) for row in records]
    gated = [max(b, c) for b, c in zip(full, comp)]
    return {
        "gated": mean(gated) if gated else 0.0,
        "delta_vs_full": (mean(gated) - mean(full)) if full else 0.0,
        "coverage": sum(c >= b for b, c in zip(full, comp)) / max(1, len(records)),
    }


def _learned_probe_scores(
    calibration: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> dict[str, tuple[list[float], list[float]]]:
    """Fit direct learned baselines on calibration only.

    joint_probe predicts compression harm (raw score > memory score).
    performance_predictor regresses the signed raw-minus-memory score gap,
    analogous to a lightweight PoC-style performance predictor.
    """
    if not calibration or not test or not all(row.get("probe_features") for row in calibration + test):
        return {}
    keys = sorted(set.intersection(*[
        set(row["probe_features"]) for row in calibration
    ]))
    if not keys:
        return {}
    if not all(all(key in row["probe_features"] for key in keys) for row in test):
        return {}
    x_cal = np.asarray([[float(row["probe_features"][k]) for k in keys] for row in calibration])
    x_test = np.asarray([[float(row["probe_features"][k]) for k in keys] for row in test])
    mu = x_cal.mean(0)
    sd = x_cal.std(0)
    sd[sd < 1e-8] = 1.0
    x_cal = np.ascontiguousarray(
        np.nan_to_num(
            np.clip((x_cal - mu) / sd, -20.0, 20.0),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ),
        dtype=np.float64,
    )
    x_test = np.ascontiguousarray(
        np.nan_to_num(
            np.clip((x_test - mu) / sd, -20.0, 20.0),
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ),
        dtype=np.float64,
    )
    x_cal = np.concatenate([x_cal, np.ones((len(x_cal), 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=1)

    y_harm = np.asarray([
        float(row["scores"]["feasible_raw"]) > float(row["scores"]["compress"]) + 1e-12
        for row in calibration
    ], dtype=float)
    w = np.zeros(x_cal.shape[1], dtype=float)
    for _ in range(400):
        z = np.clip(np.dot(x_cal, w), -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = np.dot(x_cal.T, p - y_harm) / len(x_cal)
        grad[:-1] += 1e-2 * w[:-1]
        grad = np.nan_to_num(grad, nan=0.0, posinf=10.0, neginf=-10.0)
        norm = np.linalg.norm(grad)
        if norm > 10.0:
            grad *= 10.0 / norm
        w = np.clip(w - 0.02 * grad, -20.0, 20.0)
    # Higher gate score means safer compression.
    joint_cal = (-np.dot(x_cal, w)).tolist()
    joint_test = (-np.dot(x_test, w)).tolist()

    y_gap = np.asarray([
        float(row["scores"]["feasible_raw"]) - float(row["scores"]["compress"])
        for row in calibration
    ])
    penalty = np.eye(x_cal.shape[1]) * math.sqrt(1e-2)
    penalty[-1, -1] = 0.0
    design = np.concatenate([x_cal, penalty], axis=0)
    target = np.concatenate([y_gap, np.zeros(x_cal.shape[1])])
    wr = np.clip(np.linalg.lstsq(design, target, rcond=None)[0], -20.0, 20.0)
    perf_cal = (-np.dot(x_cal, wr)).tolist()
    perf_test = (-np.dot(x_test, wr)).tolist()
    arrays = (joint_cal, joint_test, perf_cal, perf_test)
    if not all(math.isfinite(value) for values in arrays for value in values):
        return {}
    return {
        "joint_probe": (joint_cal, joint_test),
        "performance_predictor": (perf_cal, perf_test),
    }


def _validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("gate analysis requires records")
    seen = set()
    for index, row in enumerate(records):
        if not row.get("document_id"):
            raise ValueError(f"record {index} lacks document_id")
        item_key = (str(row["document_id"]), str(row.get("item_id", index)))
        if item_key in seen:
            raise ValueError(f"duplicate evaluation item: {item_key}")
        seen.add(item_key)
        for score_name in ("feasible_raw", "compress"):
            value = float(row.get("scores", {}).get(score_name, float("nan")))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {score_name} at record {index}: {value}")
        value = float(row.get("signals", {}).get("conf", float("nan")))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid conf at record {index}: {value}")


def analyze_group(
    records: list[dict[str, Any]],
    *,
    calibration_fraction: float,
    repeats: int,
    epsilon: float,
    delta: float,
    seed: int,
    max_items_per_document: int = 64,
    empirical_risk_mode: str = "accepted_positive_harm",
) -> dict[str, Any]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0,1)")
    if repeats < 1 or not 0.0 <= epsilon <= 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("invalid repeats/epsilon/delta")
    if max_items_per_document < 1:
        raise ValueError("max_items_per_document must be positive")
    if empirical_risk_mode not in EMPIRICAL_RISK_MODES:
        raise ValueError(f"unknown empirical risk mode: {empirical_risk_mode}")
    records = copy.deepcopy(records)
    _validate_records(records)
    split_results: list[dict[str, Any]] = []
    for repeat in range(repeats):
        buckets: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            group = str(record.get("document_id") or record.get("item_id") or index)
            buckets.setdefault(group, []).append(index)
        groups = sorted(buckets)
        random.Random(seed + repeat).shuffle(groups)
        target = max(1, min(len(records) - 1, round(calibration_fraction * len(records))))
        cal_indices: list[int] = []
        cal_groups: list[str] = []
        for group in groups:
            if len(cal_indices) >= target and cal_indices:
                break
            cal_groups.append(group)
            cal_indices.extend(buckets[group])
        cal_set = set(cal_indices)
        test_indices = [index for index in range(len(records)) if index not in cal_set]
        if not test_indices:
            moved = cal_groups.pop()
            moved_indices = set(buckets[moved])
            cal_indices = [index for index in cal_indices if index not in moved_indices]
            test_indices = sorted(moved_indices)
        calibration = [records[i] for i in cal_indices]
        test = [records[i] for i in test_indices]
        learned_routes = {}
        cal_doc_order = sorted(cal_groups)
        fit_docs = set(cal_doc_order[::2])
        probe_fit = [
            row for row in calibration if str(row["document_id"]) in fit_docs
        ]
        probe_threshold = [
            row for row in calibration if str(row["document_id"]) not in fit_docs
        ]
        if probe_fit and probe_threshold:
            learned = _learned_probe_scores(
                probe_fit,
                probe_threshold + test,
            )
            for signal, (_, eval_scores) in learned.items():
                n_threshold = len(probe_threshold)
                cal_records = copy.deepcopy(probe_threshold)
                test_records = copy.deepcopy(test)
                for record, score in zip(cal_records, eval_scores[:n_threshold]):
                    record.setdefault("signals", {})[signal] = float(score)
                for record, score in zip(test_records, eval_scores[n_threshold:]):
                    record.setdefault("signals", {})[signal] = float(score)
                learned_routes[signal] = (cal_records, test_records)
        result: dict[str, Any] = {
            "repeat": repeat,
            "n_calibration": len(calibration),
            "n_test": len(test),
            "n_calibration_documents": len(cal_groups),
            "n_test_documents": len(groups) - len(cal_groups),
            "calibration_documents": sorted(cal_groups),
            "test_documents": sorted(set(groups) - set(cal_groups)),
            "signals": {},
        }
        for signal in SIGNALS:
            tau, cal_metrics = _choose_threshold(
                calibration, signal, epsilon, empirical_risk_mode
            )
            test_metrics = _score(test, signal, tau)
            result["signals"][signal] = {
                "tau": tau if math.isfinite(tau) else ("always" if tau < 0 else "never"),
                "calibration": cal_metrics,
                "test": test_metrics,
                "held_out_curve": _held_out_curve(calibration, test, signal),
                "random_at_matched_coverage": _random_gate(
                    test, test_metrics["coverage"], seed + 100_000 + repeat
                ),
            }
        for signal, (signal_calibration, signal_test) in learned_routes.items():
            tau, cal_metrics = _choose_threshold(
                signal_calibration,
                signal,
                epsilon,
                empirical_risk_mode,
            )
            test_metrics = _score(signal_test, signal, tau)
            result["signals"][signal] = {
                "tau": tau if math.isfinite(tau) else ("always" if tau < 0 else "never"),
                "calibration": cal_metrics,
                "test": test_metrics,
                "held_out_curve": _held_out_curve(
                    signal_calibration,
                    signal_test,
                    signal,
                ),
                "random_at_matched_coverage": _random_gate(
                    signal_test,
                    test_metrics["coverage"],
                    seed + 200_000 + repeat,
                ),
                "probe_fit_n": len(probe_fit),
                "threshold_calibration_n": len(signal_calibration),
            }
        # One pre-registered split defines the formal deployment policy. Remaining
        # repeated holdouts are descriptive split-sensitivity checks only.
        if repeat == 0:
            ltt_tau, ltt_cal, ltt_test = _choose_ltt_threshold(
                calibration,
                "conf",
                epsilon,
                delta,
                max_items_per_document,
            )
            result["ltt_conf"] = {
                "tau": (
                    ltt_tau
                    if math.isfinite(ltt_tau)
                    else ("always" if ltt_tau < 0 else "never")
                ),
                "calibration": ltt_cal,
                "test": _score(test, "conf", ltt_tau),
                "certification": ltt_test,
                "formal_split": True,
            }
        result["always_full"] = _score(test, "conf", float("inf"))
        result["always_compress"] = _score(test, "conf", float("-inf"))
        result["oracle"] = _oracle(test)
        split_results.append(result)

    summary: dict[str, Any] = {}
    analyzed_signals = tuple(split_results[0]["signals"]) if split_results else SIGNALS
    for signal in analyzed_signals:
        metrics = {
            key: [
                split["signals"][signal]["test"][key]
                for split in split_results
            ]
            for key in (
                "gated",
                "delta_vs_full",
                "excess_risk",
                "coverage",
                "fallback_rate",
                "fallback_auroc",
            )
        }
        summary[signal] = {
            key: {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
            for key, values in metrics.items()
        }
        positive_harms = [
            split["signals"][signal]["test"]["accepted_positive_harm"]
            for split in split_results
        ]
        defined_positive_harms = [
            value for value in positive_harms if value is not None
        ]
        summary[signal]["accepted_positive_harm"] = {
            "mean": (
                mean(defined_positive_harms) if defined_positive_harms else None
            ),
            "std": (
                pstdev(defined_positive_harms)
                if len(defined_positive_harms) > 1
                else (0.0 if defined_positive_harms else None)
            ),
            "min": min(defined_positive_harms) if defined_positive_harms else None,
            "max": max(defined_positive_harms) if defined_positive_harms else None,
            "n_defined": len(defined_positive_harms),
        }
        if empirical_risk_mode == "accepted_positive_harm":
            operating_risk = (
                summary[signal]["accepted_positive_harm"]["mean"]
                if len(defined_positive_harms) == len(split_results)
                else None
            )
        else:
            operating_risk = summary[signal]["excess_risk"]["mean"]
        summary[signal]["passes_operating_point"] = (
            operating_risk is not None
            and operating_risk <= epsilon
            and summary[signal]["coverage"]["mean"] >= 0.20
        )
    ltt_keys = (
        "gated",
        "delta_vs_full",
        "excess_risk",
        "coverage",
        "fallback_rate",
        "fallback_auroc",
        "accepted_positive_harm",
        "accepted_harm_rate",
        "accepted_benefit_rate",
    )
    formal = split_results[0]["ltt_conf"]
    summary["ltt_conf"] = {}
    for key in ltt_keys:
        value = formal["test"][key]
        summary["ltt_conf"][key] = {
            "mean": value,
            "std": 0.0 if value is not None else None,
            "min": value,
            "max": value,
            "n_defined": int(value is not None),
        }
    summary["ltt_conf"]["certified"] = bool(
        formal["certification"]["certified"]
    )
    summary["ltt_conf"]["formal_repeat"] = 0
    return {
        "empirical_risk_mode": empirical_risk_mode,
        "summary": summary,
        "splits": split_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-items-per-document", type=int, default=64)
    parser.add_argument(
        "--empirical-risk-mode",
        choices=EMPIRICAL_RISK_MODES,
        default="accepted_positive_harm",
        help=(
            "Calibration constraint. signed_excess reproduces the legacy "
            "population-level criterion where benefits can offset harms."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates: dict[
        tuple[str, str, str, int],
        list[tuple[Path, list[dict[str, Any]]]],
    ] = {}
    sources: list[str] = []
    seen_paths = set()
    for path in args.inputs:
        resolved = path.resolve()
        if resolved in seen_paths:
            raise ValueError(f"duplicate input artifact: {resolved}")
        seen_paths.add(resolved)
        artifact = json.loads(path.read_text())
        sources.append(str(path))
        model = str(artifact.get("model", "unknown"))
        run_seed = int(artifact.get("seed", -1))
        cell = str(artifact.get("cell", path.stem))
        source_cell = cell.removeprefix("feat_")
        by_bench: dict[str, list[dict[str, Any]]] = {}
        for record in artifact.get("records", []):
            by_bench.setdefault(str(record["bench"]), []).append(record)
        for bench, records in by_bench.items():
            item_keys = [
                (str(row.get("document_id")), str(row.get("item_id")))
                for row in records
            ]
            if len(item_keys) != len(set(item_keys)):
                raise ValueError(f"duplicate items inside artifact {path}")
            key = (source_cell, model, bench, run_seed)
            candidates.setdefault(key, []).append((path, records))

    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    selected_sources: dict[str, str] = {}
    for key, options in candidates.items():
        # A feature rerun replaces its source artifact; it is never appended as
        # extra observations. Prefer complete probe features, then row count.
        options.sort(
            key=lambda option: (
                int(bool(option[1]) and all(row.get("probe_features") for row in option[1])),
                len(option[1]),
            ),
            reverse=True,
        )
        selected_path, records = options[0]
        grouped[key] = records
        selected_sources["/".join(map(str, key))] = str(selected_path)

    report = {
        "schema_version": 1,
        "protocol": {
            "primary_signal": "conf",
            "baseline_signals": ["targ", "margin", "joint_probe", "performance_predictor"],
            "calibration_fraction": args.calibration_fraction,
            "repeats": args.repeats,
            "split_unit": "document_id when available, otherwise item_id",
            "max_mean_excess_risk": args.epsilon,
            "empirical_threshold_risk_mode": args.empirical_risk_mode,
            "empirical_threshold_risk_epsilon": args.epsilon,
            "ltt_positive_harm_epsilon": args.epsilon,
            "ltt_familywise_delta": args.delta,
            "ltt_threshold_family": [
                value if math.isfinite(value) else "never"
                for value in FORMAL_CONF_THRESHOLDS
            ],
            "formal_split_repeat": 0,
            "formal_independent_unit": "exact source-context document_id",
            "max_items_per_document": args.max_items_per_document,
            "min_mean_coverage": 0.20,
            "signal_selection_on_test": False,
        },
        "sources": sources,
        "selected_sources": selected_sources,
        "groups": {},
    }
    for (source_cell, model, bench, run_seed), records in sorted(grouped.items()):
        name = f"{source_cell}/{model}/{bench}/seed-{run_seed}"
        report["groups"][name] = analyze_group(
            records,
            calibration_fraction=args.calibration_fraction,
            repeats=args.repeats,
            epsilon=args.epsilon,
            delta=args.delta,
            seed=args.seed,
            max_items_per_document=args.max_items_per_document,
            empirical_risk_mode=args.empirical_risk_mode,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: value["summary"]["conf"] for name, value in report["groups"].items()}, indent=2))


if __name__ == "__main__":
    main()
