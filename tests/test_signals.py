import math

from gcm.signals import auroc, gate_metrics


def test_auroc_handles_ties_and_degenerate_labels():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5
    assert math.isnan(auroc([0.1, 0.2], [1, 1]))


def test_gate_metrics_falls_back_on_harmful_compression():
    metrics = gate_metrics(
        comp=[1.0, 0.0, 1.0, 0.0],
        full=[1.0, 1.0, 1.0, 1.0],
        signal=[0.9, 0.1, 0.8, 0.2],
    )

    assert metrics["gcm_fallback"] == 1.0
    assert metrics["gcm_fallback"] >= metrics["full"]
    assert metrics["fallback_rate"] == 0.5
