"""v1.7 Gated Compressor Module (GCM) project.

Clean rebuild for the v1.7 reframe (see research-notes paper-b-forgetting-gating/
v1.7-gated-compressor-module-2026-06-10.md). This step ships only the BASELINE compressors and the
benchmark-testing harness; the gated module + the decision-metric reporter come later.

Public resources (datasets, model weights, metrics) are reused from ``llm_infra``; nothing here owns a
dataset or a metric (per the mem-test workspace conventions).
"""
