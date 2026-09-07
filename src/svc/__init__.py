"""svc: Self-Verifying Compressor (Paper B v1.7).

A context compressor on a frozen LLM that runs query-agnostic or query-conditioned, emits a structurally
fused do-no-harm gate from its own internals, and is trained to be self-verifying (its signals predict its
own failures) so detect-and-fall-back is robust. Public resources reused from ``llm_infra``.
"""

from svc.compressor import SelfVerifyingCompressor

__all__ = ["SelfVerifyingCompressor"]
