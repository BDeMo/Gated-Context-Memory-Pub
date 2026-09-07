"""GCM — Gated Context Memory.

A frozen language model is its own context compressor: run it over ``[context ; query ; K memory tokens]``,
read its own features at the memory positions, and project them (nonlinear, normalized onto the embedding
manifold) into a short soft memory ``M`` that is injected back into the *same* model in place of the context.
A do-no-harm gate decides per query whether to trust ``M`` or fall back to the full context.

One base model, small adapters — no duplicated weights.
"""
import torch as _torch
# cuDNN SDPA backward returns NaN on heavily/fully-masked query rows (our 4D context+mem mask
# triggers this). The flash/efficient/math backends are numerically correct, so disable cuDNN SDP
# process-wide. THE fix for the training-divergence NaN. Do NOT remove.
try:
    _torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass

from .compressor import SelfCompressor
from .model import GCMModel
from .lora import add_lora, set_lora_enabled, lora_disabled
from .compat import enable_torch_linear_attention
from .train import train_compressor
from .signals import first_token_signal, auroc, gate_metrics

__all__ = ["SelfCompressor", "GCMModel", "add_lora", "set_lora_enabled",
           "lora_disabled", "enable_torch_linear_attention", "train_compressor",
           "first_token_signal", "auroc", "gate_metrics"]
__version__ = "0.1.0"
