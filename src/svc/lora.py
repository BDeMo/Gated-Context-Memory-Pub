"""Minimal LoRA (no peft dependency) for A3 (v1.7.1.3): a low-rank adapter on the frozen base so it can learn to
READ the compressed memory M. Wraps target nn.Linear modules with `y = W0 x + (alpha/r)·B A x`; B is zero-init so
the adapted model starts identical to the base. Only A/B train (base stays frozen).

CORE DESIGN (per user 2026-06-12): the adapter is *never merged*. The compressed path runs with LoRA ENABLED
(base learns to read M); the FALLBACK path runs with LoRA DISABLED, recovering the EXACT original base on full
context -- this is the do-no-harm guarantee. Use `set_lora_enabled(model, False)` to toggle off for fallback."""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        dev = base.weight.device
        self.lora_A = nn.Parameter((torch.randn(r, d_in, device=dev, dtype=dtype) * 0.01))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=dtype))
        self.scale = float(alpha) / float(r)
        self.enabled = True  # toggled off for the fallback path => exact original base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:                      # fallback path: original frozen base, bit-for-bit
            return out
        delta = (x.to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()) * self.scale
        return out + delta.to(out.dtype)


def set_lora_enabled(model: nn.Module, flag: bool) -> None:
    """Enable/disable every LoRALinear in `model` (compressed path on; fallback path off)."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = flag


@contextlib.contextmanager
def lora_disabled(model: nn.Module):
    """Context manager: run the fallback/full path on the exact original base, then restore."""
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    prev = [m.enabled for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, p in zip(mods, prev):
            m.enabled = p


def add_lora(model: nn.Module, rank: int, targets=("q_proj", "v_proj"),
             alpha: float | None = None, dtype: torch.dtype = torch.float32) -> list[nn.Parameter]:
    """Replace every Linear named in `targets` with a LoRALinear. Returns the trainable LoRA params."""
    alpha = alpha if alpha is not None else float(rank)
    params: list[nn.Parameter] = []
    for mod in list(model.modules()):
        for cname, child in list(mod.named_children()):
            if cname in targets and isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank, alpha, dtype)
                setattr(mod, cname, lora)
                params += [lora.lora_A, lora.lora_B]
    return params
