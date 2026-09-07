"""Minimal toggleable LoRA (no extra deps).

Wraps target ``nn.Linear`` modules with ``y = W0 x + (alpha/r) B A x`` (B zero-init, so the adapted model starts
identical to the base). Only A/B train; the base stays frozen. The adapter is **never merged**: enable it for the
compressed path (the base learns to read the memory M) and disable it for the fallback/full path, recovering the
exact original base — the do-no-harm guarantee.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 32, alpha: float | None = None,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features, device=dev, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dtype))
        self.scale = float(alpha if alpha is not None else r) / float(r)
        self.enabled = True
        # optional SECOND adapter, active only during the ENCODE phase (ablation: "train the encoder").
        self.enc_A = None; self.enc_B = None; self.enc_scale = 1.0; self.enc_enabled = False

    def add_enc_adapter(self, r: int = 16, alpha: float | None = None, dtype: torch.dtype | None = None) -> list:
        dev = self.base.weight.device; dt = dtype or self.lora_A.dtype
        self.enc_A = nn.Parameter(torch.randn(r, self.base.in_features, device=dev, dtype=dt) * 0.01)
        self.enc_B = nn.Parameter(torch.zeros(self.base.out_features, r, device=dev, dtype=dt))
        self.enc_scale = float(alpha if alpha is not None else r) / float(r)
        return [self.enc_A, self.enc_B]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + ((x.to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t()) * self.scale).to(out.dtype)
        if self.enc_enabled and self.enc_A is not None:
            out = out + ((x.to(self.enc_A.dtype) @ self.enc_A.t() @ self.enc_B.t()) * self.enc_scale).to(out.dtype)
        return out


def add_lora(model: nn.Module, rank: int = 32, targets=("q_proj", "v_proj"),
             alpha: float | None = None, dtype: torch.dtype = torch.float32) -> list[nn.Parameter]:
    """Replace every Linear named in ``targets`` with a LoRALinear; returns the trainable LoRA params."""
    params: list[nn.Parameter] = []
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank, alpha, dtype)
                setattr(mod, name, lora)
                params += [lora.lora_A, lora.lora_B]
    return params


def add_enc_lora(model: nn.Module, rank: int = 16, alpha: float | None = None,
                 dtype: torch.dtype | None = None) -> list[nn.Parameter]:
    """Attach a second (encode-phase) adapter to every existing LoRALinear. Returns the trainable enc params."""
    params: list[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params += m.add_enc_adapter(rank, alpha, dtype)
    return params


def set_lora_enabled(model: nn.Module, flag: bool) -> None:
    """Toggle the READ (decode-phase) adapter. The encode adapter is controlled separately."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enabled = flag


def set_enc_lora_enabled(model: nn.Module, flag: bool) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enc_enabled = flag


@contextlib.contextmanager
def lora_disabled(model: nn.Module):
    """Run the fallback/full path on the exact original base, then restore."""
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    prev = [m.enabled for m in mods]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m, p in zip(mods, prev):
            m.enabled = p


def checkpoint_lora_context(model: nn.Module):
    """Return a non-reentrant checkpoint context that preserves adapter routing.

    GCM shares one base and toggles read/encode adapters between forwards. Gradient
    checkpointing recomputes a layer later during backward, after those global flags may
    have changed. Capture the flags when each checkpoint is created and restore them only
    for its recomputation.
    """

    def context_fn():
        mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
        snapshot = [(m.enabled, m.enc_enabled) for m in mods]

        @contextlib.contextmanager
        def recompute_context():
            previous = [(m.enabled, m.enc_enabled) for m in mods]
            for module, (read_enabled, enc_enabled) in zip(mods, snapshot):
                module.enabled = read_enabled
                module.enc_enabled = enc_enabled
            try:
                yield
            finally:
                for module, (read_enabled, enc_enabled) in zip(mods, previous):
                    module.enabled = read_enabled
                    module.enc_enabled = enc_enabled

        return contextlib.nullcontext(), recompute_context()

    return context_fn
