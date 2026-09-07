from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from gcm.lora import LoRALinear, checkpoint_lora_context, lora_disabled


def test_lora_disabled_is_exact_and_restores_route():
    base = torch.nn.Linear(4, 3)
    wrapped = LoRALinear(base, r=2)
    with torch.no_grad():
        wrapped.lora_B.fill_(0.5)
    x = torch.randn(5, 4)

    adapted = wrapped(x)
    with lora_disabled(wrapped):
        fallback = wrapped(x)
        assert torch.equal(fallback, base(x))

    assert wrapped.enabled is True
    assert not torch.equal(adapted, fallback)


def test_encode_adapter_is_independent_from_read_adapter():
    wrapped = LoRALinear(torch.nn.Linear(4, 4), r=2)
    wrapped.add_enc_adapter(r=2)
    with torch.no_grad():
        wrapped.enc_B.fill_(0.25)
    x = torch.randn(2, 4)

    wrapped.enabled = False
    wrapped.enc_enabled = False
    base_out = wrapped(x)
    wrapped.enc_enabled = True
    enc_out = wrapped(x)

    assert not torch.equal(base_out, enc_out)
    assert wrapped.enabled is False


def test_checkpoint_recompute_uses_forward_adapter_flags():
    wrapped = LoRALinear(torch.nn.Linear(4, 4), r=2)
    model = torch.nn.Sequential(wrapped)
    context_fn = checkpoint_lora_context(model)

    wrapped.enabled = False
    wrapped.enc_enabled = True
    _, recompute = context_fn()

    wrapped.enabled = True
    wrapped.enc_enabled = False
    with recompute:
        assert wrapped.enabled is False
        assert wrapped.enc_enabled is True

    assert wrapped.enabled is True
    assert wrapped.enc_enabled is False


def test_checkpoint_recompute_restores_flags_after_error():
    wrapped = LoRALinear(torch.nn.Linear(4, 4), r=2)
    model = torch.nn.Sequential(wrapped)
    context_fn = checkpoint_lora_context(model)

    wrapped.enabled = False
    _, recompute = context_fn()
    wrapped.enabled = True
    try:
        with recompute:
            raise RuntimeError("test")
    except RuntimeError:
        pass

    assert wrapped.enabled is True


def test_non_reentrant_checkpoint_can_recompute_after_route_toggle():
    wrapped = LoRALinear(torch.nn.Linear(4, 4), r=2)
    model = torch.nn.Sequential(wrapped)
    x = torch.randn(3, 4, requires_grad=True)

    wrapped.enabled = False
    out = checkpoint(
        model,
        x,
        use_reentrant=False,
        context_fn=checkpoint_lora_context(model),
    )
    wrapped.enabled = True
    out.sum().backward()

    assert x.grad is not None
    assert wrapped.enabled is True
