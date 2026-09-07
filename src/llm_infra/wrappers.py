"""Wrapper contract shared by mem-embedding, mem-feature, mem-weight.

This is the single source of truth for what a memory-wrapper module must
implement. Keeping it minimal (3 methods + 3 dataclasses) is intentional:
the more the contract grows, the more axes risk diverging in incompatible
ways and the 4-baseline harness loses comparability.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

import torch


@dataclass
class MemoryState:
    """Opaque memory carrier.

    The wrapper produces and consumes its own `MemoryState` shape; the harness
    treats it as a black box.

    `extra` is a free-form dict where wrappers expose auxiliary tensors for
    probes / analytics. Typical keys:

    - ``"last_delta"``: the Δm_t residual that produced the current state
      (`m_t = m_{t-1} + α·Δm_t`). Shape matches `payload`.
    - ``"alpha"``: the (detached) update scale used at the latest step.
    - ``"step"``: integer chunk index since `init_memory`.

    Probes that don't find a key should fail silently — `extra` is opt-in.
    """

    payload: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkEncoding:
    """Encoder output for a single chunk.

    `hidden` is `[batch, seq, d]`. `mask` is `[batch, seq]` with `1` for valid
    tokens and `0` for padding. Pooling, if needed, is the wrapper's call.
    """

    hidden: torch.Tensor
    mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.hidden.shape[0]

    @property
    def hidden_size(self) -> int:
        return self.hidden.shape[-1]


@dataclass
class BaseCall:
    """A generation-time call into the frozen base model.

    Wrapper.apply mutates this in-place or returns a new one; the harness then
    calls the base model with the resulting tensors.

    Either `input_ids` or `inputs_embeds` must be set, not both. The harness
    follows the same convention as HuggingFace `transformers`.
    """

    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    inputs_embeds: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def assert_consistent(self) -> None:
        if (self.input_ids is None) == (self.inputs_embeds is None):
            raise ValueError(
                "BaseCall must have exactly one of input_ids or inputs_embeds set."
            )


@runtime_checkable
class Wrapper(Protocol):
    """Contract for memory wrappers.

    Required: `init_memory`, `update`, `apply`.

    `apply` must be reversible (use `apply_scope` if registering hooks /
    patching weights). Subsequent calls expect the base model to be untouched.
    """

    def init_memory(self, batch: int, device: torch.device) -> MemoryState: ...

    def update(self, memory: MemoryState, chunk: ChunkEncoding) -> MemoryState:
        """One step of the memory recurrence. Trainable. Differentiable."""
        ...

    def apply(self, base_call: BaseCall, memory: MemoryState) -> BaseCall:
        """Wire memory into the base-model call.

        Implementations that install hooks or patch weights must register
        their cleanup in `base_call.extra["_cleanup"]` so the harness can call
        it after the base model runs.
        """
        ...


@contextmanager
def apply_scope(wrapper: Wrapper, base_call: BaseCall, memory: MemoryState) -> Iterator[BaseCall]:
    """Apply a wrapper to a base call inside a scope; restore on exit.

    This is the harness-side helper that guarantees cleanup runs even when
    the wrapped base call raises.
    """

    applied = wrapper.apply(base_call, memory)
    cleanup = applied.extra.pop("_cleanup", None)
    try:
        yield applied
    finally:
        if cleanup is not None:
            cleanup()


class NoOpWrapper:
    """A wrapper that does nothing. Used as the baseline reference + unit test fixture.

    `update` returns the same memory; `apply` returns the call unchanged. Useful
    for verifying the harness wiring without any actual memory mechanism.
    """

    def init_memory(self, batch: int, device: torch.device) -> MemoryState:
        del batch, device
        return MemoryState(payload=None)

    def update(self, memory: MemoryState, chunk: ChunkEncoding) -> MemoryState:
        del chunk
        return memory

    def apply(self, base_call: BaseCall, memory: MemoryState) -> BaseCall:
        del memory
        return base_call
