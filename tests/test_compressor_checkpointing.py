from __future__ import annotations

import torch

from gcm.compressor import SelfCompressor


def test_attention_mask_keeps_shape_with_aligned_row_stride():
    compressor = object.__new__(SelfCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.K = 5

    mask = compressor._mask(
        lc=4,
        lq=1,
        conditional=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lp=1,
    )

    assert mask.shape == (1, 1, 11, 11)
    assert mask.stride(2) % 4 == 0
    assert mask[0, 0, 0, 1] < -1e30
    assert mask[0, 0, 10, 0] == 0


def test_unconditional_memory_rows_cannot_read_query():
    compressor = object.__new__(SelfCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.K = 2

    conditional = compressor._mask(3, 2, True, torch.device("cpu"), torch.float32)
    unconditional = compressor._mask(3, 2, False, torch.device("cpu"), torch.float32)

    assert torch.all(conditional[0, 0, 5:7, 3:5] == 0)
    assert torch.all(unconditional[0, 0, 5:7, 3:5] < -1e30)


def test_hard_manifold_norm_is_finite_for_zero_vectors():
    compressor = object.__new__(SelfCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.norm_mode = "hard"
    compressor.register_buffer("embed_scale", torch.tensor(3.0))
    memory = torch.zeros(2, 4, requires_grad=True)

    normalized = compressor._norm_m(memory)
    normalized.sum().backward()

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(memory.grad).all()


def test_hard_manifold_norm_matches_embedding_scale():
    compressor = object.__new__(SelfCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.norm_mode = "hard"
    compressor.register_buffer("embed_scale", torch.tensor(2.5))

    normalized = compressor._norm_m(torch.randn(3, 8))

    assert torch.allclose(
        normalized.norm(dim=-1),
        torch.full((3,), 2.5),
        atol=1e-5,
    )
