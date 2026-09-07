from __future__ import annotations

import torch

from gcm.compressor import SelfCompressor


def _stub_compressor(state_cap: int):
    compressor = SelfCompressor.__new__(SelfCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.recur = True
    compressor.state_cap = state_cap
    compressor.xchunk = 0
    prefix_lengths: list[int] = []

    def encode_raw(ctx, _query, _conditional, prefix=None):
        prefix_lengths.append(0 if prefix is None else int(prefix.shape[1]))
        value = float(ctx[0, 0])
        return torch.full((1, 2, 1), value)

    compressor._encode_raw = encode_raw
    compressor._project = lambda value: value
    return compressor, prefix_lengths


def test_recurrent_concat_preserves_legacy_s_times_k_state():
    compressor, prefix_lengths = _stub_compressor(state_cap=0)
    result = compressor.encode_chunked(
        torch.arange(6).view(1, 6),
        torch.empty((1, 0), dtype=torch.long),
        chunk_size=2,
    )

    assert prefix_lengths == [0, 2, 4]
    assert result["memory"].shape == (1, 6, 1)
    assert result["n_chunks"] == 3
    assert result["state_cap"] == 0


def test_recurrent_state_cap_bounds_prefix_and_reader_memory():
    compressor, prefix_lengths = _stub_compressor(state_cap=3)
    result = compressor.encode_chunked(
        torch.arange(6).view(1, 6),
        torch.empty((1, 0), dtype=torch.long),
        chunk_size=2,
    )

    assert prefix_lengths == [0, 2, 3]
    assert result["memory"].shape == (1, 3, 1)
    assert result["memory"].flatten().tolist() == [2.0, 4.0, 4.0]
    assert result["n_chunks"] == 3
    assert result["state_cap"] == 3
