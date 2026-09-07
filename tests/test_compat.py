from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from gcm.compat import enable_torch_linear_attention


def test_torch_linear_attention_disables_all_fused_training_kernels():
    module = SimpleNamespace(
        chunk_gated_delta_rule=object(),
        fused_recurrent_gated_delta_rule=object(),
        FusedRMSNormGated=object(),
        causal_conv1d_fn=object(),
        causal_conv1d_update=object(),
    )
    with patch("gcm.compat.importlib.import_module", return_value=module):
        patched = enable_torch_linear_attention()

    assert len(patched) == 2
    assert module.chunk_gated_delta_rule is None
    assert module.fused_recurrent_gated_delta_rule is None
    assert module.FusedRMSNormGated is None
    assert module.causal_conv1d_fn is None
    assert module.causal_conv1d_update is None
