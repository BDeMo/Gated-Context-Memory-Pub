"""Architecture-compat helpers.

Some linear-attention models (e.g. Qwen3.5 / gated-delta-rule) ship a fast Triton (`fla`) path whose *backward*
is incorrect on certain GPU+Triton combinations (Hopper + Triton>=3.4, fla #640) and whose drop-in backend
(`tilelang`) can fail to import. These models also include a correct pure-torch fallback that the modeling code
uses automatically when the `fla` symbols are absent. `enable_torch_linear_attention()` nulls those symbols so
the correct (if slower) torch path is used for both forward and backward. Call it BEFORE loading the model.
"""
from __future__ import annotations

import importlib


def enable_torch_linear_attention(verbose: bool = False) -> list[str]:
    """Force linear-attention models onto their pure-torch training path. Returns the modules patched."""
    patched = []
    for mod in ("transformers.models.qwen3_5.modeling_qwen3_5",
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"):
        try:
            m = importlib.import_module(mod)
            m.chunk_gated_delta_rule = None
            m.fused_recurrent_gated_delta_rule = None
            # These FLA/causal-conv kernels are separate from the delta-rule symbol above.
            # Leaving them enabled still routes backward through Triton and breaks saved-tensor
            # CPU offload on Hopper (misaligned-address failures).
            m.FusedRMSNormGated = None
            m.causal_conv1d_fn = None
            m.causal_conv1d_update = None
            patched.append(mod)
        except Exception:  # noqa: BLE001  (module not present in this transformers build)
            pass
    if verbose and patched:
        print(f"[gcm.compat] torch linear-attention enabled for: {', '.join(patched)}")
    return patched
