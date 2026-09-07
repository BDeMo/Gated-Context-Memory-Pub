"""Discover wrappers by importable name.

`load_wrapper("mem_embedding")` imports the package, calls its
`build_wrapper(...)` factory, and returns the resulting `Wrapper`. Each
wrapper repo must expose a top-level `build_wrapper(config: dict, *, model,
tokenizer) -> Wrapper` to participate.
"""

from __future__ import annotations

import importlib
from typing import Any

from llm_infra.wrappers import Wrapper


def load_wrapper(name: str, config: dict[str, Any], **build_kwargs: Any) -> Wrapper:
    """Import a wrapper package and instantiate via its `build_wrapper`."""

    try:
        mod = importlib.import_module(name)
    except ImportError as e:
        raise ImportError(
            f"wrapper package '{name}' not importable. "
            f"Did you `pip install -e ../{name.replace('_', '-')}` first?"
        ) from e

    if not hasattr(mod, "build_wrapper"):
        raise AttributeError(
            f"wrapper package '{name}' must expose a top-level build_wrapper(config, **kwargs)"
        )

    wrapper = mod.build_wrapper(config, **build_kwargs)
    if not isinstance(wrapper, Wrapper):
        raise TypeError(
            f"'{name}.build_wrapper' returned object that does not satisfy the Wrapper protocol: {type(wrapper)}"
        )
    return wrapper
