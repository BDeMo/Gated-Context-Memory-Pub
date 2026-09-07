"""mem_embedding: Plan 08 v0 mem-X axis.

Exports `MemoryEmbeddingWrapper` plus a `build_wrapper(config, **kwargs)`
factory so `llm_infra.registry.load_wrapper("mem_embedding", cfg)` works.
"""

from mem_embedding.wrapper import MemoryEmbeddingWrapper, build_wrapper

__all__ = ["MemoryEmbeddingWrapper", "build_wrapper"]

__version__ = "0.0.1"
