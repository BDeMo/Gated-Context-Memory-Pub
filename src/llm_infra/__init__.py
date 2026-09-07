"""llm_infra: shared LLM training and evaluation infra for Plan 08 v0.

See [`CONVENTIONS.md`](../../../CONVENTIONS.md) at the workspace root for the
cross-repo Wrapper contract.
"""

from llm_infra.wrappers import BaseCall, ChunkEncoding, MemoryState, NoOpWrapper, Wrapper

__all__ = [
    "BaseCall",
    "ChunkEncoding",
    "MemoryState",
    "NoOpWrapper",
    "Wrapper",
]
# Submodules `capacity` and `probes_geometry` are imported explicitly by consumers
# (`import llm_infra.capacity`) to avoid pulling heavy deps at package import.

__version__ = "0.0.1"
