"""ModelBackend Protocol — uniform interface for all model inference endpoints.

Implementations live in ``mmalign/backends/`` (added in Stage 3).
This module only defines the Protocol and related data types so that Core
layer modules can reference them without circular imports.

Supported backend providers (planned):
  - ``openai``           : OpenAI API / Azure OpenAI (Chat Completions + Responses)
  - ``openai_compatible``: Any OpenAI-compatible server (vLLM, Ollama, Pujiang, …)
  - ``vllm``             : Local vLLM server
  - ``huggingface``      : HuggingFace Transformers pipeline
  - ``mock``             : Deterministic stub for unit tests
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import (
    BackendCapabilities,
    GenerationRequest,
    GenerationResult,
    InputContentPart,
)


class IncompatibleBackendError(ValueError):
    """Raised when a request cannot be satisfied by the backend's capabilities."""


@runtime_checkable
class ModelBackend(Protocol):
    """Async Protocol for model inference, embedding, and reward scoring.

    Implementations must be safe to call concurrently; the Runner controls
    concurrency externally via asyncio semaphores.

    Lifecycle
    ---------
    1. Constructed with provider-specific config (api_key, base_url, …).
    2. ``capabilities`` property queried once by Runner / Controller before
       bulk dispatch.
    3. ``generate`` / ``embed`` / ``score`` called per-sample.
    """

    model_id: str
    """Canonical model identifier (e.g. ``"gpt-4.1"``, ``"qwen3-32b"``)."""

    capabilities: BackendCapabilities
    """Static declaration of what this backend can handle."""

    async def generate(
        self,
        request: GenerationRequest,
        **kwargs: object,
    ) -> GenerationResult:
        """Generate a response for the given request.

        ``**kwargs`` allows adapters to accept provider-specific overrides
        (e.g. ``extra_body`` for vLLM chat template kwargs) without changing
        the Protocol signature.
        """
        ...

    async def embed(
        self,
        parts: list[InputContentPart],
    ) -> list[list[float]]:
        """Return dense vector embeddings for the given content parts.

        Used by value-embedding alignment methods (``value_embed``).
        Raises ``IncompatibleBackendError`` if the backend does not support
        embeddings for the requested modality.
        """
        ...

    async def score(
        self,
        request: GenerationRequest,
        response: GenerationResult,
    ) -> float:
        """Return a scalar reward score for a (prompt, response) pair.

        Used by ``best_of_n`` and ``tpo`` alignment methods.
        Raises ``IncompatibleBackendError`` if the backend is not a reward model.
        """
        ...
