"""MockBackend — deterministic stub for unit tests and local development.

``MockBackend`` never makes network calls.  All responses are generated from
a configurable template so that alignment method tests can run offline.

Usage
-----
>>> backend = MockBackend(model_id="mock", response_text="Hello!")
>>> result = await backend.generate(request)
>>> result.text()
'Hello!'

To simulate failures, set ``fail_after`` (raise ``BackendError`` after N calls):
>>> backend = MockBackend(fail_after=2)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from mmalign.core.backend import IncompatibleBackendError
from mmalign.core.types import (
    BackendCapabilities,
    GenerationRequest,
    GenerationResult,
    GenerationOutput,
    InputContentPart,
    OutputContentPart,
)
from mmalign.core.result import BackendTelemetry


class BackendError(RuntimeError):
    """Raised when the mock is configured to simulate a failure."""


class MockBackend:
    """Deterministic ModelBackend for tests.

    Parameters
    ----------
    model_id:
        Identifier reported in ``GenerationResult.model``.
    response_text:
        Fixed text returned by ``generate()``.  Defaults to
        ``"[mock response]"``.
    embed_dim:
        Dimension of the fake embedding vectors returned by ``embed()``.
    reward_score:
        Fixed scalar returned by ``score()``.
    fail_after:
        If set, raise ``BackendError`` after this many ``generate()`` calls.
    latency_ms:
        Simulated latency added to each ``generate()`` call (actual sleep NOT
        performed — only reported in telemetry).
    """

    model_id: str
    capabilities: BackendCapabilities

    def __init__(
        self,
        model_id: str = "mock",
        *,
        response_text: str = "[mock response]",
        embed_dim: int = 8,
        reward_score: float = 0.8,
        fail_after: int | None = None,
        latency_ms: float = 5.0,
    ) -> None:
        self.model_id = model_id
        self.capabilities = BackendCapabilities(
            input_modalities={"text", "image"},
            output_modalities={"text"},
            embed_modalities={"text"},
            score_modalities={"text"},
            supports_system_message=True,
            openai_compatible=False,
        )
        self._response_text = response_text
        self._embed_dim = embed_dim
        self._reward_score = reward_score
        self._fail_after = fail_after
        self._latency_ms = latency_ms
        self._call_count = 0

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        request: GenerationRequest,
        **kwargs: Any,
    ) -> GenerationResult:
        self._call_count += 1
        if self._fail_after is not None and self._call_count > self._fail_after:
            raise BackendError(
                f"MockBackend: simulated failure after {self._fail_after} calls"
            )
        return GenerationResult(
            id=f"mock-{uuid.uuid4().hex[:8]}",
            model=self.model_id,
            output=[GenerationOutput(content=[OutputContentPart.text_part(self._response_text)])],
            usage={"input_tokens": 10, "output_tokens": 5},
            metadata={
                "latency_ms": self._latency_ms,
                "call_count": self._call_count,
            },
        )

    async def embed(
        self,
        parts: list[InputContentPart],
    ) -> list[list[float]]:
        for part in parts:
            if part.type not in ("input_text",):
                raise IncompatibleBackendError(
                    f"MockBackend only supports input_text for embed, got {part.type}"
                )
        return [[0.1 * (i % self._embed_dim) for i in range(self._embed_dim)] for _ in parts]

    async def score(
        self,
        request: GenerationRequest,
        response: GenerationResult,
    ) -> float:
        return self._reward_score

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def telemetry_for(self, request: GenerationRequest) -> BackendTelemetry:
        """Return what BackendTelemetry would look like for a generate() call."""
        return BackendTelemetry(
            provider="mock",
            model_name=self.model_id,
            api_style="mock",
            latency_ms=self._latency_ms,
            usage={"input_tokens": 10, "output_tokens": 5},
        )

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset(self) -> None:
        self._call_count = 0
