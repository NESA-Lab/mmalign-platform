"""OpenAIBackend — async adapter for OpenAI / Azure / OpenAI-compatible endpoints.

Supports two API styles, selected per model in ``configs/models.yaml``:

``responses`` (default)
    Calls ``POST /responses`` using the OpenAI Responses API.
    ``GenerationRequest`` fields are forwarded directly; ``mmalign`` wrapper
    is stripped before sending.

``chat_completions``
    Calls ``POST /chat/completions``.  The adapter converts
    ``input_text`` content parts to ``{"type": "text", "text": ...}`` and
    ``input_image`` to ``{"type": "image_url", "image_url": {"url": ...}}``.

Backend telemetry
-----------------
``BackendTelemetry`` is populated from the response's ``usage`` object and
``time.perf_counter()`` elapsed time.  On API errors the telemetry captures
the error code and message instead of raising immediately, allowing callers
(DiagnosisEngine) to record partial results.

Retries
-------
Transient errors (429 rate-limit, 5xx) are retried with exponential back-off
up to ``max_retries`` times via the ``openai`` SDK's built-in retry mechanism.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from mmalign.core.backend import IncompatibleBackendError
from mmalign.core.result import BackendTelemetry
from mmalign.core.types import (
    BackendCapabilities,
    GenerationRequest,
    GenerationResult,
    GenerationOutput,
    InputContentPart,
    OutputContentPart,
)

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class OpenAIBackendError(RuntimeError):
    """Raised for non-retryable API errors after exhausting retries."""

    def __init__(self, message: str, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


def _make_chat_content(parts: list[InputContentPart]) -> list[dict[str, Any]]:
    """Convert InputContentPart list to Chat Completions content format."""
    result: list[dict[str, Any]] = []
    for part in parts:
        if part.type == "input_text":
            result.append({"type": "text", "text": part.text or ""})
        elif part.type == "input_image" and part.image_url:
            result.append({
                "type": "image_url",
                "image_url": {
                    "url": part.image_url,
                    "detail": part.detail or "auto",
                },
            })
        elif part.type == "input_image" and part.file_id:
            result.append({
                "type": "image_url",
                "image_url": {"url": f"file-{part.file_id}"},
            })
        else:
            raise IncompatibleBackendError(
                f"chat_completions adapter cannot convert content type '{part.type}'"
            )
    return result


def _messages_to_chat(request: GenerationRequest) -> list[dict[str, Any]]:
    """Build Chat Completions messages list from a GenerationRequest."""
    messages: list[dict[str, Any]] = []
    for msg in request.input_messages():
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        else:
            chat_content = _make_chat_content(msg.content)
            if len(chat_content) == 1 and chat_content[0]["type"] == "text":
                messages.append({"role": msg.role, "content": chat_content[0]["text"]})
            else:
                messages.append({"role": msg.role, "content": chat_content})
    return messages


def _messages_to_responses_input(
    request: GenerationRequest,
) -> list[dict[str, Any]]:
    """Build Responses API input list from a GenerationRequest."""
    result: list[dict[str, Any]] = []
    for msg in request.input_messages():
        if isinstance(msg.content, str):
            result.append({
                "role": msg.role,
                "content": [{"type": "input_text", "text": msg.content}],
            })
        else:
            content: list[dict[str, Any]] = []
            for part in msg.content:
                if part.type == "input_text":
                    content.append({"type": "input_text", "text": part.text or ""})
                elif part.type == "input_image" and part.image_url:
                    content.append({
                        "type": "input_image",
                        "image_url": part.image_url,
                        "detail": part.detail or "auto",
                    })
                elif part.type == "input_image" and part.file_id:
                    content.append({
                        "type": "input_image",
                        "file_id": part.file_id,
                        "detail": part.detail or "auto",
                    })
                else:
                    raise IncompatibleBackendError(
                        f"responses adapter cannot convert content type '{part.type}'"
                    )
            result.append({"role": msg.role, "content": content})
    return result


class OpenAIBackend:
    """Async ModelBackend for OpenAI / OpenAI-compatible endpoints.

    Parameters
    ----------
    model_id:
        Canonical model ID used in ``GenerationResult.model``.
    model_name:
        Model name forwarded to the API (may differ from ``model_id``).
    api_style:
        ``"responses"`` or ``"chat_completions"``.
    base_url:
        API base URL.  Defaults to ``https://api.openai.com/v1``.
    api_key:
        API key.  Prefer reading from environment via ``api_key_env``.
    timeout_s:
        Per-request timeout in seconds.
    max_retries:
        Passed to ``openai.AsyncOpenAI(max_retries=...)``.
    capabilities:
        Static capability declaration.  Defaults to text-only.
    """

    model_id: str
    capabilities: BackendCapabilities

    def __init__(
        self,
        model_id: str,
        model_name: str | None = None,
        *,
        api_style: str = "chat_completions",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "placeholder",
        timeout_s: float = 60.0,
        max_retries: int = 2,
        capabilities: BackendCapabilities | None = None,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "The 'openai' package is required for OpenAIBackend. "
                "Install it with: uv add openai"
            )
        self.model_id = model_id
        self._model_name = model_name or model_id
        self._api_style = api_style
        self._timeout_s = timeout_s
        self.capabilities = capabilities or BackendCapabilities(
            input_modalities={"text"},
            output_modalities={"text"},
            openai_compatible=True,
        )
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        request: GenerationRequest,
        **kwargs: Any,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        try:
            if self._api_style == "responses":
                return await self._generate_responses(request, **kwargs)
            else:
                return await self._generate_chat(request, **kwargs)
        except openai.APIStatusError as exc:
            raise OpenAIBackendError(
                f"API error {exc.status_code}: {exc.message}",
                error_code=str(exc.status_code),
            ) from exc
        except openai.APIConnectionError as exc:
            raise OpenAIBackendError(
                f"Connection error: {exc}",
                error_code="connection_error",
            ) from exc

    async def _generate_responses(
        self, request: GenerationRequest, **kwargs: Any
    ) -> GenerationResult:
        """Call the Responses API."""
        t0 = time.perf_counter()
        params: dict[str, Any] = {
            "model": self._model_name,
            "input": _messages_to_responses_input(request),
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            params["max_output_tokens"] = request.max_output_tokens
        params.update(kwargs)

        resp = await self._client.responses.create(**params)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        output_parts: list[OutputContentPart] = []
        for item in getattr(resp, "output", []):
            for part in getattr(item, "content", []):
                ptype = getattr(part, "type", "")
                if ptype == "output_text":
                    output_parts.append(OutputContentPart.text_part(part.text))
                elif ptype == "refusal":
                    output_parts.append(OutputContentPart.refusal_part(part.text or ""))

        usage_raw = getattr(resp, "usage", None)
        usage: dict[str, int] = {}
        if usage_raw:
            usage = {
                "input_tokens": getattr(usage_raw, "input_tokens", 0),
                "output_tokens": getattr(usage_raw, "output_tokens", 0),
            }

        return GenerationResult(
            id=getattr(resp, "id", str(uuid.uuid4())),
            model=self._model_name,
            output=[GenerationOutput(content=output_parts)],
            usage=usage,
            metadata={"latency_ms": latency_ms, "api_style": "responses"},
        )

    async def _generate_chat(
        self, request: GenerationRequest, **kwargs: Any
    ) -> GenerationResult:
        """Call the Chat Completions API."""
        t0 = time.perf_counter()
        params: dict[str, Any] = {
            "model": self._model_name,
            "messages": _messages_to_chat(request),
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            params["max_tokens"] = request.max_output_tokens
        params.update(kwargs)

        resp = await self._client.chat.completions.create(**params)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        output_parts: list[OutputContentPart] = []
        for choice in resp.choices:
            msg = choice.message
            if msg.refusal:
                output_parts.append(OutputContentPart.refusal_part(msg.refusal))
            elif msg.content:
                output_parts.append(OutputContentPart.text_part(msg.content))

        usage_raw = resp.usage
        usage: dict[str, int] = {}
        if usage_raw:
            usage = {
                "input_tokens": usage_raw.prompt_tokens,
                "output_tokens": usage_raw.completion_tokens,
            }

        return GenerationResult(
            id=resp.id,
            model=self._model_name,
            output=[GenerationOutput(content=output_parts)],
            usage=usage,
            metadata={"latency_ms": latency_ms, "api_style": "chat_completions"},
        )

    async def embed(
        self,
        parts: list[InputContentPart],
    ) -> list[list[float]]:
        texts = [p.text or "" for p in parts if p.type == "input_text"]
        if len(texts) != len(parts):
            raise IncompatibleBackendError(
                "OpenAIBackend.embed only supports input_text parts"
            )
        resp = await self._client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    async def score(
        self,
        request: GenerationRequest,
        response: GenerationResult,
    ) -> float:
        raise IncompatibleBackendError(
            f"OpenAIBackend({self.model_id}) does not support reward scoring. "
            "Use a dedicated reward model backend."
        )
