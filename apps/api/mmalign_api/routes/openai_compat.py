"""OpenAI-compatible façade routes.

POST /v1/chat/completions   — Chat Completions API compatibility layer
POST /v1/responses          — Responses API compatibility layer

Design notes (PRD §F2, TDD §api/routes/openai_compat.py)
---------------------------------------------------------
These routes are an **optional façade** for clients that already use the
OpenAI SDK.  They are NOT the primary MMAlign API; the primary interface is
``POST /api/v1/controller/generate``.

The façade forwards requests through ``ComplianceController`` so aligned
responses are returned in OpenAI-compatible format, enabling drop-in
integration without SDK changes on the caller side.

MMAlign-specific parameters (scenario, method, policy) can be passed via:
* ``model`` field suffix: ``"gpt-4.1:scenario=education"`` (simple override)
* ``extra_body`` / additional JSON keys with the ``mmalign_`` prefix.
* ``X-MMAlign-Scenario`` request header.

If no scenario is specified the controller uses the first available
scenario or falls back to ``education``.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from mmalign_api.deps import get_configs_dir

router = APIRouter(tags=["openai-compat"])

# ---------------------------------------------------------------------------
# Chat Completions request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # MMAlign extensions (optional)
    mmalign_scenario: str | None = None
    mmalign_method: str | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage


# ---------------------------------------------------------------------------
# Responses API request / response models
# ---------------------------------------------------------------------------

class ResponsesRequest(BaseModel):
    model: str
    input: str | list[dict[str, Any]]
    temperature: float | None = None
    max_output_tokens: int | None = None
    # MMAlign extensions (optional)
    mmalign_scenario: str | None = None
    mmalign_method: str | None = None


class ResponseOutput(BaseModel):
    type: str = "message"
    role: str = "assistant"
    content: list[dict[str, Any]]


class ResponsesUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponsesResponse(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    model: str
    output: list[ResponseOutput]
    usage: ResponsesUsage


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_scenario(model: str, mmalign_scenario: str | None, header: str | None) -> str:
    """Resolve the scenario from (in priority order):
    1. Explicit mmalign_scenario field in request body.
    2. X-MMAlign-Scenario request header.
    3. Suffix in model string: ``"gpt-4.1:scenario=education"`` → ``"education"``.
    4. Default: ``"education"``.
    """
    if mmalign_scenario:
        return mmalign_scenario
    if header:
        return header
    if ":" in model:
        for part in model.split(":")[1:]:
            if part.startswith("scenario="):
                return part.split("=", 1)[1]
    return "education"


def _clean_model(model: str) -> str:
    """Strip MMAlign suffix from model string."""
    return model.split(":")[0]


async def _run_controller(
    model_id: str,
    scenario: str,
    input_text: str,
    temperature: float | None,
    configs_dir: Path,
) -> tuple[str, dict[str, int], float]:
    """Run ComplianceController and return (aligned_text, usage, latency_ms)."""
    from mmalign.backends.loader import load_backends
    from mmalign.loaders.policies import load_policy_from_yaml
    from mmalign.loaders.scenarios import load_scenario
    from mmalign.engines.correction.controller import ComplianceController
    from mmalign.core.types import GenerationRequest

    backends = load_backends(configs_dir / "models.yaml")
    backend = backends.get(model_id)
    if backend is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "model_not_found",
                    "message": f"Model '{model_id}' not registered in models.yaml.",
                    "type": "invalid_request_error",
                }
            },
        )

    try:
        loaded_scenario, _ = load_scenario(scenario, sample_limit=0)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "scenario_not_found",
                    "message": f"Scenario '{scenario}' not found.",
                    "type": "invalid_request_error",
                }
            },
        )

    gen_req = GenerationRequest(
        model=model_id,
        input=input_text,
        temperature=temperature,
    )

    policy = load_policy_from_yaml(scenario, configs_dir)
    controller = ComplianceController(backend=backend, scenario=loaded_scenario, policy=policy)
    controlled = await controller.generate(gen_req)

    aligned_text = controlled.response.text()
    if not aligned_text:
        for out in controlled.response.output:
            for part in out.content:
                if getattr(part, "type", None) == "refusal" and part.text:
                    aligned_text = part.text
                    break

    usage = controlled.response.usage or {}
    return aligned_text, usage, controlled.latency_ms


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions", response_model=ChatCompletionsResponse)
async def chat_completions(
    req: ChatCompletionsRequest,
    x_mmalign_scenario: str | None = Header(default=None, alias="X-MMAlign-Scenario"),
    configs_dir: Path = Depends(get_configs_dir),
) -> ChatCompletionsResponse:
    """OpenAI Chat Completions-compatible endpoint with MMAlign compliance control.

    Drop-in replacement for ``POST https://api.openai.com/v1/chat/completions``.
    Set ``base_url="http://localhost:8000"`` in the OpenAI SDK and all
    generation will be routed through the ComplianceController.

    Example with OpenAI SDK::

        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000", api_key="ignored")
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Tell me about biology."}],
        )
    """
    if req.stream:
        raise HTTPException(
            status_code=501,
            detail={
                "error": {
                    "code": "streaming_not_supported",
                    "message": "Streaming is not yet supported by the MMAlign façade.",
                    "type": "invalid_request_error",
                }
            },
        )

    # Extract text from the last user message
    input_text = _extract_text_from_messages(req.messages)
    model_id = _clean_model(req.model)
    scenario = _resolve_scenario(req.model, req.mmalign_scenario, x_mmalign_scenario)

    aligned_text, usage, _latency = await _run_controller(
        model_id=model_id,
        scenario=scenario,
        input_text=input_text,
        temperature=req.temperature,
        configs_dir=configs_dir,
    )

    return ChatCompletionsResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content=aligned_text)
            )
        ],
        usage=ChatUsage(
            prompt_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            completion_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )


# ---------------------------------------------------------------------------
# POST /v1/responses
# ---------------------------------------------------------------------------

@router.post("/v1/responses", response_model=ResponsesResponse)
async def responses(
    req: ResponsesRequest,
    x_mmalign_scenario: str | None = Header(default=None, alias="X-MMAlign-Scenario"),
    configs_dir: Path = Depends(get_configs_dir),
) -> ResponsesResponse:
    """OpenAI Responses API-compatible endpoint with MMAlign compliance control.

    Drop-in replacement for ``POST https://api.openai.com/v1/responses``.

    Example with OpenAI SDK (Responses API)::

        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000", api_key="ignored")
        resp = client.responses.create(
            model="gpt-4.1",
            input="Tell me about biology.",
        )
    """
    # Normalise input to string
    if isinstance(req.input, str):
        input_text = req.input
    elif isinstance(req.input, list):
        # Extract text from first user content part
        input_text = _extract_text_from_messages_raw(req.input)
    else:
        input_text = str(req.input)

    model_id = _clean_model(req.model)
    scenario = _resolve_scenario(req.model, req.mmalign_scenario, x_mmalign_scenario)

    aligned_text, usage, _latency = await _run_controller(
        model_id=model_id,
        scenario=scenario,
        input_text=input_text,
        temperature=req.temperature,
        configs_dir=configs_dir,
    )

    return ResponsesResponse(
        id=f"resp-{uuid.uuid4().hex[:12]}",
        created_at=int(time.time()),
        model=req.model,
        output=[
            ResponseOutput(
                content=[{"type": "output_text", "text": aligned_text}]
            )
        ],
        usage=ResponsesUsage(
            input_tokens=usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            output_tokens=usage.get("output_tokens", usage.get("completion_tokens", 0)),
            total_tokens=usage.get("total_tokens", 0),
        ),
    )


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_messages(messages: list[ChatMessage]) -> str:
    """Extract the last user message's text content."""
    for msg in reversed(messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                return msg.content
            return " ".join(
                part.get("text", "") for part in msg.content
                if isinstance(part, dict) and part.get("type") in ("text", "input_text")
            )
    # Fall back to last assistant message if no user message
    if messages:
        last = messages[-1]
        return last.content if isinstance(last.content, str) else str(last.content)
    return ""


def _extract_text_from_messages_raw(messages: list[dict[str, Any]]) -> str:
    """Extract text from raw OpenAI Responses API input array."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") in ("input_text", "text")
                )
    return str(messages) if messages else ""
