"""OpenAI-compatible multimodal I/O data models.

MMAlign reuses the OpenAI Responses API schema for all generation, alignment,
evaluation, and controller endpoints. MMAlign-specific parameters are isolated
in the ``mmalign`` wrapper field and are never forwarded to upstream APIs.

MVP output promise: ``output_text`` and ``refusal`` only.
``output_audio`` / ``output_image`` are reserved and require explicit backend
capability declaration; otherwise requests fail fast with UNSUPPORTED_MODALITY.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Primitive literals
# ---------------------------------------------------------------------------

Modality = Literal["text", "image", "video", "audio"]
ReviewVerdict = Literal["YES", "NO", "NA"]
InputContentType = Literal["input_text", "input_image", "input_file", "input_audio"]
OutputContentType = Literal["output_text", "output_image", "output_audio", "refusal"]
ImageDetail = Literal["low", "high", "auto", "original"]


def _text_modality_set() -> set[Modality]:
    return {"text"}


# ---------------------------------------------------------------------------
# Input content parts (OpenAI-compatible)
# ---------------------------------------------------------------------------

class InputAudio(BaseModel):
    """Base64-encoded audio with format hint."""
    data: str
    format: Literal["wav", "mp3", "flac", "aac", "opus"]


class InputContentPart(BaseModel):
    """A single piece of multimodal input content.

    All fields except ``type`` are optional; only the fields relevant to the
    declared ``type`` should be populated. Adapters must reject unsupported
    fields per backend capability.
    """
    type: InputContentType
    text: str | None = None
    image_url: str | None = None
    file_id: str | None = None
    filename: str | None = None
    file_data: str | None = None
    detail: ImageDetail | None = "auto"
    input_audio: InputAudio | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def text_part(cls, text: str) -> "InputContentPart":
        return cls(type="input_text", text=text)

    @classmethod
    def image_url_part(cls, url: str, detail: ImageDetail = "auto") -> "InputContentPart":
        return cls(type="input_image", image_url=url, detail=detail)


# ---------------------------------------------------------------------------
# Output content parts
# ---------------------------------------------------------------------------

class OutputContentPart(BaseModel):
    """A single piece of model output content.

    MVP guarantees only ``output_text`` and ``refusal``.
    """
    type: OutputContentType
    text: str | None = None
    image_url: str | None = None
    file_id: str | None = None
    audio: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def text_part(cls, text: str) -> "OutputContentPart":
        return cls(type="output_text", text=text)

    @classmethod
    def refusal_part(cls, reason: str) -> "OutputContentPart":
        return cls(type="refusal", text=reason)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class InputMessage(BaseModel):
    """A single turn in a multi-turn conversation."""
    role: Literal["system", "developer", "user", "assistant"]
    content: list[InputContentPart] | str

    @classmethod
    def user(cls, text: str) -> "InputMessage":
        return cls(role="user", content=[InputContentPart.text_part(text)])

    @classmethod
    def system(cls, text: str) -> "InputMessage":
        return cls(role="system", content=[InputContentPart.text_part(text)])


# ---------------------------------------------------------------------------
# MMAlign request wrapper
# ---------------------------------------------------------------------------

class MMAlignRequestOptions(BaseModel):
    """MMAlign-specific parameters, isolated from the OpenAI-compatible payload."""
    scenario: str | None = None
    method: str | None = None
    evaluator: str | None = None
    policy: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generation request / result
# ---------------------------------------------------------------------------

class GenerationRequest(BaseModel):
    """OpenAI Responses-compatible generation request.

    ``input`` accepts either a plain string (converted to a single user text
    message by adapters) or a structured ``list[InputMessage]``.
    ``mmalign`` carries alignment/evaluation options and is never forwarded
    to upstream APIs.
    """
    model: str
    input: str | list[InputMessage]
    modalities: list[Literal["text", "audio"]] | None = None
    audio: dict[str, Any] | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    mmalign: MMAlignRequestOptions | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def input_messages(self) -> list[InputMessage]:
        """Normalise ``input`` to a list of messages."""
        if isinstance(self.input, str):
            return [InputMessage.user(self.input)]
        return self.input


class GenerationOutput(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[OutputContentPart]


class GenerationResult(BaseModel):
    id: str
    model: str
    output: list[GenerationOutput]
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def text(self) -> str:
        """Return concatenated output_text content across all output parts."""
        parts: list[str] = []
        for out in self.output:
            for part in out.content:
                if part.type == "output_text" and part.text:
                    parts.append(part.text)
        return "".join(parts)


# ---------------------------------------------------------------------------
# Backend capability declaration
# ---------------------------------------------------------------------------

class BackendCapabilities(BaseModel):
    """Declares what a ModelBackend can and cannot do.

    Used by Runners and Controllers for fast compatibility checks before
    dispatching large batches.
    """
    input_modalities: set[Modality]
    output_modalities: set[Modality] = Field(default_factory=_text_modality_set)
    embed_modalities: set[Modality] = Field(default_factory=_text_modality_set)
    score_modalities: set[Modality] = Field(default_factory=_text_modality_set)
    supports_system_message: bool = True
    supports_developer_message: bool = False
    openai_compatible: bool = False
    max_output_tokens: int | None = None
