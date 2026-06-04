"""Tests for mmalign.core.types — OpenAI-compatible I/O models."""

import pytest
from mmalign.core.types import (
    BackendCapabilities,
    GenerationRequest,
    GenerationResult,
    GenerationOutput,
    InputContentPart,
    InputMessage,
    MMAlignRequestOptions,
    OutputContentPart,
)


class TestInputContentPart:
    def test_text_part_factory(self):
        p = InputContentPart.text_part("hello")
        assert p.type == "input_text"
        assert p.text == "hello"

    def test_image_url_factory(self):
        p = InputContentPart.image_url_part("https://example.com/img.jpg", detail="high")
        assert p.type == "input_image"
        assert p.image_url == "https://example.com/img.jpg"
        assert p.detail == "high"

    def test_serialisation_round_trip(self):
        p = InputContentPart.text_part("round-trip")
        data = p.model_dump()
        restored = InputContentPart.model_validate(data)
        assert restored == p


class TestInputMessage:
    def test_user_factory(self):
        m = InputMessage.user("what is 2+2?")
        assert m.role == "user"
        assert len(m.content) == 1  # type: ignore[arg-type]
        assert m.content[0].text == "what is 2+2?"  # type: ignore[index]

    def test_system_factory(self):
        m = InputMessage.system("You are helpful.")
        assert m.role == "system"

    def test_string_content_accepted(self):
        m = InputMessage(role="user", content="plain string")
        assert m.content == "plain string"


class TestGenerationRequest:
    def test_string_input_normalised_to_messages(self):
        req = GenerationRequest(model="gpt-4.1", input="hi")
        msgs = req.input_messages()
        assert len(msgs) == 1
        assert msgs[0].role == "user"

    def test_message_list_unchanged(self):
        msgs = [InputMessage.user("hello"), InputMessage.user("world")]
        req = GenerationRequest(model="gpt-4.1", input=msgs)
        assert req.input_messages() == msgs

    def test_mmalign_options_round_trip(self):
        req = GenerationRequest(
            model="qwen3-32b",
            input="test",
            mmalign=MMAlignRequestOptions(scenario="education", method="align3"),
        )
        data = req.model_dump()
        restored = GenerationRequest.model_validate(data)
        assert restored.mmalign is not None
        assert restored.mmalign.scenario == "education"
        assert restored.mmalign.method == "align3"


class TestGenerationResult:
    def _make_result(self, text: str) -> GenerationResult:
        return GenerationResult(
            id="res-001",
            model="gpt-4.1",
            output=[GenerationOutput(content=[OutputContentPart.text_part(text)])],
        )

    def test_text_extraction(self):
        result = self._make_result("Hello world")
        assert result.text() == "Hello world"

    def test_empty_output(self):
        result = GenerationResult(id="r", model="m", output=[])
        assert result.text() == ""


class TestBackendCapabilities:
    def test_defaults(self):
        caps = BackendCapabilities(input_modalities={"text"})
        assert "text" in caps.output_modalities
        assert caps.supports_system_message is True
        assert caps.openai_compatible is False
