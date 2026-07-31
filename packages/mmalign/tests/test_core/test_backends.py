"""Tests for mmalign.backends — MockBackend, OpenAIBackend, loader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from mmalign.backends.loader import UnknownProviderError, load_backends
from mmalign.backends.mock import BackendError, MockBackend
from mmalign.core.backend import IncompatibleBackendError
from mmalign.core.result import BackendTelemetry
from mmalign.core.types import (
    GenerationRequest,
    GenerationResult,
    InputContentPart,
    InputMessage,
)


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------

class TestMockBackend:
    @pytest.fixture
    def backend(self) -> MockBackend:
        return MockBackend(response_text="hello world")

    @pytest.mark.asyncio
    async def test_generate_returns_text(self, backend: MockBackend):
        req = GenerationRequest(model="mock", input="test")
        result = await backend.generate(req)
        assert isinstance(result, GenerationResult)
        assert result.text() == "hello world"

    @pytest.mark.asyncio
    async def test_generate_increments_call_count(self, backend: MockBackend):
        req = GenerationRequest(model="mock", input="test")
        await backend.generate(req)
        await backend.generate(req)
        assert backend.call_count == 2

    @pytest.mark.asyncio
    async def test_fail_after(self):
        backend = MockBackend(fail_after=2)
        req = GenerationRequest(model="mock", input="test")
        await backend.generate(req)
        await backend.generate(req)
        with pytest.raises(BackendError, match="simulated failure"):
            await backend.generate(req)

    @pytest.mark.asyncio
    async def test_embed_returns_vectors(self, backend: MockBackend):
        parts = [
            InputContentPart.text_part("hello"),
            InputContentPart.text_part("world"),
        ]
        vecs = await backend.embed(parts)
        assert len(vecs) == 2
        assert len(vecs[0]) == 8  # default embed_dim

    @pytest.mark.asyncio
    async def test_embed_rejects_non_text(self, backend: MockBackend):
        parts = [InputContentPart.image_url_part("https://example.com/img.jpg")]
        with pytest.raises(IncompatibleBackendError):
            await backend.embed(parts)

    @pytest.mark.asyncio
    async def test_score_returns_fixed_value(self, backend: MockBackend):
        req = GenerationRequest(model="mock", input="test")
        result = await backend.generate(req)
        score = await backend.score(req, result)
        assert score == pytest.approx(0.8)

    def test_reset(self, backend: MockBackend):
        import asyncio
        req = GenerationRequest(model="mock", input="test")
        asyncio.run(backend.generate(req))
        assert backend.call_count == 1
        backend.reset()
        assert backend.call_count == 0

    def test_telemetry_for(self, backend: MockBackend):
        req = GenerationRequest(model="mock", input="test")
        telemetry = backend.telemetry_for(req)
        assert isinstance(telemetry, BackendTelemetry)
        assert telemetry.api_style == "mock"
        assert not telemetry.failed

    def test_capabilities(self, backend: MockBackend):
        assert "text" in backend.capabilities.input_modalities
        assert "image" in backend.capabilities.input_modalities
        assert backend.capabilities.supports_system_message

    @pytest.mark.asyncio
    async def test_result_usage(self, backend: MockBackend):
        req = GenerationRequest(model="mock", input="test")
        result = await backend.generate(req)
        assert "input_tokens" in result.usage
        assert result.usage["input_tokens"] == 10


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

MINIMAL_MODELS_YAML = """
models:
  mock:
    provider: mock
    model_name: mock
    api_style: mock
    timeout_s: 5
    max_retries: 0
"""

MULTI_MODELS_YAML = """
models:
  mock:
    provider: mock
    model_name: mock
    api_style: mock

  gpt-test:
    provider: openai
    model_name: gpt-4.1-mini
    api_style: chat_completions
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY_TEST
    timeout_s: 30
    max_retries: 1
    capabilities:
      input_modalities: ["text"]
      output_modalities: ["text"]
      openai_compatible: true
"""


class TestBackendLoader:
    def test_load_mock_from_yaml(self, tmp_path: Path):
        cfg = tmp_path / "models.yaml"
        cfg.write_text(MINIMAL_MODELS_YAML)
        backends = load_backends(cfg)
        assert "mock" in backends
        assert isinstance(backends["mock"], MockBackend)

    def test_load_only_subset(self, tmp_path: Path):
        cfg = tmp_path / "models.yaml"
        cfg.write_text(MULTI_MODELS_YAML)
        backends = load_backends(cfg, only=["mock"])
        assert "mock" in backends
        assert "gpt-test" not in backends

    def test_load_openai_compatible(self, tmp_path: Path):
        cfg = tmp_path / "models.yaml"
        cfg.write_text(MULTI_MODELS_YAML)
        from mmalign.backends.openai import OpenAIBackend
        backends = load_backends(cfg)
        assert isinstance(backends["gpt-test"], OpenAIBackend)

    def test_unknown_provider_raises(self, tmp_path: Path):
        cfg = tmp_path / "models.yaml"
        cfg.write_text("models:\n  bad:\n    provider: unknown_provider\n")
        with pytest.raises(UnknownProviderError, match="unknown_provider"):
            load_backends(cfg)

    def test_api_key_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENAI_API_KEY_TEST", "sk-test-123")
        cfg = tmp_path / "models.yaml"
        cfg.write_text(MULTI_MODELS_YAML)
        from mmalign.backends.openai import OpenAIBackend
        backends = load_backends(cfg, only=["gpt-test"])
        backend = backends["gpt-test"]
        assert isinstance(backend, OpenAIBackend)

    def test_missing_env_uses_placeholder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OPENAI_API_KEY_TEST", raising=False)
        cfg = tmp_path / "models.yaml"
        cfg.write_text(MULTI_MODELS_YAML)
        from mmalign.backends.openai import OpenAIBackend
        backends = load_backends(cfg, only=["gpt-test"])
        assert isinstance(backends["gpt-test"], OpenAIBackend)

    def test_load_actual_models_yaml(self):
        """Verify that the shipped configs/models.yaml is valid."""
        # Walk up to find workspace root where configs/ lives
        for candidate in Path(__file__).parents:
            p = candidate / "configs" / "models.yaml"
            if p.exists():
                yaml_path = p
                break
        else:
            yaml_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
        assert yaml_path.exists(), f"models.yaml not found at {yaml_path}"
        backends = load_backends(yaml_path)
        assert "mock" in backends


# ---------------------------------------------------------------------------
# OpenAI content conversion helpers
# ---------------------------------------------------------------------------

class TestOpenAIContentConversion:
    def test_messages_to_chat_text_only(self):
        from mmalign.backends.openai import _messages_to_chat
        req = GenerationRequest(model="gpt-4.1", input="hello world")
        messages = _messages_to_chat(req)
        assert messages == [{"role": "user", "content": "hello world"}]

    def test_messages_to_chat_image(self):
        from mmalign.backends.openai import _messages_to_chat
        req = GenerationRequest(
            model="gpt-4.1",
            input=[InputMessage(
                role="user",
                content=[
                    InputContentPart.text_part("describe this"),
                    InputContentPart.image_url_part("https://img.example.com/a.jpg"),
                ],
            )],
        )
        messages = _messages_to_chat(req)
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://img.example.com/a.jpg"

    def test_messages_to_responses_input(self):
        from mmalign.backends.openai import _messages_to_responses_input
        req = GenerationRequest(model="gpt-4.1", input="hello")
        inp = _messages_to_responses_input(req)
        assert inp[0]["role"] == "user"
        assert inp[0]["content"][0]["type"] == "input_text"
        assert inp[0]["content"][0]["text"] == "hello"

    def test_unsupported_content_type_raises(self):
        from mmalign.backends.openai import _messages_to_chat
        req = GenerationRequest(
            model="gpt-4.1",
            input=[InputMessage(
                role="user",
                content=[InputContentPart(type="input_audio")],
            )],
        )
        with pytest.raises(IncompatibleBackendError):
            _messages_to_chat(req)
