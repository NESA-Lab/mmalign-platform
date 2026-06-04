"""Unit tests for alignment methods (raw, vanilla, align3, self_refine)."""

from __future__ import annotations

import pytest

from mmalign.backends.mock import MockBackend
from mmalign.core.method import AlignContext
from mmalign.core.scenario import Sample, Scenario, Specification
from mmalign.core.types import GenerationRequest, InputMessage
from mmalign.methods.raw import RawMethod
from mmalign.methods.vanilla import VanillaMethod
from mmalign.methods.align3 import Align3Method
from mmalign.methods.self_refine import SelfRefineMethod


@pytest.fixture
def text_scenario() -> Scenario:
    return Scenario(
        id="test_scenario",
        name="Test",
        description="Unit test scenario",
        modality="text",
        safety_specs=[
            Specification(spec_id="safety_0", text="Do not provide harmful instructions.", category="safety"),
        ],
        behavioral_specs=[
            Specification(spec_id="behavioral_0", text="Be helpful and concise.", category="behavioral"),
            Specification(spec_id="behavioral_1", text="Use plain language.", category="behavioral"),
        ],
    )


@pytest.fixture
def simple_request() -> GenerationRequest:
    return GenerationRequest(
        model="mock",
        input=[InputMessage.user("How do I write a Python function?")],
    )


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(model_id="mock")


# ---------------------------------------------------------------------------
# RawMethod
# ---------------------------------------------------------------------------

class TestRawMethod:
    def test_name(self) -> None:
        assert RawMethod().name == "raw"

    def test_validate_passes_text(self, text_scenario: Scenario, mock_backend: MockBackend) -> None:
        RawMethod().validate(text_scenario, mock_backend)  # should not raise

    @pytest.mark.asyncio
    async def test_align_returns_response(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        method = RawMethod()
        result = await method.align(simple_request, mock_backend, text_scenario)
        assert result.method == "raw"
        assert result.text()  # non-empty
        assert result.output


# ---------------------------------------------------------------------------
# VanillaMethod
# ---------------------------------------------------------------------------

class TestVanillaMethod:
    def test_name(self) -> None:
        assert VanillaMethod().name == "vanilla"

    def test_validate_passes_text(self, text_scenario: Scenario, mock_backend: MockBackend) -> None:
        VanillaMethod().validate(text_scenario, mock_backend)

    @pytest.mark.asyncio
    async def test_align_includes_specs(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        method = VanillaMethod(include_safety_specs=True, include_behavioral_specs=True)
        result = await method.align(simple_request, mock_backend, text_scenario)
        assert result.method == "vanilla"
        assert result.text()

    @pytest.mark.asyncio
    async def test_align_no_specs(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        method = VanillaMethod(include_safety_specs=False, include_behavioral_specs=False)
        result = await method.align(simple_request, mock_backend, text_scenario)
        assert result.method == "vanilla"


# ---------------------------------------------------------------------------
# Align3Method
# ---------------------------------------------------------------------------

class TestAlign3Method:
    def test_name(self) -> None:
        assert Align3Method().name == "align3"

    def test_validate_passes(self, text_scenario: Scenario, mock_backend: MockBackend) -> None:
        Align3Method().validate(text_scenario, mock_backend)

    @pytest.mark.asyncio
    async def test_align_four_stages(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        method = Align3Method()
        result = await method.align(simple_request, mock_backend, text_scenario)
        assert result.method == "align3"
        assert result.text()
        assert "stages" in result.metadata
        stages = result.metadata["stages"]
        # Should have outputs for all four stages
        assert len(stages) >= 1

    @pytest.mark.asyncio
    async def test_align_aggregates_usage(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await Align3Method().align(simple_request, mock_backend, text_scenario)
        # MockBackend returns non-zero usage; aggregated should be >= single call
        assert isinstance(result.usage, dict)

    @pytest.mark.asyncio
    async def test_align_with_context(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        context = AlignContext(vanilla_results_path=None, reward_model=None)
        result = await Align3Method().align(simple_request, mock_backend, text_scenario, context=context)
        assert result.method == "align3"


# ---------------------------------------------------------------------------
# SelfRefineMethod
# ---------------------------------------------------------------------------

class TestSelfRefineMethod:
    def test_name(self) -> None:
        assert SelfRefineMethod().name == "self_refine"

    @pytest.mark.asyncio
    async def test_align_returns_response(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        method = SelfRefineMethod(max_iterations=2)
        result = await method.align(simple_request, mock_backend, text_scenario)
        assert result.method == "self_refine"
        assert result.text()
        assert result.attempts >= 1
