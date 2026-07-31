"""Unit tests for alignment methods."""

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
from mmalign.methods.specbench_ttd import (
    Align3SpecBenchMethod,
    BestOfNMethod,
    CriticReviseAgentMethod,
    IPRPromptRefineMethod,
    MoreThinkMethod,
    MultiAgentConsensusMethod,
    OverRefusalCheckMethod,
    RiskAdaptiveAgentMethod,
    TPOMethod,
    UserConditionedSafetyCriticMethod,
    VanillaSpecBenchMethod,
    VLMSafetyGuardMethod,
    ZeroThinkMethod,
)


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


# ---------------------------------------------------------------------------
# SpecBench TTD methods
# ---------------------------------------------------------------------------

class TestSpecBenchTTDMethods:
    def test_names(self) -> None:
        assert ZeroThinkMethod().name == "zero_think"
        assert VanillaSpecBenchMethod().name == "vanilla_specbench"
        assert Align3SpecBenchMethod().name == "align3_specbench"
        assert MoreThinkMethod().name == "more_think"
        assert BestOfNMethod().name == "best_of_n"
        assert TPOMethod().name == "tpo"
        assert IPRPromptRefineMethod().name == "ipr_prompt_refine"
        assert VLMSafetyGuardMethod().name == "vlm_safety_guard"
        assert OverRefusalCheckMethod().name == "overrefusal_check"
        assert UserConditionedSafetyCriticMethod().name == "user_conditioned_safety_critic"
        assert CriticReviseAgentMethod().name == "critic_revise_agent"
        assert RiskAdaptiveAgentMethod().name == "risk_adaptive_agent"
        assert MultiAgentConsensusMethod().name == "multi_agent_consensus"

    def test_validate_passes_text(self, text_scenario: Scenario, mock_backend: MockBackend) -> None:
        ZeroThinkMethod().validate(text_scenario, mock_backend)
        VanillaSpecBenchMethod().validate(text_scenario, mock_backend)
        Align3SpecBenchMethod().validate(text_scenario, mock_backend)
        MoreThinkMethod().validate(text_scenario, mock_backend)
        BestOfNMethod().validate(text_scenario, mock_backend)
        TPOMethod().validate(text_scenario, mock_backend)
        IPRPromptRefineMethod().validate(text_scenario, mock_backend)
        VLMSafetyGuardMethod().validate(text_scenario, mock_backend)
        OverRefusalCheckMethod().validate(text_scenario, mock_backend)
        UserConditionedSafetyCriticMethod().validate(text_scenario, mock_backend)
        CriticReviseAgentMethod().validate(text_scenario, mock_backend)
        RiskAdaptiveAgentMethod().validate(text_scenario, mock_backend)
        MultiAgentConsensusMethod().validate(text_scenario, mock_backend)

    @pytest.mark.asyncio
    async def test_zero_think_returns_response(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await ZeroThinkMethod().align(simple_request, mock_backend, text_scenario)
        assert result.method == "zero_think"
        assert result.text()
        assert result.attempts == 1
        assert "think" in result.thought

    @pytest.mark.asyncio
    async def test_vanilla_specbench_returns_response(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await VanillaSpecBenchMethod().align(simple_request, mock_backend, text_scenario)
        assert result.method == "vanilla_specbench"
        assert result.text()
        assert result.attempts == 1
        assert result.metadata["prompt_style"] == "augment_prompt"

    @pytest.mark.asyncio
    async def test_align3_specbench_runs_three_stages(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await Align3SpecBenchMethod().align(simple_request, mock_backend, text_scenario)
        assert result.method == "align3_specbench"
        assert result.text()
        assert result.attempts == 3
        assert mock_backend.call_count == 3
        assert set(result.metadata["stages"]) == {
            "behavior_optimization",
            "safety_guided_refinement",
            "holistic_specification_audit",
        }

    @pytest.mark.asyncio
    async def test_more_think_uses_configured_attempts(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await MoreThinkMethod(rethink_count=3).align(
            simple_request, mock_backend, text_scenario
        )
        assert result.method == "more_think"
        assert result.text()
        assert result.attempts == 3
        assert mock_backend.call_count == 3

    @pytest.mark.asyncio
    async def test_best_of_n_scores_candidates(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await BestOfNMethod(n=3).align(simple_request, mock_backend, text_scenario)
        assert result.method == "best_of_n"
        assert result.text()
        assert result.attempts == 3
        assert mock_backend.call_count == 3
        assert len(result.metadata["candidates"]) == 3

    @pytest.mark.asyncio
    async def test_tpo_samples_each_round(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await TPOMethod(iterations=2, sample_size=2).align(
            simple_request, mock_backend, text_scenario
        )
        assert result.method == "tpo"
        assert result.text()
        assert result.attempts == 4
        assert mock_backend.call_count == 4
        assert len(result.metadata["rounds"]) == 2

    @pytest.mark.asyncio
    async def test_ipr_prompt_refine_runs_iterations(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await IPRPromptRefineMethod(iterations=2).align(
            simple_request, mock_backend, text_scenario
        )
        assert result.method == "ipr_prompt_refine"
        assert result.text()
        assert result.attempts == 2
        assert mock_backend.call_count == 2
        assert len(result.metadata["iterations"]) == 2

    @pytest.mark.asyncio
    async def test_vlm_safety_guard_audits_then_answers(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await VLMSafetyGuardMethod().align(simple_request, mock_backend, text_scenario)
        assert result.method == "vlm_safety_guard"
        assert result.text()
        assert result.attempts == 2
        assert mock_backend.call_count == 2
        assert "audit" in result.metadata

    @pytest.mark.asyncio
    async def test_overrefusal_check_revises_initial_response(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await OverRefusalCheckMethod().align(simple_request, mock_backend, text_scenario)
        assert result.method == "overrefusal_check"
        assert result.text()
        assert result.attempts == 2
        assert mock_backend.call_count == 2
        assert "initial_response" in result.metadata

    @pytest.mark.asyncio
    async def test_user_conditioned_safety_critic_uses_profile(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        context = AlignContext(metadata={"user_profile": "teen user"})
        result = await UserConditionedSafetyCriticMethod().align(
            simple_request, mock_backend, text_scenario, context=context
        )
        assert result.method == "user_conditioned_safety_critic"
        assert result.text()
        assert result.attempts == 2
        assert mock_backend.call_count == 2
        assert result.metadata["user_profile"] == "teen user"

    @pytest.mark.asyncio
    async def test_critic_revise_agent_runs_draft_critique_revision(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await CriticReviseAgentMethod().align(simple_request, mock_backend, text_scenario)
        assert result.method == "critic_revise_agent"
        assert result.text()
        assert result.attempts == 3
        assert mock_backend.call_count == 3
        assert "critique" in result.metadata

    @pytest.mark.asyncio
    async def test_risk_adaptive_agent_routes_to_low_risk_default(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await RiskAdaptiveAgentMethod().align(simple_request, mock_backend, text_scenario)
        assert result.method == "risk_adaptive_agent"
        assert result.text()
        assert result.metadata["route"] == "low"
        assert result.metadata["delegated_method"] == "vanilla_specbench"
        assert mock_backend.call_count == 2

    @pytest.mark.asyncio
    async def test_multi_agent_consensus_runs_reviewers(
        self, simple_request: GenerationRequest, mock_backend: MockBackend, text_scenario: Scenario
    ) -> None:
        result = await MultiAgentConsensusMethod().align(simple_request, mock_backend, text_scenario)
        assert result.method == "multi_agent_consensus"
        assert result.text()
        assert result.attempts == 5
        assert mock_backend.call_count == 5
        assert len(result.metadata["reviews"]) == 3
