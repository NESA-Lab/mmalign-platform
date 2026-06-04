"""Unit tests for evaluators (mock_judge, rule_based, llm_judge)."""

from __future__ import annotations

import pytest

from mmalign.backends.mock import MockBackend
from mmalign.core.evaluator import EvalResult
from mmalign.core.scenario import Scenario, Specification
from mmalign.core.types import GenerationRequest, GenerationResult, GenerationOutput, OutputContentPart
from mmalign.evaluators.mock_judge import MockJudgeEvaluator
from mmalign.evaluators.rule_based import RuleBasedEvaluator
from mmalign.evaluators.llm_judge import LLMJudgeEvaluator


@pytest.fixture
def text_scenario() -> Scenario:
    return Scenario(
        id="eval_test",
        name="Eval Test",
        description="Unit test scenario",
        modality="text",
        safety_specs=[
            Specification(spec_id="safety_0", text="Do not provide harmful content.", category="safety"),
        ],
        behavioral_specs=[
            Specification(spec_id="behavioral_0", text="Be helpful.", category="behavioral"),
            Specification(spec_id="behavioral_1", text="Be concise.", category="behavioral"),
        ],
    )


@pytest.fixture
def simple_request() -> GenerationRequest:
    return GenerationRequest(model="mock", input="How do I bake a cake?")


def _make_response(text: str) -> GenerationResult:
    return GenerationResult(
        id="test-001",
        model="mock",
        output=[GenerationOutput(content=[OutputContentPart.text_part(text)])],
    )


# ---------------------------------------------------------------------------
# MockJudgeEvaluator
# ---------------------------------------------------------------------------

class TestMockJudgeEvaluator:
    @pytest.mark.asyncio
    async def test_returns_eval_result(
        self, simple_request: GenerationRequest, text_scenario: Scenario
    ) -> None:
        evaluator = MockJudgeEvaluator(deterministic=True)
        response = _make_response("To bake a cake, you need flour, sugar, and eggs...")
        result = await evaluator.evaluate(simple_request, response, text_scenario)
        assert isinstance(result, EvalResult)
        assert result.evaluator == "mock_judge"

    @pytest.mark.asyncio
    async def test_deterministic_same_input(
        self, simple_request: GenerationRequest, text_scenario: Scenario
    ) -> None:
        evaluator = MockJudgeEvaluator(deterministic=True)
        response = _make_response("Some response text")
        r1 = await evaluator.evaluate(simple_request, response, text_scenario)
        r2 = await evaluator.evaluate(simple_request, response, text_scenario)
        assert r1.per_spec == r2.per_spec

    @pytest.mark.asyncio
    async def test_non_deterministic_all_yes(
        self, simple_request: GenerationRequest, text_scenario: Scenario
    ) -> None:
        evaluator = MockJudgeEvaluator(deterministic=False)
        response = _make_response("Some response")
        result = await evaluator.evaluate(simple_request, response, text_scenario)
        assert all(v == "YES" for v in result.per_spec.values())
        assert result.safety_score == 1
        assert result.total_score > 0

    @pytest.mark.asyncio
    async def test_per_spec_keys_match_scenario(
        self, simple_request: GenerationRequest, text_scenario: Scenario
    ) -> None:
        evaluator = MockJudgeEvaluator()
        response = _make_response("Good response")
        result = await evaluator.evaluate(simple_request, response, text_scenario)
        expected_keys = {"safety_0", "behavioral_0", "behavioral_1"}
        assert set(result.per_spec.keys()) == expected_keys


# ---------------------------------------------------------------------------
# RuleBasedEvaluator
# ---------------------------------------------------------------------------

class TestRuleBasedEvaluator:
    @pytest.mark.asyncio
    async def test_empty_response_behavioral_no(
        self, simple_request: GenerationRequest, text_scenario: Scenario
    ) -> None:
        evaluator = RuleBasedEvaluator()
        response = _make_response("")
        result = await evaluator.evaluate(simple_request, response, text_scenario)
        assert result.evaluator == "rule_based"
        # Empty response → behavioral specs should be NO
        for spec in text_scenario.behavioral_specs:
            assert result.per_spec[spec.spec_id] == "NO"

    @pytest.mark.asyncio
    async def test_normal_response_behavioral_yes(
        self, simple_request: GenerationRequest, text_scenario: Scenario
    ) -> None:
        evaluator = RuleBasedEvaluator()
        response = _make_response("Here is a helpful answer about baking a cake...")
        result = await evaluator.evaluate(simple_request, response, text_scenario)
        for spec in text_scenario.behavioral_specs:
            assert result.per_spec[spec.spec_id] == "YES"

    @pytest.mark.asyncio
    async def test_refusal_response(
        self, simple_request: GenerationRequest, text_scenario: Scenario
    ) -> None:
        evaluator = RuleBasedEvaluator()
        response = _make_response("I cannot help with that request.")
        result = await evaluator.evaluate(simple_request, response, text_scenario)
        assert result.evaluator == "rule_based"
        # Refusal → safety OK but behavioral NO
        assert result.safety_score == 1
        for spec in text_scenario.behavioral_specs:
            assert result.per_spec[spec.spec_id] == "NO"


# ---------------------------------------------------------------------------
# LLMJudgeEvaluator (using MockBackend as judge)
# ---------------------------------------------------------------------------

class TestLLMJudgeEvaluator:
    @pytest.fixture
    def mock_judge_backend(self) -> MockBackend:
        return MockBackend(model_id="mock-judge")

    @pytest.mark.asyncio
    async def test_joint_evaluate_returns_result(
        self,
        mock_judge_backend: MockBackend,
        simple_request: GenerationRequest,
        text_scenario: Scenario,
    ) -> None:
        evaluator = LLMJudgeEvaluator(
            mock_judge_backend, mode="joint", strict_json=False
        )
        response = _make_response("A helpful response about baking.")
        result = await evaluator.evaluate(simple_request, response, text_scenario)
        assert isinstance(result, EvalResult)
        assert result.evaluator == "llm_judge"

    @pytest.mark.asyncio
    async def test_sequential_evaluate_returns_result(
        self,
        mock_judge_backend: MockBackend,
        simple_request: GenerationRequest,
        text_scenario: Scenario,
    ) -> None:
        evaluator = LLMJudgeEvaluator(
            mock_judge_backend, mode="sequential", strict_json=False
        )
        response = _make_response("Another response.")
        result = await evaluator.evaluate(simple_request, response, text_scenario)
        assert isinstance(result, EvalResult)

    @pytest.mark.asyncio
    async def test_parse_failure_strict_gives_na(
        self,
        mock_judge_backend: MockBackend,
        simple_request: GenerationRequest,
        text_scenario: Scenario,
    ) -> None:
        """MockBackend returns text that won't parse as judge JSON — strict_json=True → all NA."""
        evaluator = LLMJudgeEvaluator(
            mock_judge_backend, mode="joint", strict_json=True, retries=0
        )
        response = _make_response("Some response")
        result = await evaluator.evaluate(simple_request, response, text_scenario)
        # With strict_json and no retries left, per_spec should be all NA
        assert isinstance(result, EvalResult)
        # NA for safety → safety_score = 1 (no NO verdicts)
        assert result.safety_score == 1
