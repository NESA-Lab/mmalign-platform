"""Tests for HumanReviewEngine queue classification."""

from __future__ import annotations

from mmalign.core.evaluator import EvalResult
from mmalign.core.types import GenerationRequest
from mmalign.engines.diagnosis.human_review import HumanReviewEngine
from mmalign.core.result import SampleResult


def _sample(total_score: float, per_spec: dict) -> SampleResult:
    return SampleResult(
        run_id="run-test",
        sample_id="s1",
        scenario_id="education",
        model="mock",
        method="align3",
        evaluator="mock_judge",
        request=GenerationRequest(model="mock", input="test prompt"),
        eval_result=EvalResult.from_per_spec(
            per_spec,
            _minimal_scenario(per_spec),
            evaluator="mock_judge",
        ),
    )


def _minimal_scenario(per_spec: dict):
    from mmalign.core.scenario import Scenario, Specification

    safety = [Specification(spec_id=k, category="safety", text="s") for k in per_spec if k.startswith("safety")]
    behavioral = [
        Specification(spec_id=k, category="behavioral", text="b")
        for k in per_spec
        if k.startswith("behavioral")
    ]
    return Scenario(
        id="education",
        name="Education",
        description="",
        modality="text",
        language="zh",
        safety_specs=safety or [Specification(spec_id="safety_0", category="safety", text="no harm")],
        behavioral_specs=behavioral or [Specification(spec_id="behavioral_0", category="behavioral", text="be helpful")],
    )


def test_classify_high_risk_on_safety_no():
    engine = HumanReviewEngine()
    sr = _sample(0.9, {"safety_0": "NO", "behavioral_0": "YES"})
    result = engine.classify_sample(sr)
    assert result is not None
    reason, risk, priority = result
    assert reason == "high_risk"
    assert risk == "high"
    assert priority == "high"


def test_classify_low_confidence():
    engine = HumanReviewEngine()
    # behavioral NO → low SAR total_score (0.3) without safety veto
    sr = _sample(0.3, {"safety_0": "YES", "behavioral_0": "NO"})
    result = engine.classify_sample(sr)
    assert result is not None
    assert result[0] == "low_confidence"


def test_skip_high_confidence_safe():
    engine = HumanReviewEngine()
    sr = _sample(0.95, {"safety_0": "YES", "behavioral_0": "YES"})
    assert engine.classify_sample(sr) is None
