"""Tests for mmalign.core.result — result models and compute_run_summary."""

from datetime import datetime, timezone
from typing import Literal

import pytest
from mmalign.core.evaluator import EvalResult, compute_sar
from mmalign.core.result import (
    BackendTelemetry,
    BenchmarkRun,
    BenchmarkRunSummary,
    HumanReviewLabel,
    RunSummary,
    SampleResult,
    SpecLabel,
    compute_run_summary,
)
from mmalign.core.types import (
    GenerationRequest,
    GenerationResult,
    GenerationOutput,
    OutputContentPart,
)


def _utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_eval(safety: Literal[0, 1], b_score: float) -> EvalResult:
    return EvalResult(
        safety_score=safety,
        behavioral_score=b_score,
        total_score=compute_sar(safety, b_score),
        per_spec={"safety_0": "YES" if safety == 1 else "NO"},
    )


def _make_sample(
    run_id: str,
    sample_id: str,
    safety: Literal[0, 1] = 1,
    b_score: float = 0.8,
    status: str = "completed",
) -> SampleResult:
    req = GenerationRequest(model="mock", input="test prompt")
    return SampleResult(
        run_id=run_id,
        sample_id=sample_id,
        scenario_id="education",
        model="mock",
        method="align3",
        evaluator="llm_judge",
        request=req,
        eval_result=_make_eval(safety, b_score) if status == "completed" else None,
        status=status,  # type: ignore[arg-type]
        created_at=_utc(),
    )


class TestRunSummary:
    def test_valid(self):
        s = RunSummary(
            safety_score=0.9,
            behavioral_score=0.8,
            total_score=0.77,
            unsafe_rate=0.1,
            sample_count=10,
        )
        assert s.error_count == 0

    def test_out_of_range_raises(self):
        with pytest.raises(Exception):
            RunSummary(
                safety_score=1.5,
                behavioral_score=0.5,
                total_score=0.5,
                unsafe_rate=0.0,
                sample_count=1,
            )


class TestBackendTelemetry:
    def test_failed_flag(self):
        t = BackendTelemetry(provider="openai", model_name="gpt-4.1", api_style="chat_completions")
        assert not t.failed
        t2 = BackendTelemetry(
            provider="openai",
            model_name="gpt-4.1",
            api_style="chat_completions",
            error_code="429",
            error_message="Rate limit exceeded",
        )
        assert t2.failed

    def test_serialisation(self):
        t = BackendTelemetry(
            provider="mock",
            model_name="mock",
            api_style="mock",
            latency_ms=12.5,
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        restored = BackendTelemetry.model_validate(t.model_dump())
        assert restored == t


class TestSampleResult:
    def test_is_unsafe_with_zero_safety(self):
        sr = _make_sample("run-1", "s-0", safety=0)
        assert sr.is_unsafe

    def test_is_not_unsafe_with_one_safety(self):
        sr = _make_sample("run-1", "s-1", safety=1)
        assert not sr.is_unsafe

    def test_is_not_unsafe_when_no_eval(self):
        sr = _make_sample("run-1", "s-2", status="failed")
        assert not sr.is_unsafe

    def test_serialisation(self):
        sr = _make_sample("run-abc", "s-1")
        restored = SampleResult.model_validate_json(sr.model_dump_json())
        assert restored.sample_id == "s-1"
        assert restored.eval_result is not None
        assert restored.eval_result.safety_score == 1


class TestBenchmarkRun:
    def test_to_summary(self):
        run = BenchmarkRun(
            run_id="run-001",
            scenario="education",
            model="gpt-4.1",
            method="align3",
            evaluator="llm_judge",
            started_at=_utc(),
        )
        summary = run.to_summary()
        assert isinstance(summary, BenchmarkRunSummary)
        assert summary.run_id == "run-001"
        assert summary.summary is None

    def test_serialisation(self):
        run = BenchmarkRun(
            run_id="run-xyz",
            status="completed",
            scenario="privacy",
            model="qwen3-32b",
            method="vanilla",
            evaluator="rule_based",
            summary=RunSummary(
                safety_score=0.8,
                behavioral_score=0.7,
                total_score=0.77,
                unsafe_rate=0.2,
                sample_count=5,
            ),
            started_at=_utc(),
        )
        restored = BenchmarkRun.model_validate_json(run.model_dump_json())
        assert restored.run_id == "run-xyz"
        assert restored.summary is not None
        assert restored.summary.sample_count == 5


class TestComputeRunSummary:
    def test_all_completed(self):
        samples = [
            _make_sample("r", "s0", safety=1, b_score=1.0),
            _make_sample("r", "s1", safety=1, b_score=0.5),
            _make_sample("r", "s2", safety=0, b_score=0.8),
        ]
        summary = compute_run_summary(samples)
        assert summary.sample_count == 3
        assert summary.error_count == 0
        # 2 out of 3 have safety=1 → avg safety_score ≈ 0.667
        assert summary.safety_score == pytest.approx(2 / 3, rel=1e-3)
        assert summary.unsafe_rate == pytest.approx(1 / 3, rel=1e-3)

    def test_with_failures(self):
        samples = [
            _make_sample("r", "s0", safety=1, b_score=1.0),
            _make_sample("r", "s1", status="failed"),
            _make_sample("r", "s2", status="skipped"),
        ]
        summary = compute_run_summary(samples)
        assert summary.sample_count == 3
        assert summary.error_count == 2
        # Only 1 scored sample
        assert summary.safety_score == pytest.approx(1.0)

    def test_all_failed_returns_zeros(self):
        samples = [
            _make_sample("r", "s0", status="failed"),
            _make_sample("r", "s1", status="failed"),
        ]
        summary = compute_run_summary(samples)
        assert summary.total_score == pytest.approx(0.0)
        assert summary.sample_count == 2
        assert summary.error_count == 2

    def test_sar_formula_applied(self):
        # safety=1, b_score=0.5 → SAR = 1*(0.5*0.7+0.3) = 0.65
        samples = [_make_sample("r", "s0", safety=1, b_score=0.5)]
        summary = compute_run_summary(samples)
        assert summary.total_score == pytest.approx(0.65)
