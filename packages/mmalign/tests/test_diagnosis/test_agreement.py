"""Tests for AgreementCalculator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mmalign.core.evaluator import EvalResult
from mmalign.core.result import BenchmarkRun, HumanReviewLabel, RunSummary, SampleResult, SpecLabel
from mmalign.core.store import FileResultStore, new_run_id
from mmalign.core.types import GenerationRequest
from mmalign.engines.diagnosis.agreement import AgreementCalculator


@pytest.fixture
def store(tmp_path: Path) -> FileResultStore:
    return FileResultStore(tmp_path)


def _seed_run_with_review(store: FileResultStore) -> None:
    run_id = new_run_id()
    run = BenchmarkRun(
        run_id=run_id,
        status="completed",
        scenario="education",
        model="mock",
        method="align3",
        evaluator="mock_judge",
        summary=RunSummary(
            safety_score=0.8,
            behavioral_score=0.9,
            total_score=0.85,
            unsafe_rate=0.2,
            sample_count=1,
        ),
        started_at=datetime.now(tz=timezone.utc),
    )
    store.save_run(run)
    sr = SampleResult(
        run_id=run_id,
        sample_id="edu-1",
        scenario_id="education",
        model="mock",
        method="align3",
        evaluator="mock_judge",
        request=GenerationRequest(model="mock", input="hello"),
        eval_result=EvalResult(
            safety_score=1,
            behavioral_score=1.0,
            total_score=1.0,
            per_spec={"safety_0": "YES", "behavioral_0": "YES"},
            evaluator="mock_judge",
        ),
    )
    store.append_sample_result(sr)
    store.append_review_label(
        HumanReviewLabel(
            run_id=run_id,
            sample_id="edu-1",
            reviewer_id="expert-01",
            queue_reason="low_confidence",
            per_spec=[
                SpecLabel(spec_id="safety_0", verdict="YES"),
                SpecLabel(spec_id="behavioral_0", verdict="YES"),
            ],
        )
    )


def test_agreement_stats_after_review(store: FileResultStore) -> None:
    _seed_run_with_review(store)
    stats = AgreementCalculator(store).compute_stats()
    assert stats.total_reviewed == 1
    assert stats.agreement_rate == 1.0
    assert stats.hitl_accuracy == 1.0


def test_daily_metric_values_empty(store: FileResultStore) -> None:
    daily = AgreementCalculator(store).daily_metric_values("1.6", days=7)
    assert len(daily) == 7
    assert all(v == 0.0 for _, v in daily)
