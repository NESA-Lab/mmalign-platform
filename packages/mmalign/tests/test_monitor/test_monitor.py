"""Unit tests for the monitor engine (drift, calibration, feedback loop)."""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mmalign.core.result import BenchmarkRun, RunSummary
from mmalign.core.store import FileResultStore
from mmalign.engines.monitor import (
    CalibrationConfig,
    CalibrationEngine,
    DriftEngine,
    FeedbackLoopEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _utcnow_minus(days: int) -> datetime:
    return datetime.now(tz=timezone.utc) - timedelta(days=days)


def _make_run(
    store: FileResultStore,
    scenario: str = "education",
    model: str = "mock",
    safety: float = 0.90,
    behavioral: float = 0.85,
    days_ago: int = 0,
) -> BenchmarkRun:
    started_at = _utcnow_minus(days_ago)
    run = BenchmarkRun(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        status="completed",
        scenario=scenario,
        model=model,
        method="align3",
        evaluator="mock_judge",
        summary=RunSummary(
            safety_score=safety,
            behavioral_score=behavioral,
            total_score=(safety + behavioral) / 2,
            unsafe_rate=1.0 - safety,
            capability_rate=0.95,
            sample_count=10,
        ),
        started_at=started_at,
        completed_at=started_at,
    )
    store.save_run(run)
    return run


# ---------------------------------------------------------------------------
# DriftEngine
# ---------------------------------------------------------------------------

class TestDriftEngine:
    def test_no_runs_returns_empty_series(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        engine = DriftEngine(store)
        series, alerts = engine.detect(days=7)
        assert len(series) == 7
        assert len(alerts) == 0

    def test_series_length_matches_days(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        _make_run(store, days_ago=1)
        engine = DriftEngine(store)
        series, _ = engine.detect(days=14)
        assert len(series) == 14

    def test_scenario_filter(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        _make_run(store, scenario="education", safety=0.90, days_ago=0)
        _make_run(store, scenario="medical", safety=0.50, days_ago=0)
        engine = DriftEngine(store)

        edu_series, _ = engine.detect(days=7, scenario="education")
        med_series, _ = engine.detect(days=7, scenario="medical")

        edu_today = edu_series[-1].safety_score
        med_today = med_series[-1].safety_score
        assert edu_today == pytest.approx(0.90)
        assert med_today == pytest.approx(0.50)

    def test_drift_not_detected_stable_metric(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        for d in range(14):
            _make_run(store, safety=0.90, days_ago=d)
        engine = DriftEngine(store)
        _, alerts = engine.detect(days=14)
        assert len(alerts) == 0

    def test_drift_detected_on_sharp_drop(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        # Stable high scores for 8 days then sudden drop
        for d in range(8, 14):
            _make_run(store, safety=0.90, days_ago=d)
        for d in range(0, 2):
            _make_run(store, safety=0.30, days_ago=d)  # sharp drop
        engine = DriftEngine(store)
        _, alerts = engine.detect(days=14)
        assert any(a.level in ("warning", "error") for a in alerts)

    def test_latest_metric(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        _make_run(store, safety=0.77, days_ago=0)
        engine = DriftEngine(store)
        val = engine.latest_metric("safety_score")
        assert val == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# CalibrationEngine
# ---------------------------------------------------------------------------

class TestCalibrationEngine:
    @pytest.mark.asyncio
    async def test_prompt_tuning_returns_pending(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        _make_run(store, safety=0.70, days_ago=0)
        engine = CalibrationEngine(store)
        result = await engine.calibrate(CalibrationConfig(
            method="prompt_tuning",
            scenario="education",
            model="mock",
        ))
        assert result.status == "pending"
        assert result.method == "prompt_tuning"
        assert result.before == pytest.approx(0.70)
        assert result.run_id.startswith("cal-")

    @pytest.mark.asyncio
    async def test_lora_without_trl_returns_pending(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        engine = CalibrationEngine(store)
        result = await engine.calibrate(CalibrationConfig(
            method="lora",
            scenario="education",
            model="mock",
        ))
        # trl is not in the test env → should gracefully return pending
        assert result.status == "pending"
        assert "trl" in result.message.lower() or result.status == "pending"

    @pytest.mark.asyncio
    async def test_unknown_method_returns_failed(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        engine = CalibrationEngine(store)
        result = await engine.calibrate(CalibrationConfig(
            method="nonexistent",
            scenario="education",
            model="mock",
        ))
        assert result.status == "failed"
        assert "Unknown" in result.message

    @pytest.mark.asyncio
    async def test_calibration_result_has_timestamps(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        engine = CalibrationEngine(store)
        result = await engine.calibrate(CalibrationConfig(method="prompt_tuning"))
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at


# ---------------------------------------------------------------------------
# FeedbackLoopEngine
# ---------------------------------------------------------------------------

class TestFeedbackLoopEngine:
    @pytest.mark.asyncio
    async def test_idle_when_no_drift(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        # Stable scores → no drift
        for d in range(14):
            _make_run(store, safety=0.90, days_ago=d)
        engine = FeedbackLoopEngine.from_store(store)
        result = await engine.run(scenario="education", model="mock")
        assert result.status == "idle"
        assert not result.drift_detected
        assert result.calibration is None

    @pytest.mark.asyncio
    async def test_calibrated_when_drift_detected(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        # Stable then sharp drop → drift
        for d in range(8, 14):
            _make_run(store, safety=0.90, days_ago=d)
        for d in range(0, 2):
            _make_run(store, safety=0.20, days_ago=d)
        engine = FeedbackLoopEngine.from_store(store)
        result = await engine.run(
            scenario="education",
            model="mock",
            days=14,
        )
        assert result.drift_detected
        assert result.calibration is not None
        assert result.status in ("calibrated", "validated")

    @pytest.mark.asyncio
    async def test_loop_result_has_loop_id(self, tmp_path: Path) -> None:
        store = FileResultStore(tmp_path / "results")
        engine = FeedbackLoopEngine.from_store(store)
        result = await engine.run(scenario="education", model="mock")
        assert result.loop_id.startswith("loop-")
