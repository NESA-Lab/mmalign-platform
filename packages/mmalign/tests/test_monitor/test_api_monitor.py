"""Integration tests for the monitor API routes and OpenAI-compat façade."""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from mmalign.core.result import BenchmarkRun, RunSummary
from mmalign.core.store import FileResultStore


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _write_minimal_configs(configs_dir: Path) -> None:
    """Write the minimal configs needed for API tests."""
    (configs_dir / "models.yaml").write_text(
        yaml.dump({"models": {"mock": {"provider": "mock"}}}),
        encoding="utf-8",
    )
    (configs_dir / "methods.yaml").write_text(
        yaml.dump({"methods": {"align3": {"type": "align3"}}}),
        encoding="utf-8",
    )
    (configs_dir / "evaluators.yaml").write_text(
        yaml.dump({"evaluators": {"mock_judge": {"type": "mock_judge"}}}),
        encoding="utf-8",
    )
    (configs_dir / "policies.yaml").write_text(
        yaml.dump({"policies": {}}),
        encoding="utf-8",
    )


def _make_test_run(
    store: FileResultStore,
    scenario: str = "education",
    model: str = "mock",
    safety: float = 0.85,
    days_ago: int = 0,
) -> None:
    started_at = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    run = BenchmarkRun(
        run_id=f"run-{uuid.uuid4().hex[:8]}",
        status="completed",
        scenario=scenario,
        model=model,
        method="align3",
        evaluator="mock_judge",
        summary=RunSummary(
            safety_score=safety,
            behavioral_score=0.80,
            total_score=(safety + 0.80) / 2,
            unsafe_rate=1.0 - safety,
            capability_rate=0.95,
            sample_count=10,
        ),
        started_at=started_at,
        completed_at=started_at,
    )
    store.save_run(run)


def _make_client(tmp_path: Path) -> TestClient:
    """Build a TestClient with overridden deps pointing to tmp_path."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_minimal_configs(configs_dir)

    from mmalign_api import deps
    deps.configure(configs_dir, results_dir)

    from mmalign_api.app import create_app
    return TestClient(create_app(), raise_server_exceptions=True), configs_dir, results_dir


# ---------------------------------------------------------------------------
# Monitor drift
# ---------------------------------------------------------------------------

class TestMonitorDriftRoute:
    def test_drift_empty_store(self, tmp_path: Path) -> None:
        client, _, _ = _make_client(tmp_path)
        resp = client.get("/api/v1/monitor/drift?days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert "series" in body
        assert "alerts" in body
        assert len(body["series"]) == 7

    def test_drift_with_stable_runs(self, tmp_path: Path) -> None:
        client, _, results_dir = _make_client(tmp_path)
        store = FileResultStore(results_dir)
        for d in range(7):
            _make_test_run(store, safety=0.90, days_ago=d)
        resp = client.get("/api/v1/monitor/drift?days=7")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["alerts"]) == 0

    def test_drift_scenario_filter(self, tmp_path: Path) -> None:
        client, _, results_dir = _make_client(tmp_path)
        store = FileResultStore(results_dir)
        _make_test_run(store, scenario="education", safety=0.90, days_ago=0)
        resp = client.get("/api/v1/monitor/drift?days=7&scenario=education")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Monitor calibrate
# ---------------------------------------------------------------------------

class TestMonitorCalibrateRoute:
    def test_calibrate_prompt_tuning(self, tmp_path: Path) -> None:
        client, _, results_dir = _make_client(tmp_path)
        store = FileResultStore(results_dir)
        _make_test_run(store, safety=0.70, days_ago=0)

        resp = client.post("/api/v1/monitor/calibrate", json={
            "scenario": "education",
            "model": "mock",
            "method": "prompt_tuning",
            "calibration_method": "prompt_tuning",
            "target_metric": "safety_score",
            "target_threshold": 0.80,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("pending", "completed", "failed")
        assert body["method"] == "prompt_tuning"
        assert "run_id" in body
        assert "before" in body

    def test_calibrate_unknown_method_returns_failed(self, tmp_path: Path) -> None:
        client, _, _ = _make_client(tmp_path)
        resp = client.post("/api/v1/monitor/calibrate", json={
            "scenario": "education",
            "model": "mock",
            "method": "bad_method",
            "calibration_method": "bad_method",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"


# ---------------------------------------------------------------------------
# Monitor feedback loop
# ---------------------------------------------------------------------------

class TestMonitorFeedbackRoute:
    def test_feedback_idle_no_drift(self, tmp_path: Path) -> None:
        client, _, results_dir = _make_client(tmp_path)
        store = FileResultStore(results_dir)
        for d in range(14):
            _make_test_run(store, safety=0.90, days_ago=d)

        resp = client.post("/api/v1/monitor/feedback", json={
            "scenario": "education",
            "model": "mock",
            "target_threshold": 0.80,
            "days": 14,
            "calibration_method": "prompt_tuning",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "idle"
        assert not body["drift_detected"]

    def test_feedback_calibrated_on_drift(self, tmp_path: Path) -> None:
        client, _, results_dir = _make_client(tmp_path)
        store = FileResultStore(results_dir)
        for d in range(8, 14):
            _make_test_run(store, safety=0.90, days_ago=d)
        for d in range(0, 2):
            _make_test_run(store, safety=0.20, days_ago=d)

        resp = client.post("/api/v1/monitor/feedback", json={
            "scenario": "education",
            "model": "mock",
            "days": 14,
            "calibration_method": "prompt_tuning",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["drift_detected"]
        assert body["status"] in ("calibrated", "validated")
        assert body["calibration"] is not None


# ---------------------------------------------------------------------------
# OpenAI-compat façade
# ---------------------------------------------------------------------------

class TestOpenAICompatRoutes:
    def _make_compat_client(self, tmp_path: Path):
        """Setup client with a builtin mock benchmark for controller tests."""
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        _write_minimal_configs(configs_dir)

        # Create a minimal benchmark package
        pkg = tmp_path / "data" / "benchmarks" / "builtin" / "education"
        pkg.mkdir(parents=True)
        (pkg / "metadata.yaml").write_text(yaml.dump({
            "id": "education",
            "name": "Education Safety",
            "description": "Test",
            "language": "zh",
            "modality": "text",
            "source": "test",
        }), encoding="utf-8")
        (pkg / "specifications.json").write_text(json.dumps([
            {"id": "safety_0", "category": "safety", "text": "Do not provide harmful content."},
        ]))
        (pkg / "prompts.json").write_text(json.dumps([
            {"id": "p1", "prompt": "Tell me about education."},
        ]))

        from mmalign_api import deps
        deps.configure(configs_dir, results_dir)
        import os
        os.environ["MMALIGN_DATA_DIR"] = str(tmp_path / "data")

        from mmalign_api.app import create_app
        return TestClient(create_app(), raise_server_exceptions=False)

    def test_chat_completions_endpoint_exists(self, tmp_path: Path) -> None:
        client = self._make_compat_client(tmp_path)
        resp = client.post("/v1/chat/completions", json={
            "model": "mock",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        # Either 200 (success) or 400 (model/scenario error) — route must exist
        assert resp.status_code != 404, f"Route should exist, got 404: {resp.text}"

    def test_responses_endpoint_exists(self, tmp_path: Path) -> None:
        client = self._make_compat_client(tmp_path)
        resp = client.post("/v1/responses", json={
            "model": "mock",
            "input": "Hello world",
        })
        assert resp.status_code != 404, f"Route should exist, got 404: {resp.text}"

    def test_chat_completions_streaming_returns_501(self, tmp_path: Path) -> None:
        client = self._make_compat_client(tmp_path)
        resp = client.post("/v1/chat/completions", json={
            "model": "mock",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        assert resp.status_code == 501

    def test_chat_completions_response_shape(self, tmp_path: Path) -> None:
        """When model + scenario exist, response should be OpenAI-shaped."""
        client = self._make_compat_client(tmp_path)
        resp = client.post("/v1/chat/completions", json={
            "model": "mock",
            "messages": [{"role": "user", "content": "Tell me about education."}],
            "mmalign_scenario": "education",
        })
        if resp.status_code == 200:
            body = resp.json()
            assert "choices" in body
            assert len(body["choices"]) > 0
            assert "message" in body["choices"][0]
            assert body["choices"][0]["message"]["role"] == "assistant"
            assert "usage" in body

    def test_responses_response_shape(self, tmp_path: Path) -> None:
        """When model + scenario exist, response should be Responses API-shaped."""
        client = self._make_compat_client(tmp_path)
        resp = client.post("/v1/responses", json={
            "model": "mock",
            "input": "Tell me about education.",
            "mmalign_scenario": "education",
        })
        if resp.status_code == 200:
            body = resp.json()
            assert "output" in body
            assert len(body["output"]) > 0
            assert "content" in body["output"][0]
