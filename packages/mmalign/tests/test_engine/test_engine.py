"""Unit tests for DiagnosisEngine end-to-end with mock backends."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from mmalign.core.result import BenchmarkRun, RunStatus
from mmalign.core.store import FileResultStore
from mmalign.engines.diagnosis.runner import DiagnosisEngine


def _write_configs(tmpdir: Path) -> None:
    """Write minimal config files and a self-contained builtin benchmark package."""
    configs = tmpdir / "configs"
    configs.mkdir()

    (configs / "models.yaml").write_text(yaml.dump({
        "models": {"mock": {"provider": "mock"}},
    }), encoding="utf-8")

    (configs / "methods.yaml").write_text(yaml.dump({
        "methods": {
            "raw": {"type": "raw"},
            "vanilla": {"type": "vanilla"},
            "align3": {"type": "align3", "max_stage_tokens": 256},
        }
    }), encoding="utf-8")

    (configs / "evaluators.yaml").write_text(yaml.dump({
        "evaluators": {"mock_judge": {"type": "mock_judge"}},
    }), encoding="utf-8")

    pkg = tmpdir / "data" / "benchmarks" / "builtin" / "test_domain"
    pkg.mkdir(parents=True)
    (pkg / "metadata.yaml").write_text(yaml.dump({
        "schema_version": 1,
        "id": "test_domain",
        "name": "Test Domain",
        "description": "A unit-test-only scenario",
        "origin": "builtin",
        "source": "native",
        "language": "en",
        "modality": "text",
        "enabled": True,
    }), encoding="utf-8")
    (pkg / "specifications.json").write_text(json.dumps({
        "safety_specifications": ["Do not cause harm."],
        "behavioral_specifications": ["Be helpful.", "Be clear."],
    }), encoding="utf-8")
    (pkg / "prompts.json").write_text(json.dumps([
        {"prompt": "Hello, how are you?", "label": "safe"},
        {"prompt": "Tell me something useful.", "label": "safe"},
    ]), encoding="utf-8")


class TestDiagnosisEngine:
    @pytest.mark.asyncio
    async def test_run_benchmark_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _write_configs(tmpdir)
            engine = DiagnosisEngine(
                configs_dir=tmpdir / "configs",
                results_dir=tmpdir / "data" / "results",
            )
            run = await engine.run_benchmark(
                model="mock",
                scenario="test_domain",
                method="raw",
                evaluator="mock_judge",
                benchmark_origin="builtin",
                concurrency=2,
            )
            assert run.status == "completed"
            assert run.summary is not None
            assert run.summary.sample_count == 2

    @pytest.mark.asyncio
    async def test_run_benchmark_align3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _write_configs(tmpdir)
            engine = DiagnosisEngine(
                configs_dir=tmpdir / "configs",
                results_dir=tmpdir / "data" / "results",
            )
            run = await engine.run_benchmark(
                model="mock",
                scenario="test_domain",
                method="align3",
                evaluator="mock_judge",
                benchmark_origin="builtin",
                concurrency=2,
            )
            assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_failed_run_persists_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _write_configs(tmpdir)
            engine = DiagnosisEngine(
                configs_dir=tmpdir / "configs",
                results_dir=tmpdir / "data" / "results",
            )
            run = await engine.run_benchmark(
                model="mock",
                scenario="nonexistent",
                method="raw",
                evaluator="mock_judge",
                concurrency=2,
            )
            assert run.status == "failed"

    @pytest.mark.asyncio
    async def test_sample_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _write_configs(tmpdir)
            engine = DiagnosisEngine(
                configs_dir=tmpdir / "configs",
                results_dir=tmpdir / "data" / "results",
            )
            run = await engine.run_benchmark(
                model="mock",
                scenario="test_domain",
                method="raw",
                evaluator="mock_judge",
                benchmark_origin="builtin",
                sample_limit=1,
                concurrency=2,
            )
            assert run.status == "completed"
            assert run.summary is not None
            assert run.summary.sample_count == 1
