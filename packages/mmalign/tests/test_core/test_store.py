"""Tests for mmalign.core.store — FileResultStore."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import pytest
from mmalign.core.evaluator import EvalResult, compute_sar
from mmalign.core.result import (
    BenchmarkRun,
    HumanReviewLabel,
    RunSummary,
    SampleResult,
    SpecLabel,
)
from mmalign.core.store import FileResultStore, RunNotFoundError, new_run_id
from mmalign.core.types import GenerationRequest


def _utc(offset_h: float = 0.0) -> datetime:
    return datetime.now(tz=timezone.utc) - timedelta(hours=abs(offset_h))


def _make_run(
    run_id: str | None = None,
    scenario: str = "education",
    model: str = "mock",
    method: str = "align3",
    status: str = "completed",
    age_hours: float = 0.0,
) -> BenchmarkRun:
    return BenchmarkRun(
        run_id=run_id or new_run_id(),
        status=status,  # type: ignore[arg-type]
        scenario=scenario,
        model=model,
        method=method,
        evaluator="llm_judge",
        summary=RunSummary(
            safety_score=0.9,
            behavioral_score=0.8,
            total_score=0.86,
            unsafe_rate=0.1,
            sample_count=5,
        ),
        started_at=_utc(age_hours),
    )


def _make_sample(run_id: str, sample_id: str) -> SampleResult:
    return SampleResult(
        run_id=run_id,
        sample_id=sample_id,
        scenario_id="education",
        model="mock",
        method="align3",
        evaluator="llm_judge",
        request=GenerationRequest(model="mock", input="test"),
        eval_result=EvalResult(
            safety_score=1,
            behavioral_score=0.8,
            total_score=compute_sar(1, 0.8),
            per_spec={"safety_0": "YES"},
        ),
        created_at=_utc(),
    )


def _make_review(run_id: str, sample_id: str) -> HumanReviewLabel:
    return HumanReviewLabel(
        run_id=run_id,
        sample_id=sample_id,
        reviewer_id="reviewer-001",
        queue_reason="audit",
        per_spec=[SpecLabel(spec_id="safety_0", verdict="YES", confidence=0.9)],
        created_at=_utc(),
    )


@pytest.fixture
def store(tmp_path: Path) -> FileResultStore:
    return FileResultStore(root=tmp_path / "results")


class TestFileResultStoreBasics:
    def test_save_and_get_run(self, store: FileResultStore):
        run = _make_run("run-001")
        store.save_run(run)
        loaded = store.get_run("run-001")
        assert loaded.run_id == "run-001"
        assert loaded.scenario == "education"

    def test_get_missing_run_raises(self, store: FileResultStore):
        with pytest.raises(RunNotFoundError):
            store.get_run("nonexistent")

    def test_save_run_overwrites(self, store: FileResultStore):
        run = _make_run("run-002", status="running")
        store.save_run(run)
        run.status = "completed"  # type: ignore[assignment]
        store.save_run(run)
        loaded = store.get_run("run-002")
        assert loaded.status == "completed"

    def test_directory_created_on_first_write(self, tmp_path: Path):
        root = tmp_path / "new" / "nested" / "dir"
        store = FileResultStore(root=root)
        store.save_run(_make_run("r"))
        assert root.exists()

    def test_root(self, tmp_path: Path):
        root = tmp_path / "results"
        store = FileResultStore(root=root)
        assert store.root() == root


class TestSampleResultIO:
    def test_append_and_iter(self, store: FileResultStore):
        run = _make_run("run-s")
        store.save_run(run)
        s1 = _make_sample("run-s", "sample-0")
        s2 = _make_sample("run-s", "sample-1")
        store.append_sample_result(s1)
        store.append_sample_result(s2)

        loaded = list(store.iter_sample_results("run-s"))
        assert len(loaded) == 2
        assert {sr.sample_id for sr in loaded} == {"sample-0", "sample-1"}

    def test_iter_missing_run_yields_nothing(self, store: FileResultStore):
        results = list(store.iter_sample_results("no-such-run"))
        assert results == []

    def test_sample_result_round_trip(self, store: FileResultStore):
        run = _make_run("run-rt")
        store.save_run(run)
        sr = _make_sample("run-rt", "s-0")
        store.append_sample_result(sr)
        [loaded] = list(store.iter_sample_results("run-rt"))
        assert loaded.sample_id == "s-0"
        assert loaded.eval_result is not None
        assert loaded.eval_result.safety_score == 1


class TestReviewLabelIO:
    def test_append_and_iter(self, store: FileResultStore):
        run = _make_run("run-rv")
        store.save_run(run)
        label = _make_review("run-rv", "s-0")
        store.append_review_label(label)

        labels = list(store.iter_review_labels("run-rv"))
        assert len(labels) == 1
        assert labels[0].sample_id == "s-0"
        assert labels[0].queue_reason == "audit"


class TestListRuns:
    def _populate(self, store: FileResultStore) -> list[BenchmarkRun]:
        runs = [
            _make_run("r1", scenario="education", model="gpt-4.1", method="align3", age_hours=1),
            _make_run("r2", scenario="education", model="qwen3-32b", method="vanilla", age_hours=2),
            _make_run("r3", scenario="privacy", model="gpt-4.1", method="align3", age_hours=3),
            _make_run("r4", scenario="privacy", model="deepseek-v3", method="raw", age_hours=4, status="running"),
        ]
        for r in runs:
            store.save_run(r)
        return runs

    def test_list_all(self, store: FileResultStore):
        self._populate(store)
        page, total = store.list_runs()
        assert total == 4
        assert len(page) == 4

    def test_filter_scenario(self, store: FileResultStore):
        self._populate(store)
        page, total = store.list_runs(scenario="privacy")
        assert total == 2
        assert all(r.scenario == "privacy" for r in page)

    def test_filter_model(self, store: FileResultStore):
        self._populate(store)
        page, total = store.list_runs(model="gpt-4.1")
        assert total == 2

    def test_filter_status(self, store: FileResultStore):
        self._populate(store)
        page, total = store.list_runs(status="running")
        assert total == 1
        assert page[0].status == "running"

    def test_pagination(self, store: FileResultStore):
        self._populate(store)
        page, total = store.list_runs(limit=2, offset=0)
        assert len(page) == 2
        assert total == 4
        page2, _ = store.list_runs(limit=2, offset=2)
        assert len(page2) == 2
        ids_1 = {r.run_id for r in page}
        ids_2 = {r.run_id for r in page2}
        assert ids_1.isdisjoint(ids_2)

    def test_newest_first(self, store: FileResultStore):
        self._populate(store)
        page, _ = store.list_runs()
        # r1 has smallest age (most recent)
        assert page[0].run_id == "r1"

    def test_empty_store(self, store: FileResultStore):
        page, total = store.list_runs()
        assert page == []
        assert total == 0


class TestRescanAndDelete:
    def test_rescan_rebuilds_index(self, store: FileResultStore):
        run = _make_run("r-rescan")
        store.save_run(run)
        # Corrupt the index
        (store.root() / "index.json").write_text("bad json")
        count = store.rescan()
        assert count == 1
        page, total = store.list_runs()
        assert total == 1

    def test_delete_run(self, store: FileResultStore):
        run = _make_run("r-del")
        store.save_run(run)
        store.append_sample_result(_make_sample("r-del", "s0"))
        store.delete_run("r-del")
        with pytest.raises(RunNotFoundError):
            store.get_run("r-del")
        _, total = store.list_runs()
        assert total == 0


class TestNewRunId:
    def test_format(self):
        rid = new_run_id()
        assert rid.startswith("run-")
        parts = rid.split("-")
        # run-{date}T{time}-{hex8}
        assert len(parts) >= 3

    def test_uniqueness(self):
        ids = {new_run_id() for _ in range(20)}
        assert len(ids) == 20
