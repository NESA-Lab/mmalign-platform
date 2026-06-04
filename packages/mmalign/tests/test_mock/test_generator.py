"""Integration tests for the mock data generator + derived metrics.

Verifies that a single `generate()` call produces a self-consistent corpus
from which every dashboard metric can be derived (none hardcoded).
"""

from __future__ import annotations

import json

import pytest

from mmalign.core.store import FileResultStore
from mmalign.engines.diagnosis.agreement import AgreementCalculator
from mmalign.engines.diagnosis.consistency import ConsistencyCalculator
from mmalign.mock import generate


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    base = tmp_path_factory.mktemp("mmalign") / "data"
    results = base / "results"
    report = generate(scale="small", reset=True, results_dir=results, workspace_start=base)
    return base, results, report


def test_report_counts(corpus):
    _, _, report = corpus
    d = report.as_dict()
    assert d["runs"] > 0
    assert d["samples"] > 0
    assert d["reviews"] > 0
    assert d["scenarios"] == 7
    assert d["benchmark_prompts"] >= 7 * 45


def test_benchmark_packages_written(corpus):
    base, _, _ = corpus
    builtin = base / "benchmarks" / "builtin"
    for scenario in ("education", "privacy", "child", "biochem", "health", "code", "travel"):
        pkg = builtin / scenario
        assert (pkg / "metadata.yaml").exists()
        assert (pkg / "specifications.json").exists()
        prompts = json.loads((pkg / "prompts.json").read_text(encoding="utf-8"))
        assert len(prompts) >= 45
        assert {p["label"] for p in prompts} <= {"safe", "unsafe"}


def test_runs_and_summaries_persisted(corpus):
    _, results, _ = corpus
    store = FileResultStore(results)
    runs, total = store.list_runs(limit=1000)
    assert total > 0
    completed = [r for r in runs if r.status == "completed"]
    assert completed
    # capability_rate is set on non-raw runs
    assert any(r.summary and r.summary.capability_rate is not None for r in completed)


def test_platform_stats(corpus):
    _, results, _ = corpus
    store = FileResultStore(results)
    stats = store.read_platform_stats()
    assert stats["github_stars"] > 0
    assert len(stats["finetuned_architectures"]) >= 3


def test_cross_modal_consistency_derivable(corpus):
    _, results, _ = corpus
    store = FileResultStore(results)
    calc = ConsistencyCalculator(store)
    # there should be vision/text pairs producing a real consistency value
    assert calc.metric_1_1() > 0
    assert calc.family_consistency()  # non-empty


def test_privacy_and_false_positive(corpus):
    _, results, _ = corpus
    store = FileResultStore(results)
    calc = ConsistencyCalculator(store)
    assert calc.privacy_consistency() >= 0  # may be 0 if no DP pairs at small scale
    fp = calc.false_positive_rate()
    assert 0.0 <= fp <= 1.0


def test_human_agreement_not_perfect(corpus):
    _, results, _ = corpus
    store = FileResultStore(results)
    stats = AgreementCalculator(store).compute_stats()
    assert stats.total_reviewed > 0
    # intentional disagreement seeded → agreement strictly between 0 and 1
    assert 0.0 < stats.agreement_rate < 1.0


def test_all_nine_metrics_nonzero_at_large_scale(tmp_path):
    base = tmp_path / "data"
    results = base / "results"
    generate(scale="large", reset=True, results_dir=results, workspace_start=base)
    store = FileResultStore(results)

    calc = ConsistencyCalculator(store)
    stats = store.read_platform_stats()
    runs, _ = store.list_runs(status="completed", limit=500)

    # 1.1 cross-modal, 1.9 privacy
    assert calc.metric_1_1() > 0
    assert calc.privacy_consistency() > 0
    # 1.2 unsafe rate present (some unsafe samples exist)
    assert any(r.summary and r.summary.unsafe_rate > 0 for r in runs)
    # 1.3 capability retention
    assert any(r.summary and r.summary.capability_rate for r in runs)
    # 1.4 stars, 1.8 finetuned archs
    assert stats["github_stars"] > 0
    assert len(stats["finetuned_architectures"]) >= 3
    # 1.6 / 1.7 agreement
    ag = AgreementCalculator(store).compute_stats()
    assert ag.total_reviewed > 0
