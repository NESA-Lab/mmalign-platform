"""Overview routes: metrics, project-board, heatmap, controller-series, alerts, recent-runs."""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, Query

from mmalign_api.deps import get_configs_dir, get_store
from mmalign_api.schemas import (
    AlertsResponse,
    ControllerSeriesPoint,
    ControllerSeriesResponse,
    HeatmapResponse,
    HeatmapScore,
    IndicatorStatus,
    MetricHistoryPoint,
    MetricHistoryResponse,
    MetricsResponse,
    OverviewAlert,
    ProjectBoardResponse,
    ProjectMetric,
    ProjectMilestone,
    ProjectQualityMetric,
    RecentRun,
    RecentRunsResponse,
)
from mmalign.core.metrics import METRIC_SPECS
from mmalign.core.store import FileResultStore

router = APIRouter(tags=["overview"])


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    store: FileResultStore = Depends(get_store),
    configs_dir: Path = Depends(get_configs_dir),
) -> MetricsResponse:
    """Return all project performance indicators with current computed values."""
    now = datetime.now(tz=timezone.utc).isoformat()
    runs, _ = store.list_runs(status="completed", limit=200)

    # Compute dynamic values from ResultStore
    values = _compute_metric_values(runs, store, configs_dir)
    values.update(_platform_metrics(store))

    indicators: dict[str, IndicatorStatus] = {}
    for mid, spec in METRIC_SPECS.items():
        val = values.get(mid, 0.0)
        indicators[mid] = IndicatorStatus(
            name=spec.name,
            value=round(val, 2),
            unit=spec.unit,
            trend="up" if spec.direction == "gte" and val > 0 else "flat",
            delta="",
            target_mid=spec.target_mid,
            target_final=spec.target_final,
            direction=spec.direction,
            description=spec.description,
            source=", ".join(spec.source_components) if spec.source_components else "",
        )
    return MetricsResponse(indicators=indicators, updated_at=now)


# ---------------------------------------------------------------------------
# GET /metrics/{id}/history
# ---------------------------------------------------------------------------

@router.get("/metrics/{indicator_id}/history", response_model=MetricHistoryResponse)
async def get_metric_history(
    indicator_id: str,
    days: int = Query(default=30, ge=7, le=365),
    store: FileResultStore = Depends(get_store),
    configs_dir: Path = Depends(get_configs_dir),
) -> MetricHistoryResponse:
    """Return the daily time-series of a single indicator over the past N days."""
    from fastapi import HTTPException
    if indicator_id not in METRIC_SPECS:
        raise HTTPException(
            status_code=404,
            detail={"code": "INVALID_METRIC", "message": f"Metric {indicator_id!r} not found"},
        )

    now = datetime.now(tz=timezone.utc)

    if indicator_id in ("1.6", "1.7"):
        from mmalign.engines.diagnosis.agreement import AgreementCalculator
        daily = AgreementCalculator(store).daily_metric_values(indicator_id, days)
        history = [
            MetricHistoryPoint(timestamp=ts, value=val) for ts, val in daily
        ]
        return MetricHistoryResponse(indicator_id=indicator_id, history=history)

    if indicator_id in ("1.1", "1.9"):
        from mmalign.engines.diagnosis.consistency import ConsistencyCalculator
        calc = ConsistencyCalculator(store)
        if indicator_id == "1.1":
            daily = calc.daily_consistency(days, dp=False)
        else:
            # privacy retention = DP consistency vs baseline, per day
            base = dict(calc.daily_consistency(days, dp=False))
            prot = dict(calc.daily_consistency(days, dp=True))
            daily = [
                (ts, round(prot[ts] / base[ts] * 100, 2) if base.get(ts) and prot.get(ts) else 0.0)
                for ts in base
            ]
        history = [MetricHistoryPoint(timestamp=ts, value=val) for ts, val in daily]
        return MetricHistoryResponse(indicator_id=indicator_id, history=history)

    if indicator_id in ("1.4_stars", "1.8"):
        # No per-day history; report the current value as a flat line.
        current = _platform_metrics(store).get(indicator_id, 0.0)
        now2 = datetime.now(tz=timezone.utc)
        history = [
            MetricHistoryPoint(
                timestamp=f"{(now2 - timedelta(days=days - 1 - i)).strftime('%Y-%m-%d')}T00:00:00Z",
                value=current,
            )
            for i in range(days)
        ]
        return MetricHistoryResponse(indicator_id=indicator_id, history=history)

    runs, _ = store.list_runs(status="completed", limit=500)
    cutoff = now - timedelta(days=days)

    by_day: dict[str, list[Any]] = defaultdict(list)
    for r in runs:
        if r.started_at >= cutoff:
            by_day[r.started_at.strftime("%Y-%m-%d")].append(r)

    history: list[MetricHistoryPoint] = []
    for i in range(days):
        day = (now - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        day_runs = by_day.get(day, [])
        if day_runs:
            values = _compute_metric_values(day_runs, store, configs_dir)
            val = values.get(indicator_id, 0.0)
        else:
            val = 0.0
        history.append(MetricHistoryPoint(
            timestamp=f"{day}T00:00:00Z",
            value=round(val, 4),
        ))

    return MetricHistoryResponse(indicator_id=indicator_id, history=history)


def _compute_metric_values(
    runs: list[Any],
    store: FileResultStore,
    configs_dir: Path,
) -> dict[str, float]:
    """Compute current indicator values from available data."""
    values: dict[str, float] = {}

    # 1.2 unsafe_rate — "violation rate after the compliance controller", so
    # measured on controlled (align3) runs; falls back to all runs if absent.
    if runs:
        controlled = [r for r in runs if r.summary and r.method == "align3"]
        pool = controlled or [r for r in runs if r.summary]
        unsafe_rates = [r.summary.unsafe_rate * 100 for r in pool]
        values["1.2"] = round(sum(unsafe_rates) / len(unsafe_rates), 2) if unsafe_rates else 0.0

    # 1.3 capability_rate — avg capability_rate across runs
    cap_rates = [r.summary.capability_rate * 100 for r in runs
                 if r.summary and r.summary.capability_rate is not None]
    values["1.3"] = round(sum(cap_rates) / len(cap_rates), 2) if cap_rates else 0.0

    # 1.4 algorithms — count from methods.yaml + 8 SpecBench TTD methods
    try:
        raw = yaml.safe_load((configs_dir / "methods.yaml").read_text(encoding="utf-8"))
        native_count = len(raw.get("methods", {}))
        values["1.4_algorithms"] = float(native_count + 8)  # +8 for SpecBench TTD
    except Exception:
        values["1.4_algorithms"] = 0.0

    # 1.5 scenarios
    try:
        from mmalign.loaders.benchmarks import discover_packages

        values["1.5_scenarios"] = float(len(discover_packages()))
    except Exception:
        values["1.5_scenarios"] = 0.0

    # 1.5 samples — sum prompts across all scenarios
    values["1.5_samples"] = float(_count_all_samples(configs_dir))

    # 1.5 models
    try:
        raw = yaml.safe_load((configs_dir / "models.yaml").read_text(encoding="utf-8"))
        values["1.5_models"] = float(len(raw.get("models", {})))
    except Exception:
        values["1.5_models"] = 0.0

    # 1.6 / 1.7 — from HumanReview labels via AgreementCalculator
    try:
        from mmalign.engines.diagnosis.agreement import AgreementCalculator
        stats = AgreementCalculator(store).compute_stats()
        values["1.6"] = round(stats.agreement_rate * 100, 2)
        values["1.7"] = round(stats.hitl_accuracy * 100, 2)
    except Exception:
        values.setdefault("1.6", 0.0)
        values.setdefault("1.7", 0.0)

    return values


def _platform_metrics(store: FileResultStore) -> dict[str, float]:
    """Store-wide metrics with no per-run home: cross-modal consistency (1.1),
    privacy retention (1.9), GitHub stars (1.4), fine-tuned architectures (1.8).

    Computed once per request (not inside per-day history loops)."""
    from mmalign.engines.diagnosis.consistency import ConsistencyCalculator

    values: dict[str, float] = {}
    calc = ConsistencyCalculator(store)
    values["1.1"] = calc.metric_1_1()
    values["1.9"] = calc.privacy_consistency()

    stats = store.read_platform_stats()
    values["1.4_stars"] = float(stats.get("github_stars", 0.0))
    values["1.8"] = float(len(stats.get("finetuned_architectures", [])))
    return values


def _count_all_samples(configs_dir: Path) -> int:  # noqa: ARG001
    from mmalign.loaders.benchmarks import count_samples, discover_packages

    return sum(count_samples(p) for p in discover_packages())


# ---------------------------------------------------------------------------
# GET /overview/recent-runs
# ---------------------------------------------------------------------------

@router.get("/overview/recent-runs", response_model=RecentRunsResponse)
async def get_recent_runs(
    limit: int = Query(default=10, ge=1, le=50),
    store: FileResultStore = Depends(get_store),
) -> RecentRunsResponse:
    from mmalign.engines.diagnosis.consistency import ConsistencyCalculator
    calc = ConsistencyCalculator(store)

    runs, _ = store.list_runs(limit=limit)
    result: list[RecentRun] = []
    for r in runs:
        s = r.summary
        result.append(RecentRun(
            id=r.run_id,
            time=r.started_at.isoformat(),
            model=r.model,
            scenario=r.scenario,
            method=r.method,
            safety_score=round(s.safety_score * 100, 2) if s else 0.0,
            behavioral_score=round(s.behavioral_score * 100, 2) if s else 0.0,
            consistency_score=round(calc.consistency_for_model(r.scenario, r.model) * 100, 2),
            samples=s.sample_count if s else 0,
        ))
    return RecentRunsResponse(runs=result)


# ---------------------------------------------------------------------------
# GET /overview/heatmap
# ---------------------------------------------------------------------------

@router.get("/overview/heatmap", response_model=HeatmapResponse)
async def get_heatmap(
    store: FileResultStore = Depends(get_store),
) -> HeatmapResponse:
    """Aggregate SAR scores grouped by scenario × model from completed runs."""
    from mmalign.engines.diagnosis.consistency import ConsistencyCalculator
    calc = ConsistencyCalculator(store)

    runs, _ = store.list_runs(status="completed", limit=500)

    # (scenario, model) → [safety, behavioral]
    agg: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in runs:
        if r.summary:
            agg[(r.scenario, r.model)].append(
                (r.summary.safety_score, r.summary.behavioral_score)
            )

    scores: list[HeatmapScore] = []
    for (scenario, model), pairs in agg.items():
        avg_safety = sum(p[0] for p in pairs) / len(pairs)
        avg_behavioral = sum(p[1] for p in pairs) / len(pairs)
        scores.append(HeatmapScore(
            scenario=scenario,
            model=model,
            safety=round(avg_safety, 4),
            behavioral=round(avg_behavioral, 4),
            consistency=round(calc.consistency_for_model(scenario, model), 4),
        ))

    return HeatmapResponse(scores=scores)


# ---------------------------------------------------------------------------
# GET /overview/controller-series
# ---------------------------------------------------------------------------

@router.get("/overview/controller-series", response_model=ControllerSeriesResponse)
async def get_controller_series(
    days: int = Query(default=30, ge=7, le=180),
    store: FileResultStore = Depends(get_store),
) -> ControllerSeriesResponse:
    """Daily aggregation of unsafe_rate and intercept metrics over the past N days."""
    from mmalign.engines.diagnosis.consistency import ConsistencyCalculator

    runs, _ = store.list_runs(status="completed", limit=500)
    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=days)

    # Bucket by day
    by_day: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r.started_at >= cutoff and r.summary:
            day = r.started_at.strftime("%Y-%m-%d")
            by_day[day].append(r.summary.unsafe_rate)

    fp_by_day = dict(ConsistencyCalculator(store).daily_false_positive(days))

    series: list[ControllerSeriesPoint] = []
    for i in range(days):
        d = (now - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        rates = by_day.get(d, [])
        avg_unsafe = sum(rates) / len(rates) if rates else 0.0
        series.append(ControllerSeriesPoint(
            date=d,
            intercept=round(1 - avg_unsafe, 4),
            false_positive=fp_by_day.get(f"{d}T00:00:00Z", 0.0),
            unsafe_rate=round(avg_unsafe, 4),
        ))
    return ControllerSeriesResponse(series=series)


# ---------------------------------------------------------------------------
# GET /overview/alerts
# ---------------------------------------------------------------------------

@router.get("/overview/alerts", response_model=AlertsResponse)
async def get_alerts(
    limit: int = Query(default=20, ge=1, le=100),
    store: FileResultStore = Depends(get_store),
) -> AlertsResponse:
    """Generate alerts from failed runs and high unsafe-rate runs."""
    runs, _ = store.list_runs(limit=100)
    alerts: list[OverviewAlert] = []

    for r in runs:
        if r.status == "failed":
            alerts.append(OverviewAlert(
                id=f"alert-{r.run_id[:8]}",
                level="error",
                message=f"Run {r.run_id[:12]}... failed (model={r.model}, scenario={r.scenario})",
                time=r.started_at.isoformat(),
            ))
        elif r.summary and r.summary.unsafe_rate > 0.3:
            alerts.append(OverviewAlert(
                id=f"alert-unsafe-{r.run_id[:8]}",
                level="warning",
                message=(
                    f"High unsafe rate {r.summary.unsafe_rate*100:.0f}% detected "
                    f"(model={r.model}, scenario={r.scenario})"
                ),
                time=r.started_at.isoformat(),
            ))

    return AlertsResponse(alerts=alerts[:limit])


# ---------------------------------------------------------------------------
# GET /overview/project-board
# ---------------------------------------------------------------------------

@router.get("/overview/project-board", response_model=ProjectBoardResponse)
async def get_project_board(
    store: FileResultStore = Depends(get_store),
    configs_dir: Path = Depends(get_configs_dir),
) -> ProjectBoardResponse:
    """Return project progress metrics, quality indicators, and milestones."""
    runs, _ = store.list_runs(status="completed", limit=200)
    values = _compute_metric_values(runs, store, configs_dir)
    values.update(_platform_metrics(store))

    # Progress metrics (course-deliverable counts)
    progress: list[ProjectMetric] = [
        ProjectMetric(
            id="1.4_algorithms",
            name="开源算法数量",
            current=values.get("1.4_algorithms", 0.0),
            target_mid=10.0,
            target_final=40.0,
            unit="个",
            description="已注册对齐/控制算法数量（含 SpecBench 8 个 TTD 方法）",
            evidence="methods.yaml + SpecBench adapter",
        ),
        ProjectMetric(
            id="1.5_scenarios",
            name="评测场景数量",
            current=values.get("1.5_scenarios", 0.0),
            target_mid=2.0,
            target_final=5.0,
            unit="个",
            description="已构建的评测场景（含 SpecBench 场景）",
            evidence="data/benchmarks/{builtin,user}/metadata.yaml",
        ),
        ProjectMetric(
            id="1.5_samples",
            name="评测样本数量",
            current=values.get("1.5_samples", 0.0),
            target_mid=1000.0,
            target_final=5000.0,
            unit="条",
            description="评测集可用样本总数",
            evidence="data/benchmarks/",
        ),
        ProjectMetric(
            id="1.5_models",
            name="支持模型数量",
            current=values.get("1.5_models", 0.0),
            target_mid=10.0,
            target_final=30.0,
            unit="个",
            description="已注册可评测的模型数量",
            evidence="configs/models.yaml",
        ),
        ProjectMetric(
            id="1.4_stars",
            name="GitHub Stars",
            current=values.get("1.4_stars", 0.0),
            target_mid=500.0,
            target_final=1000.0,
            unit="star",
            description="项目 GitHub 仓库累计 Star 数",
            evidence="GitHub API",
        ),
    ]

    # Quality metrics (runtime performance)
    quality: list[ProjectQualityMetric] = [
        ProjectQualityMetric(
            id="1.2",
            name="违规内容发生率",
            value=values.get("1.2", 0.0),
            unit="%",
            target_mid=5.0,
            target_final=2.0,
            direction="lte",
            source="ResultStore unsafe_rate",
        ),
        ProjectQualityMetric(
            id="1.3",
            name="正常能力保持率",
            value=values.get("1.3", 0.0),
            unit="%",
            target_mid=90.0,
            target_final=90.0,
            direction="gte",
            source="ResultStore capability_rate",
        ),
        ProjectQualityMetric(
            id="1.1",
            name="跨模态一致性保持率",
            value=values.get("1.1", 0.0),
            unit="%",
            target_mid=85.0,
            target_final=90.0,
            direction="gte",
            source="AlignmentEngine (Phase 10)",
        ),
        ProjectQualityMetric(
            id="1.6",
            name="自动评估器与人工标注一致率",
            value=values.get("1.6", 0.0),
            unit="%",
            target_mid=80.0,
            target_final=85.0,
            direction="gte",
            source="HumanReview Engine (Phase 8)",
        ),
    ]

    milestones: list[ProjectMilestone] = [
        ProjectMilestone(title="WebUI Demo 与产品叙事", date="2026-04", status="finish",
                         description="WebUI 9 页面 + 全局布局"),
        ProjectMilestone(title="Core 契约与 ResultStore", date="2026-04", status="finish",
                         description="Pydantic 类型体系、ResultStore、场景数据"),
        ProjectMilestone(title="OpenAI Adapter", date="2026-05", status="finish",
                         description="MockBackend + OpenAIBackend + 后端加载器"),
        ProjectMilestone(title="DiagnosisEngine + CLI", date="2026-05", status="finish",
                         description="Align3 + LLMJudge + mmalign benchmark"),
        ProjectMilestone(title="REST API + WebUI 真实数据", date="2026-05", status="process",
                         description="FastAPI + /api/v1 接口 + WebUI 切换"),
        ProjectMilestone(title="合规控制器 MVP", date="2026-07", status="wait",
                         description="CorrectionEngine + POST /controller/generate"),
        ProjectMilestone(title="人机协同审核", date="2026-08", status="wait",
                         description="ReviewQueue + HumanReview Engine"),
        ProjectMilestone(title="开源发布", date="2026-09", status="wait",
                         description="pip install mmalign + 浦江平台对接"),
    ]

    return ProjectBoardResponse(progress=progress, quality=quality, milestones=milestones)
