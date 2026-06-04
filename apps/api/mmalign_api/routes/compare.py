"""Compare route: GET /api/v1/compare — aggregate method A vs B from ResultStore."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, Query

from mmalign_api.deps import get_store
from mmalign_api.schemas import CompareMetricPair, CompareResult
from mmalign.core.store import FileResultStore

router = APIRouter(tags=["compare"])


@router.get("/compare", response_model=CompareResult)
async def compare_methods(
    method_a: str = Query(default="align3"),
    method_b: str = Query(default="raw"),
    scenario: str | None = Query(default=None),
    store: FileResultStore = Depends(get_store),
) -> CompareResult:
    """Aggregate benchmark scores across completed runs and compare two methods."""
    runs, _ = store.list_runs(status="completed", limit=500)

    def _collect(method: str) -> dict[str, list[float]]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for r in runs:
            if r.method != method:
                continue
            if scenario and scenario != "all" and r.scenario != scenario:
                continue
            if not r.summary:
                continue
            s = r.summary
            buckets["safety_score"].append(s.safety_score)
            buckets["behavioral_score"].append(s.behavioral_score)
            buckets["total_score"].append(s.total_score)
            buckets["unsafe_rate"].append(s.unsafe_rate)
            if s.capability_rate is not None:
                buckets["capability_rate"].append(s.capability_rate)
        return buckets

    def _mean(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    a_data = _collect(method_a)
    b_data = _collect(method_b)

    metric_keys = ["safety_score", "behavioral_score", "total_score", "unsafe_rate", "capability_rate"]
    metrics = {
        k: CompareMetricPair(a=_mean(a_data.get(k, [])), b=_mean(b_data.get(k, [])))
        for k in metric_keys
    }

    # Trade-off scatter points (safety vs capability)
    tradeoff_points: list[dict[str, Any]] = []
    for method, data in [(method_a, a_data), (method_b, b_data)]:
        safety_vals = data.get("safety_score", [])
        cap_vals = data.get("capability_rate", [])
        if safety_vals:
            tradeoff_points.append({
                "method": method,
                "safety": _mean(safety_vals),
                "capability": _mean(cap_vals) if cap_vals else 0.0,
                "label": method,
            })

    # Per-scenario breakdown
    per_scenario_data: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {method_a: defaultdict(list), method_b: defaultdict(list)}
    )
    for r in runs:
        if r.method not in (method_a, method_b):
            continue
        if not r.summary:
            continue
        per_scenario_data[r.scenario][r.method]["total"].append(r.summary.total_score)

    per_scenario = [
        {
            "scenario": sc,
            "a": _mean(data.get(method_a, {}).get("total", [])),
            "b": _mean(data.get(method_b, {}).get("total", [])),
        }
        for sc, data in per_scenario_data.items()
    ]

    return CompareResult(
        method_a=method_a,
        method_b=method_b,
        scenario=scenario or "all",
        metrics=metrics,
        tradeoff_points=tradeoff_points,
        per_scenario=per_scenario,
    )
