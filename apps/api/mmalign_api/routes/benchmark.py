"""Benchmark routes: POST /benchmark/run, GET /benchmark/{run_id}, GET /benchmark/list."""

from __future__ import annotations

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from mmalign_api.deps import get_configs_dir, get_results_dir, get_store
from mmalign_api.schemas import (
    BenchmarkListOut,
    BenchmarkRunDetailOut,
    BenchmarkRunSummaryOut,
    DefectHeatmapCell,
    DefectHeatmapResponse,
    PerSpecResultOut,
    RunSummaryOut,
    StartBenchmarkRequest,
    StartBenchmarkResponse,
    UploadResultsRequest,
    UploadResultsResponse,
)
from mmalign.core.result import BenchmarkRun, BenchmarkRunSummary
from mmalign.core.store import FileResultStore, RunNotFoundError
from mmalign.core.store import new_run_id

router = APIRouter(tags=["benchmark"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary_out(s: Any) -> RunSummaryOut | None:
    if s is None:
        return None
    return RunSummaryOut(
        safety_score=round(s.safety_score * 100, 2),
        behavioral_score=round(s.behavioral_score * 100, 2),
        total_score=round(s.total_score * 100, 2),
        unsafe_rate=round(s.unsafe_rate * 100, 2),
        capability_rate=round(s.capability_rate * 100, 2) if s.capability_rate is not None else None,
        sample_count=s.sample_count,
        error_count=s.error_count,
    )


def _run_to_summary_out(run: BenchmarkRun) -> BenchmarkRunSummaryOut:
    samples = (
        run.summary.sample_count if run.summary
        else (run.sample_limit or 0)
    )
    return BenchmarkRunSummaryOut(
        run_id=run.run_id,
        created_at=run.started_at.isoformat(),
        status=run.status,
        model=run.model,
        scenario=run.scenario,
        method=run.method,
        evaluator=run.evaluator,
        samples=samples,
        summary=_summary_out(run.summary),
    )


def _core_summary_to_out(s: BenchmarkRunSummary) -> BenchmarkRunSummaryOut:
    samples = s.summary.sample_count if s.summary else 0
    return BenchmarkRunSummaryOut(
        run_id=s.run_id,
        created_at=s.started_at.isoformat(),
        status=s.status,
        model=s.model,
        scenario=s.scenario,
        method=s.method,
        evaluator=s.evaluator,
        samples=samples,
        summary=_summary_out(s.summary),
    )


# ---------------------------------------------------------------------------
# List runs
# ---------------------------------------------------------------------------

@router.get("/benchmark/list", response_model=BenchmarkListOut)
async def list_runs(
    scenario: str | None = Query(default=None),
    model: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    store: FileResultStore = Depends(get_store),
) -> BenchmarkListOut:
    summaries, total = store.list_runs(
        scenario=scenario,
        model=model,
        status=status,  # type: ignore[arg-type]
        limit=limit,
        offset=offset,
    )
    return BenchmarkListOut(
        runs=[_core_summary_to_out(s) for s in summaries],
        total=total,
    )


# ---------------------------------------------------------------------------
# Get run detail
# ---------------------------------------------------------------------------

@router.get("/benchmark/{run_id}", response_model=BenchmarkRunDetailOut)
async def get_run(
    run_id: str,
    store: FileResultStore = Depends(get_store),
    configs_dir: Path = Depends(get_configs_dir),
) -> BenchmarkRunDetailOut:
    try:
        run = store.get_run(run_id)
    except (RunNotFoundError, KeyError):
        raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND", "message": f"Run {run_id} not found"})

    base = _run_to_summary_out(run)

    # Aggregate per_spec stats from sample results
    per_spec = _aggregate_per_spec(run, store, configs_dir)

    return BenchmarkRunDetailOut(**base.model_dump(), per_spec=per_spec)


@router.get("/benchmark/{run_id}/defect-heatmap", response_model=DefectHeatmapResponse)
async def get_defect_heatmap(
    run_id: str,
    store: FileResultStore = Depends(get_store),
    configs_dir: Path = Depends(get_configs_dir),
) -> DefectHeatmapResponse:
    """Per-spec defect attribution cells for the benchmark heatmap visualization."""
    from mmalign.core.store import RunNotFoundError

    try:
        run = store.get_run(run_id)
    except (RunNotFoundError, KeyError):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail={"code": "RUN_NOT_FOUND", "message": f"Run {run_id} not found"},
        )

    per_spec = _aggregate_per_spec(run, store, configs_dir)
    cells = [
        DefectHeatmapCell(
            spec_id=p.spec_id,
            category=p.category,
            fail_count=p.fail_count,
            sample_count=p.sample_count,
            pass_rate=p.score,
        )
        for p in per_spec
    ]
    return DefectHeatmapResponse(run_id=run_id, cells=cells)


def _aggregate_per_spec(
    run: BenchmarkRun,
    store: FileResultStore,
    configs_dir: Path,
) -> list[PerSpecResultOut]:
    """Aggregate per-spec verdict stats across all sample results."""
    from collections import defaultdict

    # spec_id → {YES: int, NO: int, NA: int}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"YES": 0, "NO": 0, "NA": 0})

    for sr in store.iter_sample_results(run.run_id):
        if sr.eval_result and sr.eval_result.per_spec:
            for spec_id, verdict in sr.eval_result.per_spec.items():
                counts[spec_id][verdict] += 1

    if not counts:
        return []

    # Load scenario to get spec texts
    spec_texts: dict[str, tuple[str, str]] = {}  # spec_id → (text, category)
    try:
        from mmalign.loaders.scenarios import load_scenario
        origin = (run.metadata or {}).get("benchmark_origin")
        scenario, _ = load_scenario(
            run.scenario,
            origin=origin if origin in ("builtin", "user") else None,
            sample_limit=0,
            workspace_start=configs_dir.parent,
        )
        for spec in scenario.all_specs:
            spec_texts[spec.spec_id] = (spec.text, spec.category)
    except Exception:
        pass

    results: list[PerSpecResultOut] = []
    for spec_id, c in sorted(counts.items()):
        total = c["YES"] + c["NO"] + c["NA"]
        fail = c["NO"]
        score = c["YES"] / total if total > 0 else 0.0
        majority = max(c, key=lambda k: c[k])
        text, cat = spec_texts.get(spec_id, ("", "safety" if "safety" in spec_id else "behavioral"))
        results.append(PerSpecResultOut(
            spec_id=spec_id,
            spec_text=text,
            category=cat,
            verdict=majority,
            score=round(score * 100, 2),
            sample_count=total,
            fail_count=fail,
        ))
    return results


# ---------------------------------------------------------------------------
# Start benchmark run (async background)
# ---------------------------------------------------------------------------

@router.post("/benchmark/run", response_model=StartBenchmarkResponse, status_code=202)
async def start_run(
    req: StartBenchmarkRequest,
    configs_dir: Path = Depends(get_configs_dir),
    results_dir: Path = Depends(get_results_dir),
) -> StartBenchmarkResponse:
    run_id = new_run_id()

    # Create pending run record immediately
    from mmalign.core.result import BenchmarkRun
    from mmalign.core.store import FileResultStore as FRS
    store = FRS(results_dir)
    pending = BenchmarkRun(
        run_id=run_id,
        status="pending",
        scenario=req.scenario,
        model=req.model,
        method=req.method,
        evaluator=req.evaluator,
        sample_limit=req.sample_limit,
        metadata={"benchmark_origin": req.benchmark_origin} if req.benchmark_origin else {},
    )
    store.save_run(pending)

    # Launch engine in background
    asyncio.create_task(
        _run_benchmark_task(
            run_id=run_id,
            req=req,
            configs_dir=configs_dir,
            results_dir=results_dir,
        )
    )

    return StartBenchmarkResponse(run_id=run_id)


# ---------------------------------------------------------------------------
# Upload external results
# ---------------------------------------------------------------------------

@router.post("/results/upload", response_model=UploadResultsResponse, status_code=201)
async def upload_results(
    req: UploadResultsRequest,
    results_dir: Path = Depends(get_results_dir),
) -> UploadResultsResponse:
    """Accept benchmark results from an external runner (e.g. CLI) into the store.

    The caller supplies a complete run_id, model/scenario/method/evaluator,
    a summary dict, and an optional list of per-sample result dicts.  The API
    writes these into the ResultStore so the WebUI can display them.
    """
    from mmalign.core.result import BenchmarkRun, RunSummary, SampleResult
    from mmalign.core.store import FileResultStore as FRS

    store = FRS(results_dir)

    # Build summary
    summary: RunSummary | None = None
    try:
        s = req.summary
        summary = RunSummary(
            safety_score=float(s.get("safety_score", 0.0)),
            behavioral_score=float(s.get("behavioral_score", 0.0)),
            total_score=float(s.get("total_score", 0.0)),
            unsafe_rate=float(s.get("unsafe_rate", 0.0)),
            capability_rate=s.get("capability_rate"),
            sample_count=int(s.get("sample_count", len(req.samples))),
            error_count=int(s.get("error_count", 0)),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_SUMMARY", "message": str(exc)},
        ) from exc

    samples: list[SampleResult] = []
    for raw_sample in req.samples:
        try:
            samples.append(
                SampleResult.model_validate(
                    {
                        "run_id": req.run_id,
                        "scenario_id": req.scenario,
                        "model": req.model,
                        "method": req.method,
                        "evaluator": req.evaluator,
                        **raw_sample,
                    }
                )
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_SAMPLE",
                    "message": f"Invalid sample for run {req.run_id}: {exc}",
                },
            ) from exc

    try:
        store.delete_run(req.run_id)
    except Exception:
        pass

    run = BenchmarkRun(
        run_id=req.run_id,
        status="completed",
        scenario=req.scenario,
        model=req.model,
        method=req.method,
        evaluator=req.evaluator,
        sample_limit=summary.sample_count if summary else None,
        summary=summary,
    )
    store.save_run(run)

    for sample in samples:
        store.append_sample_result(sample)

    return UploadResultsResponse(run_id=req.run_id, ok=True)


async def _run_benchmark_task(
    run_id: str,
    req: StartBenchmarkRequest,
    configs_dir: Path,
    results_dir: Path,
) -> None:
    """Background coroutine that runs the full benchmark."""
    from mmalign.engines.diagnosis.runner import DiagnosisEngine
    from mmalign.core.store import FileResultStore as FRS

    engine = DiagnosisEngine(configs_dir=configs_dir, results_dir=results_dir)
    store = FRS(results_dir)
    try:
        await engine.run_benchmark(
            run_id=run_id,
            model=req.model,
            scenario=req.scenario,
            method=req.method,
            evaluator=req.evaluator,
            benchmark_origin=req.benchmark_origin,
            sample_limit=req.sample_limit,
            result_store=store,
        )
    except Exception:
        logger.exception("Benchmark background task failed for run_id=%s", run_id)
        try:
            run = store.get_run(run_id)
        except Exception:
            run = BenchmarkRun(
                run_id=run_id,
                status="pending",
                scenario=req.scenario,
                model=req.model,
                method=req.method,
                evaluator=req.evaluator,
                sample_limit=req.sample_limit,
                metadata={"benchmark_origin": req.benchmark_origin}
                if req.benchmark_origin
                else {},
            )
        failed = run.model_copy(
            update={
                "status": "failed",
                "completed_at": datetime.now(tz=timezone.utc),
                "metadata": {
                    **run.metadata,
                    "error": traceback.format_exc(),
                },
            }
        )
        store.save_run(failed)
