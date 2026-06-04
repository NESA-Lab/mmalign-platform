"""Monitor routes.

GET  /monitor/drift     — time-series metrics + threshold alerts
POST /monitor/calibrate — trigger lightweight calibration (US-13)
POST /monitor/feedback  — run one full feedback-loop iteration
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query

from mmalign_api.deps import get_store, get_configs_dir
from mmalign_api.schemas import (
    CalibrateRequest,
    CalibrateResponse,
    DriftAlert,
    DriftPoint,
    DriftResponse,
    FeedbackLoopRequest,
    FeedbackLoopResponse,
)
from mmalign.core.store import FileResultStore
from mmalign.engines.monitor import (
    CalibrationConfig,
    CalibrationEngine,
    DriftEngine,
    FeedbackLoopEngine,
)

router = APIRouter(tags=["monitor"])


@router.get("/monitor/drift", response_model=DriftResponse)
async def get_drift(
    scenario: str | None = Query(default=None),
    model: str | None = Query(default=None),
    days: int = Query(default=30, ge=7, le=180),
    store: FileResultStore = Depends(get_store),
) -> DriftResponse:
    """Return daily metric time-series and drift alerts.

    Drift is detected when a metric's daily average deviates more than
    10 % from its rolling 7-day baseline.
    """
    engine = DriftEngine(store)
    series, alerts = engine.detect(days=days, scenario=scenario, model=model)

    return DriftResponse(
        series=[
            DriftPoint(
                date=p.date,
                safety_score=p.safety_score,
                behavioral_score=p.behavioral_score,
                unsafe_rate=p.unsafe_rate,
                capability_rate=p.capability_rate,
                drift_detected=p.drift_detected,
            )
            for p in series
        ],
        alerts=[
            DriftAlert(
                id=a.id,
                level=a.level,
                scenario=a.scenario,
                model=a.model,
                metric=a.metric,
                message=a.message,
                detected_at=a.detected_at,
                baseline=a.baseline,
                current=a.current,
                change_pct=a.change_pct,
            )
            for a in alerts
        ],
    )


@router.post("/monitor/calibrate", response_model=CalibrateResponse)
async def trigger_calibrate(
    body: CalibrateRequest,
    store: FileResultStore = Depends(get_store),
) -> CalibrateResponse:
    """Trigger a lightweight calibration run in response to drift (US-13).

    * ``prompt_tuning``: adjusts safety prompts for failing specs — no GPU needed.
    * ``lora``: launches LoRA fine-tune on human review labels — requires GPU + TRL.

    Returns immediately with a ``CalibrateResponse``; long-running LoRA jobs
    will return ``status="pending"`` and can be tracked via the drift endpoint.
    """
    method = body.calibration_method or body.method
    engine = CalibrationEngine(store)
    result = await engine.calibrate(
        CalibrationConfig(
            method=method,
            scenario=body.scenario,
            model=body.model,
            target_metric=body.target_metric,
            target_threshold=body.target_threshold,
        )
    )
    return CalibrateResponse(
        run_id=result.run_id,
        method=result.method,
        scenario=result.scenario,
        model=result.model,
        before=result.before,
        after=result.after,
        status=result.status,
        message=result.message,
        started_at=result.started_at.isoformat(),
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
    )


@router.post("/monitor/feedback", response_model=FeedbackLoopResponse)
async def run_feedback_loop(
    body: FeedbackLoopRequest,
    store: FileResultStore = Depends(get_store),
) -> FeedbackLoopResponse:
    """Run one iteration of the full diagnosis → calibrate → validate loop.

    Returns immediately after triggering calibration (the validate phase may
    produce ``validation_passed=None`` if calibration is "pending").
    """
    engine = FeedbackLoopEngine.from_store(store, configs_dir=get_configs_dir())
    result = await engine.run(
        scenario=body.scenario,
        model=body.model,
        target_threshold=body.target_threshold,
        days=body.days,
        calibration_method=body.calibration_method,
    )

    cal_out: CalibrateResponse | None = None
    if result.calibration is not None:
        c = result.calibration
        cal_out = CalibrateResponse(
            run_id=c.run_id,
            method=c.method,
            scenario=c.scenario,
            model=c.model,
            before=c.before,
            after=c.after,
            status=c.status,
            message=c.message,
            started_at=c.started_at.isoformat(),
            completed_at=c.completed_at.isoformat() if c.completed_at else None,
        )

    return FeedbackLoopResponse(
        loop_id=result.loop_id,
        scenario=result.scenario,
        model=result.model,
        drift_detected=result.drift_detected,
        calibration=cal_out,
        validation_passed=result.validation_passed,
        status=result.status,
        started_at=result.started_at.isoformat(),
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
    )
