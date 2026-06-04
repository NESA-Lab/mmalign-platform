"""Review routes — delegates to HumanReviewEngine and AgreementCalculator."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Query

from mmalign_api.deps import get_results_dir, get_store
from mmalign_api.schemas import (
    AgreementStats,
    ReviewAnnotationIn,
    ReviewItemOut,
    ReviewQueueResponse,
    ReviewSubmitResponse,
)
from mmalign.core.result import HumanReviewLabel, SpecLabel
from mmalign.core.store import FileResultStore
from mmalign.engines.diagnosis.agreement import AgreementCalculator
from mmalign.engines.diagnosis.human_review import HumanReviewEngine

router = APIRouter(tags=["review"])


@router.get("/review/queue", response_model=ReviewQueueResponse)
async def get_review_queue(
    scenario: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    store: FileResultStore = Depends(get_store),
) -> ReviewQueueResponse:
    engine = HumanReviewEngine()
    items = engine.build_queue(store, scenario=scenario, priority=priority, limit=limit)
    return ReviewQueueResponse(
        items=[
            ReviewItemOut(
                sample_id=i.sample_id,
                scenario=i.scenario,
                model=i.model,
                input_text=i.input_text,
                output_text=i.output_text,
                auto_eval=i.auto_eval,
                queue_reason=i.queue_reason,
                risk_level=i.risk_level,
                priority=i.priority,
                created_at=i.created_at,
            )
            for i in items
        ]
    )


@router.post("/review/submit", response_model=ReviewSubmitResponse)
async def submit_annotation(
    annotation: ReviewAnnotationIn,
    results_dir: Path = Depends(get_results_dir),
) -> ReviewSubmitResponse:
    store = FileResultStore(results_dir)
    engine = HumanReviewEngine()
    run_id = _find_run_id(store, annotation.sample_id)

    queue_reason = "audit"
    if run_id:
        for item in engine.build_queue(store, limit=500):
            if item.sample_id == annotation.sample_id:
                queue_reason = item.queue_reason
                break

        label = HumanReviewLabel(
            run_id=run_id,
            sample_id=annotation.sample_id,
            reviewer_id=annotation.reviewer,
            queue_reason=queue_reason,  # type: ignore[arg-type]
            per_spec=[
                SpecLabel(
                    spec_id=spec_id,
                    verdict=entry.get("verdict", "NA"),
                    confidence=entry.get("confidence"),
                    note=entry.get("note"),
                )
                for spec_id, entry in annotation.per_spec.items()
            ],
            comment=annotation.comment,
        )
        store.append_review_label(label)

    calc = AgreementCalculator(store)
    if run_id:
        agree, total, rate = calc.agreement_for_label(
            HumanReviewLabel(
                run_id=run_id,
                sample_id=annotation.sample_id,
                reviewer_id=annotation.reviewer,
                queue_reason="audit",
                per_spec=[
                    SpecLabel(
                        spec_id=s,
                        verdict=e.get("verdict", "NA"),
                        confidence=e.get("confidence"),
                    )
                    for s, e in annotation.per_spec.items()
                ],
            )
        )
        agreement = rate if total > 0 else 0.0
    else:
        agreement = 0.0

    return ReviewSubmitResponse(ok=True, agreement_with_auto=round(agreement, 4))


@router.get("/review/agreement", response_model=AgreementStats)
async def get_agreement_stats(
    store: FileResultStore = Depends(get_store),
) -> AgreementStats:
    stats = AgreementCalculator(store).compute_stats()
    return AgreementStats(
        total_reviewed=stats.total_reviewed,
        agreement_rate=stats.agreement_rate,
        kappa=stats.kappa,
        hitl_accuracy=stats.hitl_accuracy,
        by_scenario=stats.by_scenario,
        by_reviewer=stats.by_reviewer,
    )


def _find_run_id(store: FileResultStore, sample_id: str) -> str | None:
    runs, _ = store.list_runs(status="completed", limit=200)
    for run_summary in runs:
        try:
            for sr in store.iter_sample_results(run_summary.run_id):
                if sr.sample_id == sample_id:
                    return run_summary.run_id
        except Exception:
            continue
    return None
