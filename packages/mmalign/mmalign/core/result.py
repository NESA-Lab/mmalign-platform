"""Benchmark result data models — the single source of truth for all outputs.

CLI, REST API, and WebUI all share these types.  ``BenchmarkRun`` is the
top-level container; ``SampleResult`` holds per-sample generation +
evaluation output; ``HumanReviewLabel`` carries human-in-the-loop verdicts.

Serialisation convention
------------------------
- All datetime fields are stored as UTC ISO-8601 strings via Pydantic's
  default ``datetime`` serializer.
- ``SampleResult`` is appended to JSONL one record per line so that large
  runs can be streamed without loading the entire file into memory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field, model_validator

from .types import GenerationRequest, GenerationResult
from .evaluator import EvalResult

RunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
QueueReason = Literal["low_confidence", "high_risk", "audit", "disagreement"]
ApiStyle = Literal["responses", "chat_completions", "native", "mock"]


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------

class RunSummary(BaseModel):
    """Aggregate statistics for a completed ``BenchmarkRun``."""

    safety_score: float
    """Average safety_score across all samples (after veto: 0 or 1 per sample)."""

    behavioral_score: float
    """Average behavioral_score ∈ [0, 1]."""

    total_score: float
    """Average SAR total_score ∈ [0, 1]."""

    unsafe_rate: float
    """Fraction of samples where safety_score == 0."""

    capability_rate: float | None = None
    """Retained-capability ratio (aligned / baseline); None until computed."""

    sample_count: int
    error_count: int = 0

    @model_validator(mode="after")
    def _check_rates(self) -> "RunSummary":
        for field, val in [
            ("safety_score", self.safety_score),
            ("behavioral_score", self.behavioral_score),
            ("total_score", self.total_score),
            ("unsafe_rate", self.unsafe_rate),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {val}")
        return self


class BackendTelemetry(BaseModel):
    """Runtime telemetry captured by the ModelBackend adapter."""

    provider: str
    model_name: str
    api_style: ApiStyle
    latency_ms: float | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    """Token counts; keys mirror OpenAI usage: ``input_tokens``, ``output_tokens``."""
    error_code: str | None = None
    error_message: str | None = None

    @property
    def failed(self) -> bool:
        return self.error_code is not None


# ---------------------------------------------------------------------------
# Core result types
# ---------------------------------------------------------------------------

class SampleResult(BaseModel):
    """Result for a single (sample, method, model) evaluation triple.

    ``response`` and ``eval_result`` are ``None`` for failed/skipped samples;
    callers should check ``status`` before reading them.
    """

    run_id: str
    sample_id: str
    scenario_id: str
    model: str
    method: str
    evaluator: str
    request: GenerationRequest
    response: GenerationResult | None = None
    eval_result: EvalResult | None = None
    status: Literal["completed", "failed", "skipped"] = "completed"
    telemetry: BackendTelemetry | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_unsafe(self) -> bool:
        return self.eval_result is not None and self.eval_result.safety_score == 0


class SpecLabel(BaseModel):
    """Per-specification verdict from a reviewer or evaluator."""

    spec_id: str
    verdict: Literal["YES", "NO", "NA"]
    confidence: float | None = None
    note: str | None = None


class HumanReviewLabel(BaseModel):
    """Human-in-the-loop annotation for a single sample.

    Written to ``results/reviews/{run_id}.jsonl`` by the Review Engine;
    read by MetricEngine to compute indicators 1.6 and 1.7.
    """

    run_id: str
    sample_id: str
    reviewer_id: str
    queue_reason: QueueReason
    per_spec: list[SpecLabel] = Field(default_factory=list)
    comment: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Run-level container
# ---------------------------------------------------------------------------

class BenchmarkRunSummary(BaseModel):
    """Lightweight run record for list endpoints (no per-sample data)."""

    run_id: str
    status: RunStatus
    scenario: str
    model: str
    method: str
    evaluator: str
    summary: RunSummary | None = None
    started_at: datetime
    completed_at: datetime | None = None


class BenchmarkRun(BaseModel):
    """Full benchmark run metadata + summary.

    Written to ``results/runs/{run_id}.json``.  Per-sample data is stored
    in ``results/samples/{run_id}.jsonl`` and loaded on demand.
    """

    run_id: str
    status: RunStatus = "pending"
    scenario: str
    model: str
    method: str
    evaluator: str
    dataset: str | None = None
    sample_limit: int | None = None
    summary: RunSummary | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_summary(self) -> BenchmarkRunSummary:
        return BenchmarkRunSummary(
            run_id=self.run_id,
            status=self.status,
            scenario=self.scenario,
            model=self.model,
            method=self.method,
            evaluator=self.evaluator,
            summary=self.summary,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def compute_run_summary(results: list[SampleResult]) -> RunSummary:
    """Compute ``RunSummary`` from a list of ``SampleResult`` objects.

    Only ``completed`` samples with a non-None ``eval_result`` contribute to
    score averages.  Failed/skipped samples are counted in ``error_count``.
    """
    scored = [
        (r, r.eval_result)
        for r in results
        if r.status == "completed" and r.eval_result is not None
    ]
    errors = [r for r in results if r.status in ("failed", "skipped")]

    if not scored:
        return RunSummary(
            safety_score=0.0,
            behavioral_score=0.0,
            total_score=0.0,
            unsafe_rate=0.0,
            sample_count=len(results),
            error_count=len(errors),
        )

    n = len(scored)
    avg_safety = sum(ev.safety_score for _, ev in scored) / n
    avg_behavioral = sum(ev.behavioral_score for _, ev in scored) / n
    avg_total = sum(ev.total_score for _, ev in scored) / n
    unsafe_rate = sum(1 for r, _ in scored if r.is_unsafe) / n

    return RunSummary(
        safety_score=avg_safety,
        behavioral_score=avg_behavioral,
        total_score=avg_total,
        unsafe_rate=unsafe_rate,
        sample_count=len(results),
        error_count=len(errors),
    )
