"""API-layer Pydantic schemas — adapter between Core models and WebUI types.

Core models use snake_case datetime fields; the WebUI expects ISO-8601 strings
and slightly different field names (e.g. ``started_at`` → ``created_at``,
``sample_limit`` → ``samples``).  All mapping lives here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared error format (mirrors API_SPEC.md)
# ---------------------------------------------------------------------------

class ApiError(BaseModel):
    error: dict[str, Any]

    @classmethod
    def make(cls, code: str, message: str, status: int = 400) -> "ApiError":
        return cls(error={"code": code, "message": message, "status": status})


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class RunSummaryOut(BaseModel):
    safety_score: float
    behavioral_score: float
    total_score: float
    unsafe_rate: float
    capability_rate: float | None = None
    sample_count: int
    error_count: int = 0


class BenchmarkRunSummaryOut(BaseModel):
    run_id: str
    created_at: str        # = started_at.isoformat()
    status: str
    model: str
    scenario: str
    method: str
    evaluator: str
    samples: int           # = summary.sample_count or sample_limit or 0
    summary: RunSummaryOut | None = None


class PerSpecResultOut(BaseModel):
    spec_id: str
    spec_text: str
    category: str
    verdict: str           # majority verdict across samples ("YES"/"NO"/"NA")
    score: float           # fraction of YES verdicts
    sample_count: int
    fail_count: int        # count of "NO"


class BenchmarkRunDetailOut(BenchmarkRunSummaryOut):
    per_spec: list[PerSpecResultOut] = Field(default_factory=list)


class BenchmarkListOut(BaseModel):
    runs: list[BenchmarkRunSummaryOut]
    total: int


class StartBenchmarkRequest(BaseModel):
    model: str
    scenario: str
    benchmark_origin: Literal["builtin", "user"] | None = None
    method: str = "align3"
    evaluator: str = "rule_based"
    sample_limit: int | None = None


class StartBenchmarkResponse(BaseModel):
    run_id: str
    status: Literal["pending"] = "pending"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class ModelOut(BaseModel):
    id: str
    provider: str
    input_modalities: list[str]
    output_modalities: list[str]
    status: str = "online"


class ModelsResponse(BaseModel):
    models: list[ModelOut]


class MethodOut(BaseModel):
    name: str
    type: str = "inference"
    source: str = "native"
    supported_modalities: list[str] = Field(default_factory=list)
    description: str = ""


class MethodsResponse(BaseModel):
    methods: list[MethodOut]


class EvaluatorOut(BaseModel):
    id: str
    type: str
    description: str = ""
    requires_api_key: bool = False


class EvaluatorsResponse(BaseModel):
    evaluators: list[EvaluatorOut]


class SpecificationOut(BaseModel):
    spec_id: str
    category: str
    text: str
    tags: list[str] = Field(default_factory=list)


class ScenarioOut(BaseModel):
    id: str
    origin: Literal["builtin", "user"]
    name: str
    description: str
    modality: str
    language: str
    safety_spec_count: int
    behavioral_spec_count: int
    sample_count: int
    source: str
    specs: list[SpecificationOut] | None = None


class CreateUserBenchmarkRequest(BaseModel):
    """Create a self-contained user benchmark package under data/benchmarks/user/."""

    name: str
    description: str = ""
    language: str = "zh"
    modality: str = "text"
    source: str = "custom"
    safety_specifications: list[str] = Field(default_factory=list)
    behavioral_specifications: list[str] = Field(default_factory=list)
    prompts: list[dict[str, Any]] = Field(default_factory=list)
    id: str | None = None


class CreateUserBenchmarkResponse(BaseModel):
    id: str
    origin: Literal["user"] = "user"
    ok: bool = True


class ScenariosListResponse(BaseModel):
    scenarios: list[ScenarioOut]


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

class IndicatorStatus(BaseModel):
    name: str
    value: float
    unit: str
    trend: str = "flat"
    delta: str = ""
    target_mid: float | None = None
    target_final: float | None = None
    direction: str | None = None
    source: str = ""
    description: str = ""


class MetricsResponse(BaseModel):
    indicators: dict[str, IndicatorStatus]
    updated_at: str


class ControllerSeriesPoint(BaseModel):
    date: str
    intercept: float
    false_positive: float
    unsafe_rate: float


class ControllerSeriesResponse(BaseModel):
    series: list[ControllerSeriesPoint]


class HeatmapScore(BaseModel):
    scenario: str
    model: str
    safety: float
    behavioral: float
    consistency: float


class HeatmapResponse(BaseModel):
    scores: list[HeatmapScore]


class OverviewAlert(BaseModel):
    id: str
    level: str
    message: str
    time: str


class AlertsResponse(BaseModel):
    alerts: list[OverviewAlert]


class RecentRun(BaseModel):
    id: str
    time: str
    model: str
    scenario: str
    method: str
    safety_score: float
    behavioral_score: float
    consistency_score: float
    samples: int


class RecentRunsResponse(BaseModel):
    runs: list[RecentRun]


class ProjectMetric(BaseModel):
    id: str
    name: str
    current: float
    target_mid: float
    target_final: float
    unit: str
    description: str
    evidence: str = ""


class ProjectQualityMetric(BaseModel):
    id: str
    name: str
    value: float
    unit: str
    target_mid: float
    target_final: float
    direction: str
    source: str = ""


class ProjectMilestone(BaseModel):
    title: str
    date: str
    status: str = "wait"
    description: str = ""


class ProjectBoardResponse(BaseModel):
    progress: list[ProjectMetric]
    quality: list[ProjectQualityMetric]
    milestones: list[ProjectMilestone]


# ---------------------------------------------------------------------------
# Metric history
# ---------------------------------------------------------------------------

class MetricHistoryPoint(BaseModel):
    timestamp: str
    value: float


class MetricHistoryResponse(BaseModel):
    indicator_id: str
    history: list[MetricHistoryPoint]


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

class ReviewItemOut(BaseModel):
    sample_id: str
    scenario: str
    model: str
    input_text: str
    output_text: str
    auto_eval: dict[str, Any]
    queue_reason: str
    risk_level: str
    priority: str
    created_at: str


class ReviewQueueResponse(BaseModel):
    items: list[ReviewItemOut]


class ReviewAnnotationIn(BaseModel):
    sample_id: str
    reviewer: str
    per_spec: dict[str, dict[str, Any]]
    comment: str | None = None


class ReviewSubmitResponse(BaseModel):
    ok: bool
    agreement_with_auto: float


class AgreementStats(BaseModel):
    total_reviewed: int
    agreement_rate: float
    kappa: float
    hitl_accuracy: float = 0.0
    by_scenario: list[dict[str, Any]]
    by_reviewer: list[dict[str, Any]]


class DefectHeatmapCell(BaseModel):
    spec_id: str
    category: str
    fail_count: int
    sample_count: int
    pass_rate: float


class DefectHeatmapResponse(BaseModel):
    run_id: str
    cells: list[DefectHeatmapCell]


# ---------------------------------------------------------------------------
# Results upload
# ---------------------------------------------------------------------------

class UploadResultsRequest(BaseModel):
    run_id: str
    model: str
    scenario: str
    method: str
    evaluator: str
    summary: dict[str, Any]
    samples: list[dict[str, Any]] = Field(default_factory=list)


class UploadResultsResponse(BaseModel):
    run_id: str
    ok: bool


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ControllerGenerateRequest(BaseModel):
    model: str
    scenario: str
    input: str
    method: str = "align3"
    benchmark_origin: Literal["builtin", "user"] | None = "builtin"
    debug: bool = False


class RiskAssessmentOut(BaseModel):
    level: str
    score: float
    reason: str = ""
    triggered_specs: list[str] = Field(default_factory=list)


class ControlAttemptOut(BaseModel):
    strategy: str
    risk_level: str
    passed: bool
    attempt_number: int = 1
    response_text: str = ""


class ControllerGenerateResponse(BaseModel):
    raw_text: str
    aligned_text: str
    risk: RiskAssessmentOut
    strategy: str
    attempts: list[ControlAttemptOut] = Field(default_factory=list)
    latency_ms: float = 0.0
    debug_trace: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

class CompareMetricPair(BaseModel):
    a: float
    b: float


class CompareResult(BaseModel):
    method_a: str
    method_b: str
    scenario: str
    metrics: dict[str, CompareMetricPair]
    tradeoff_points: list[dict[str, Any]] = Field(default_factory=list)
    per_scenario: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------

class ConsistencyMatrixRow(BaseModel):
    model: str
    text_image: float | None = None
    text_video: float | None = None
    image_video: float | None = None
    overall: float


class ConsistencyMatrixResponse(BaseModel):
    rows: list[ConsistencyMatrixRow]
    scenario: str


class ContradictionSample(BaseModel):
    sample_id: str
    prompt_text: str
    modality_a: str
    modality_b: str
    verdict_a: str
    verdict_b: str
    scenario: str
    model: str
    severity: str


class ContradictionsResponse(BaseModel):
    items: list[ContradictionSample]
    total: int


# ---------------------------------------------------------------------------
# Monitor / Drift
# ---------------------------------------------------------------------------

class DriftPoint(BaseModel):
    date: str
    safety_score: float
    behavioral_score: float
    unsafe_rate: float
    capability_rate: float
    drift_detected: bool = False


class DriftAlert(BaseModel):
    id: str
    level: str
    scenario: str
    model: str
    metric: str
    message: str
    detected_at: str
    baseline: float
    current: float
    change_pct: float


class DriftResponse(BaseModel):
    series: list[DriftPoint]
    alerts: list[DriftAlert]


# ---------------------------------------------------------------------------
# Monitor / Calibrate
# ---------------------------------------------------------------------------

class CalibrateRequest(BaseModel):
    scenario: str
    model: str
    method: str = "prompt_tuning"   # "prompt_tuning" | "lora"
    target_metric: str = "safety_score"
    target_threshold: float = 0.80
    calibration_method: str = "prompt_tuning"   # alias kept for compat


class CalibrateResponse(BaseModel):
    run_id: str
    method: str
    scenario: str
    model: str
    before: float
    after: float | None
    status: str     # "completed" | "pending" | "failed"
    message: str = ""
    started_at: str
    completed_at: str | None


# ---------------------------------------------------------------------------
# Monitor / Feedback loop
# ---------------------------------------------------------------------------

class FeedbackLoopRequest(BaseModel):
    scenario: str
    model: str
    target_threshold: float = 0.80
    days: int = 14
    calibration_method: str = "prompt_tuning"


class FeedbackLoopResponse(BaseModel):
    loop_id: str
    scenario: str
    model: str
    drift_detected: bool
    calibration: CalibrateResponse | None
    validation_passed: bool | None
    status: str
    started_at: str
    completed_at: str | None
