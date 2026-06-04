"""Core Layer: Protocol definitions, data models, and registries.

Import order must respect module dependencies:
  types  →  (standalone)
  scenario → types
  registry → (standalone)
  backend  → types
  evaluator → types, scenario
  method   → types, scenario, backend
  control  → types, scenario, backend
  metrics  → (standalone)
"""

from .types import (
    Modality,
    InputContentPart,
    OutputContentPart,
    InputMessage,
    MMAlignRequestOptions,
    GenerationRequest,
    GenerationOutput,
    GenerationResult,
    BackendCapabilities,
)
from .scenario import Specification, SpecCategory, Scenario, Sample
from .registry import Registry, BuiltinOverrideError
from .backend import ModelBackend
from .evaluator import EvalResult, Evaluator, compute_sar
from .method import AlignContext, AlignedResponse, InferenceAlignMethod, TrainingAlignMethod
from .control import RiskLevel, RiskAssessment, ControlPolicy, ControlAttempt, ControlStrategy
from .metrics import MetricSpec, METRIC_SPECS
from .result import (
    RunStatus,
    RunSummary,
    BackendTelemetry,
    SampleResult,
    SpecLabel,
    HumanReviewLabel,
    BenchmarkRun,
    BenchmarkRunSummary,
    compute_run_summary,
)
from .store import ResultStore, FileResultStore, RunNotFoundError, new_run_id

__all__ = [
    # types
    "Modality",
    "InputContentPart",
    "OutputContentPart",
    "InputMessage",
    "MMAlignRequestOptions",
    "GenerationRequest",
    "GenerationOutput",
    "GenerationResult",
    "BackendCapabilities",
    # scenario
    "Specification",
    "SpecCategory",
    "Scenario",
    "Sample",
    # registry
    "Registry",
    "BuiltinOverrideError",
    # backend
    "ModelBackend",
    # evaluator
    "EvalResult",
    "Evaluator",
    "compute_sar",
    # method
    "AlignContext",
    "AlignedResponse",
    "InferenceAlignMethod",
    "TrainingAlignMethod",
    # control
    "RiskLevel",
    "RiskAssessment",
    "ControlPolicy",
    "ControlAttempt",
    "ControlStrategy",
    # metrics
    "MetricSpec",
    "METRIC_SPECS",
    # result
    "RunStatus",
    "RunSummary",
    "BackendTelemetry",
    "SampleResult",
    "SpecLabel",
    "HumanReviewLabel",
    "BenchmarkRun",
    "BenchmarkRunSummary",
    "compute_run_summary",
    # store
    "ResultStore",
    "FileResultStore",
    "RunNotFoundError",
    "new_run_id",
]
