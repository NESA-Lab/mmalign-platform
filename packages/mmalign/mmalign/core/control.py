"""ControlStrategy Protocol and supporting data models.

The Correction Engine (Stage 6) owns the full implementation.  This module
provides only the Protocol signature and the data models shared across Core
and Application layers so that other Core modules can reference them without
circular imports.

Risk level ladder
-----------------
    safe   → passthrough, no intervention
    low    → spec prompting (vanilla strategy)
    medium → TTD method (align3 / self_refine)
    high   → semantic rewrite + safety reconstruction; refusal fallback

The 4-level ladder avoids over-intervention on low-risk prompts (alignment
tax) while guaranteeing a refusal path for high-risk content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field

from .types import GenerationRequest, GenerationResult

if TYPE_CHECKING:
    from .backend import ModelBackend
    from .scenario import Scenario

RiskLevel = Literal["safe", "low", "medium", "high"]


class RiskAssessment(BaseModel):
    """Result of the risk detection step."""
    level: RiskLevel
    score: float
    """Raw risk score in [0, 1]."""
    reason: str = ""
    """Human-readable explanation of why this risk level was assigned."""
    triggered_specs: list[str] = Field(default_factory=list)
    """Spec IDs predicted to be at risk, used by the strategy router."""
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyMapping(BaseModel):
    """Maps each risk level to a named strategy."""
    safe: str = "passthrough"
    low: str = "vanilla"
    medium: str = "align3"
    high: str = "spec_rewrite"


def _default_risk_thresholds() -> dict[RiskLevel, float]:
    return {"safe": 0.2, "low": 0.5, "medium": 0.8, "high": 1.0}


class ControlPolicy(BaseModel):
    """Per-scenario compliance policy configured by operators via WebUI/API.

    ``risk_thresholds`` are the score boundaries that determine the risk
    level bucket.  ``strategy_mapping`` determines which registered
    ``ControlStrategy`` is invoked at each risk level.

    Built-in strategy names (``passthrough``, ``vanilla``, ``align3``,
    ``self_refine``, ``spec_rewrite``, ``refusal``) cannot be overwritten;
    operators can only update ``strategy_mapping`` to point to custom
    registered strategies.
    """
    scenario_id: str
    risk_thresholds: dict[RiskLevel, float] = Field(default_factory=_default_risk_thresholds)
    strategy_mapping: StrategyMapping = Field(default_factory=StrategyMapping)
    max_attempts: int = 3
    """Maximum number of correction attempts before falling back to refusal."""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def strategy_for(self, level: RiskLevel) -> str:
        return cast(str, getattr(self.strategy_mapping, level))

    def risk_level_for_score(self, score: float) -> RiskLevel:
        """Bucket a raw risk score into a RiskLevel using threshold boundaries."""
        if score <= self.risk_thresholds.get("safe", 0.2):
            return "safe"
        if score <= self.risk_thresholds.get("low", 0.5):
            return "low"
        if score <= self.risk_thresholds.get("medium", 0.8):
            return "medium"
        return "high"


class ControlAttempt(BaseModel):
    """A single attempt by a ControlStrategy to produce a compliant response."""
    strategy: str
    request: GenerationRequest
    response: GenerationResult
    risk_level: RiskLevel
    attempt_number: int = 1
    passed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ControlStrategy(Protocol):
    """Online compliance control strategy.

    Produces one candidate response per call.  The CorrectionEngine's
    controller loop may invoke multiple strategies in sequence (retry with
    escalation) until ``passed=True`` or ``max_attempts`` is exhausted.

    Implementations can directly call ``backend.generate`` (e.g. passthrough,
    refusal) or wrap an ``InferenceAlignMethod`` (e.g. align3-based strategy).
    """

    name: str
    supported_modalities: set[str]

    async def generate(
        self,
        request: GenerationRequest,
        backend: "ModelBackend",
        scenario: "Scenario",
        risk: RiskAssessment,
        policy: ControlPolicy,
    ) -> ControlAttempt: ...
