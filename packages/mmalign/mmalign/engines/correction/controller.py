"""ComplianceController — online compliance control for generation requests.

Architecture (TDD §合规生成控制器内部架构)
-----------------------------------------
    GenerationRequest
        → RiskDetector        (rule-based keyword + spec scan)
        → StrategyRouter      (risk + policy → ControlStrategy)
        → ControlStrategy     (generate candidate response)
        → PostGuard           (rule-based quick check)
        → retry / escalate    (up to max_attempts)
        → ControlledResponse  (final result + full trace)

Two equivalent entry points:
    Python SDK:  from mmalign import ComplianceController
    REST API:    POST /api/v1/controller/generate

Both share the same controller logic; the API route is a thin HTTP adapter.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mmalign.core.backend import ModelBackend
from mmalign.core.control import (
    ControlAttempt,
    ControlPolicy,
    RiskAssessment,
)
from mmalign.core.scenario import Scenario
from mmalign.core.types import GenerationRequest, GenerationResult

from .risk_detector import RuleBasedRiskDetector
from .strategy_router import StrategyRouter


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class GuardResult(BaseModel):
    passed: bool
    reason: str = ""


class ControlledResponse(BaseModel):
    """Full result of a ComplianceController.generate() call.

    ``response`` is the final compliant response.
    ``raw_response`` is the unmodified backend response (None for passthrough).
    ``trace`` is the full list of ControlAttempt records (for WebUI / debug).
    """
    response: GenerationResult
    raw_response: GenerationResult | None = None
    risk: RiskAssessment
    strategy: str
    guard: GuardResult
    attempts: list[ControlAttempt] = Field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# PostGuard
# ---------------------------------------------------------------------------

import re as _post_guard_re

_HARD_FAIL_PATTERNS = [
    _post_guard_re.compile(p, _post_guard_re.IGNORECASE | _post_guard_re.DOTALL)
    for p in [
        r"以下是.*炸弹.*制作方法",
        r"step.{0,20}(1|one).{0,30}(bomb|explosive|weapon)",
        r"Here.{0,20}how to (make|create|build|synthesize).{0,20}(drug|weapon|bomb)",
    ]
]


class _PostGuard:
    """Fast rule-based post-generation safety check."""

    def check(self, response: GenerationResult, risk: RiskAssessment) -> GuardResult:
        text = response.text()
        for pattern in _HARD_FAIL_PATTERNS:
            if pattern.search(text):
                return GuardResult(passed=False, reason=f"PostGuard: hard-fail pattern matched")
        return GuardResult(passed=True)


_guard = _PostGuard()


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ComplianceController:
    """Coordinate risk detection, strategy routing, and post-guard for a single request.

    Parameters
    ----------
    backend:
        The ModelBackend to call for generation.
    scenario:
        The compliance scenario (provides safety/behavioral specs and metadata).
    policy:
        The ControlPolicy for this scenario (risk thresholds and strategy mapping).
        Defaults to the built-in ControlPolicy if not provided.
    """

    def __init__(
        self,
        backend: ModelBackend,
        scenario: Scenario,
        policy: ControlPolicy | None = None,
    ) -> None:
        self._backend = backend
        self._scenario = scenario
        self._policy = policy or ControlPolicy(scenario_id=scenario.id)
        self._detector = RuleBasedRiskDetector()
        self._router = StrategyRouter()

    async def generate(self, request: GenerationRequest) -> ControlledResponse:
        """Run the full compliance control pipeline for a single request.

        Returns a ``ControlledResponse`` containing the final compliant
        response, the risk assessment, and the full attempt trace.
        """
        t0 = time.monotonic()
        attempts: list[ControlAttempt] = []

        # Step 1: Detect risk
        input_text = _extract_text(request)
        risk = self._detector.detect(input_text, self._scenario)

        # Re-bucket the level using the operator-configured thresholds in the
        # policy.  The detector returns a level based on its own hard-coded
        # boundaries (US-9 fast path); the policy thresholds (US-14/15) are
        # what give operators control over routing.
        bucketed_level = self._policy.risk_level_for_score(risk.score)
        if bucketed_level != risk.level:
            risk = risk.model_copy(update={"level": bucketed_level})

        # Step 2: Initial strategy
        strategy = self._router.route(risk.level, self._policy)

        # Step 3: Generate with retry + escalation
        raw_response: GenerationResult | None = None
        final_attempt: ControlAttempt | None = None

        for attempt_num in range(1, self._policy.max_attempts + 1):
            attempt = await strategy.generate(
                request=request,
                backend=self._backend,
                scenario=self._scenario,
                risk=risk,
                policy=self._policy,
            )
            attempt.attempt_number = attempt_num
            attempts.append(attempt)

            # Capture raw response on first attempt if we're not in passthrough
            if attempt_num == 1 and strategy.name != "passthrough":
                raw_response = attempt.response  # save pre-guard candidate

            guard = _guard.check(attempt.response, risk)
            if guard.passed:
                final_attempt = attempt
                break

            # Escalate to stronger strategy
            strategy = self._router.escalate(strategy.name, self._policy)

        if final_attempt is None:
            # All attempts failed — force refusal
            from .strategies.refusal import RefusalStrategy
            refusal_strat = RefusalStrategy()
            fallback = await refusal_strat.generate(
                request=request,
                backend=self._backend,
                scenario=self._scenario,
                risk=risk,
                policy=self._policy,
            )
            fallback.attempt_number = len(attempts) + 1
            attempts.append(fallback)
            final_attempt = fallback
            guard = GuardResult(passed=True, reason="forced refusal after all attempts failed")

        latency_ms = (time.monotonic() - t0) * 1000

        return ControlledResponse(
            response=final_attempt.response,
            raw_response=raw_response,
            risk=risk,
            strategy=final_attempt.strategy,
            guard=guard,
            attempts=attempts,
            latency_ms=round(latency_ms, 1),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(request: GenerationRequest) -> str:
    inp = request.input
    if isinstance(inp, str):
        return inp
    parts: list[str] = []
    for msg in inp:
        if isinstance(msg.content, str):
            parts.append(msg.content)
        else:
            for part in msg.content:
                if getattr(part, "type", None) == "input_text" and part.text:
                    parts.append(part.text)
    return " ".join(parts)


def load_controller_from_config(
    model: str,
    scenario_id: str,
    configs_dir: Path,
    workspace_start: Path | None = None,
) -> ComplianceController:
    """Factory: load backend + scenario + policy from YAML configs.

    Convenience function for CLI and API routes.  The ``ControlPolicy`` is
    loaded from ``configs_dir/policies.yaml``; missing scenarios fall back
    to the built-in default.
    """
    from mmalign.backends.loader import load_backends
    from mmalign.loaders.policies import load_policy_from_yaml
    from mmalign.loaders.scenarios import load_scenario

    backends = load_backends(configs_dir / "models.yaml")
    backend = backends[model]
    scenario, _ = load_scenario(
        scenario_id,
        origin=None,
        sample_limit=0,
        workspace_start=workspace_start or configs_dir.parent,
    )
    policy = load_policy_from_yaml(scenario_id, configs_dir)
    return ComplianceController(backend=backend, scenario=scenario, policy=policy)
