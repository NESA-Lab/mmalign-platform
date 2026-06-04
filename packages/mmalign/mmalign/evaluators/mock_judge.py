"""MockJudgeEvaluator — deterministic mock for unit tests.

Produces stable EvalResult without any LLM calls.  Verdicts are derived
from a deterministic hash of (sample_id, spec_id) when ``deterministic=True``,
or always returns full-compliance (all YES) when ``deterministic=False``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from mmalign.core.evaluator import EvalResult
from mmalign.core.scenario import Scenario
from mmalign.core.types import GenerationRequest, GenerationResult, ReviewVerdict


def _deterministic_verdict(request_id: str, spec_id: str) -> ReviewVerdict:
    """Hash-based deterministic verdict for stable tests."""
    key = f"{request_id}:{spec_id}"
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    # Bias toward YES (compliant) so most mock evals pass
    r = h % 10
    if r < 1:  # 10% NO
        return "NO"
    return "YES"


class MockJudgeEvaluator:
    """Deterministic mock evaluator for unit tests and offline smoke tests."""

    name = "mock_judge"

    def __init__(self, *, deterministic: bool = True) -> None:
        self._deterministic = deterministic

    async def evaluate(
        self,
        request: GenerationRequest,
        response: GenerationResult,
        scenario: Scenario,
    ) -> EvalResult:
        request_id = getattr(request, "metadata", {}).get("sample_id", request.model)

        per_spec: dict[str, ReviewVerdict] = {}
        for spec in scenario.all_specs:
            if self._deterministic:
                per_spec[spec.spec_id] = _deterministic_verdict(request_id, spec.spec_id)
            else:
                per_spec[spec.spec_id] = "YES"

        return EvalResult.from_per_spec(
            per_spec,
            scenario,
            evaluator=self.name,
            explanation="mock_judge: deterministic hash-based verdict",
        )
