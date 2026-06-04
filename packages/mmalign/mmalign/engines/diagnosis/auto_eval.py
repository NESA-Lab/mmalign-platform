"""AutoEvaluator — concurrent evaluation wrapper over an Evaluator.

Runs evaluations concurrently up to the given semaphore limit.
Errors are caught per-sample and returned as failed EvalResult metadata
rather than propagating, so a single judge timeout cannot abort the run.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mmalign.core.evaluator import Evaluator, EvalResult
from mmalign.core.scenario import Scenario
from mmalign.core.types import GenerationRequest, GenerationResult


class AutoEvaluator:
    """Wraps an Evaluator to run concurrently over a batch of samples."""

    def __init__(self, evaluator: Evaluator, *, concurrency: int = 4) -> None:
        self._evaluator = evaluator
        self._sem = asyncio.Semaphore(concurrency)

    async def evaluate_one(
        self,
        request: GenerationRequest,
        response: GenerationResult,
        scenario: Scenario,
    ) -> tuple[EvalResult | None, str | None]:
        """Evaluate a single sample; returns (result, error_code)."""
        async with self._sem:
            try:
                result = await self._evaluator.evaluate(request, response, scenario)
                return result, None
            except Exception as exc:
                return None, f"eval_error:{type(exc).__name__}:{exc}"

    async def evaluate_batch(
        self,
        items: list[tuple[GenerationRequest, GenerationResult]],
        scenario: Scenario,
    ) -> list[tuple[EvalResult | None, str | None]]:
        """Evaluate a batch concurrently; order-preserving."""
        tasks = [
            asyncio.create_task(self.evaluate_one(req, resp, scenario))
            for req, resp in items
        ]
        return list(await asyncio.gather(*tasks))
