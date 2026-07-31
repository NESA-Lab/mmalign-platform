"""DiagnosisEngine — orchestrates benchmark evaluation runs.

Execution flow (TDD §DiagnosisEngine 技术设计)
-----------------------------------------------
1. Load scenario + samples from self-contained benchmark package.
2. Load model backend from models.yaml.
3. Instantiate InferenceAlignMethod from methods.yaml.
4. Instantiate Evaluator from evaluators.yaml.
5. For each sample, concurrently: method.align() → GenerationResult → evaluator.evaluate().
6. Append SampleResult to ResultStore immediately after each sample.
7. After all samples: compute RunSummary, update BenchmarkRun.status="completed".
8. Return the final BenchmarkRun.

A run that fails mid-way stores status="failed" but keeps all already-written
SampleResults in the ResultStore for partial replay.
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmalign.core.backend import ModelBackend
from mmalign.core.evaluator import Evaluator
from mmalign.core.method import AlignContext, InferenceAlignMethod
from mmalign.core.result import (
    BackendTelemetry,
    BenchmarkRun,
    RunSummary,
    SampleResult,
    compute_run_summary,
)
from mmalign.core.scenario import Sample, Scenario
from mmalign.core.store import FileResultStore, ResultStore, new_run_id
from mmalign.core.types import GenerationOutput, GenerationRequest, GenerationResult
from mmalign.core.types import OutputContentPart

from .auto_eval import AutoEvaluator


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _aligned_to_generation_result(
    aligned_output: list[OutputContentPart],
    request: GenerationRequest,
    usage: dict[str, int],
    method: str,
    thought: str,
    metadata: dict[str, Any],
) -> GenerationResult:
    return GenerationResult(
        id=f"aligned-{uuid.uuid4().hex[:8]}",
        model=request.model,
        output=[GenerationOutput(content=aligned_output)],
        usage=usage,
        metadata={"method": method, "thought": thought, **metadata},
    )


class DiagnosisEngine:
    """Application-layer orchestrator for benchmark evaluation runs."""

    def __init__(
        self,
        *,
        configs_dir: str | Path = "configs",
        results_dir: str | Path = "results",
    ) -> None:
        self._configs = Path(configs_dir)
        self._results = Path(results_dir)

    async def run_benchmark(
        self,
        *,
        model: str,
        scenario: str,
        run_id: str | None = None,
        method: str = "align3",
        evaluator: str = "llm_judge",
        benchmark_origin: str | None = None,
        dataset: str | None = None,
        sample_limit: int | None = None,
        concurrency: int = 4,
        result_store: ResultStore | None = None,
        context: AlignContext | None = None,
    ) -> BenchmarkRun:
        """Run a full benchmark evaluation and return the completed BenchmarkRun."""
        store = result_store or FileResultStore(self._results)
        run_id = run_id or new_run_id()

        run = BenchmarkRun(
            run_id=run_id,
            status="pending",
            scenario=scenario,
            model=model,
            method=method,
            evaluator=evaluator,
            dataset=dataset,
            sample_limit=sample_limit,
            metadata={"benchmark_origin": benchmark_origin} if benchmark_origin else {},
        )
        store.save_run(run)

        try:
            # Step 1: Load scenario
            from mmalign.loaders.scenarios import load_scenario
            from mmalign.loaders.benchmarks import BenchmarkOrigin

            origin: BenchmarkOrigin | None
            if benchmark_origin == "builtin":
                origin = "builtin"
            elif benchmark_origin == "user":
                origin = "user"
            else:
                origin = None
            loaded_scenario, samples = load_scenario(
                scenario,
                origin=origin,
                model=model,
                sample_limit=sample_limit,
                workspace_start=self._configs.parent,
            )

            # Step 2: Load backend
            from mmalign.backends.loader import load_backends
            backends = load_backends(self._configs / "models.yaml", only=[model])
            backend = backends[model]

            # Step 3: Load method
            from mmalign.loaders.methods import load_method
            align_method = load_method(method, self._configs / "methods.yaml")
            align_method.validate(loaded_scenario, backend)

            # Step 4: Load evaluator
            from mmalign.loaders.evaluators import load_evaluator
            eval_instance = load_evaluator(
                evaluator,
                self._configs / "evaluators.yaml",
                models_config_path=self._configs / "models.yaml",
            )
            auto_eval = AutoEvaluator(eval_instance, concurrency=concurrency)

            # Update run to running
            run = run.model_copy(update={"status": "running", "started_at": _utcnow()})
            store.save_run(run)

            # Step 5-7: Process samples concurrently
            sem = asyncio.Semaphore(concurrency)
            tasks = [
                asyncio.create_task(
                    self._process_sample(
                        sample=sample,
                        backend=backend,
                        align_method=align_method,
                        auto_eval=auto_eval,
                        scenario=loaded_scenario,
                        run_id=run_id,
                        evaluator_name=evaluator,
                        method_name=method,
                        context=context,
                        sem=sem,
                        store=store,
                    )
                )
                for sample in samples
            ]
            sample_results: list[SampleResult] = await asyncio.gather(*tasks)

            # Step 8: Compute summary
            summary = compute_run_summary(sample_results)
            run = run.model_copy(update={
                "status": "completed",
                "summary": summary,
                "completed_at": _utcnow(),
            })
            store.save_run(run)

        except Exception as exc:
            run = run.model_copy(update={
                "status": "failed",
                "completed_at": _utcnow(),
                "metadata": {**run.metadata, "error": traceback.format_exc()},
            })
            store.save_run(run)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return run

        return run

    async def _process_sample(
        self,
        *,
        sample: Sample,
        backend: ModelBackend,
        align_method: InferenceAlignMethod,
        auto_eval: AutoEvaluator,
        scenario: Scenario,
        run_id: str,
        evaluator_name: str,
        method_name: str,
        context: AlignContext | None,
        sem: asyncio.Semaphore,
        store: ResultStore,
    ) -> SampleResult:
        async with sem:
            try:
                # Inject sample_id into request metadata for evaluator tracing
                request = sample.request.model_copy(
                    update={"metadata": {**sample.request.metadata, "sample_id": sample.id}}
                )

                aligned = await align_method.align(
                    request, backend, scenario, context=context
                )
                gen_result = _aligned_to_generation_result(
                    aligned.output,
                    request,
                    aligned.usage,
                    aligned.method,
                    aligned.thought,
                    aligned.metadata,
                )

                eval_result, error_code = await auto_eval.evaluate_one(
                    request, gen_result, scenario
                )

                telemetry = BackendTelemetry(
                    provider=getattr(backend, "_provider", "unknown"),
                    model_name=getattr(backend, "_model_name", backend.model_id),
                    api_style=getattr(backend, "_api_style", "mock"),
                    usage=aligned.usage,
                    error_code=error_code,
                )

                sr = SampleResult(
                    run_id=run_id,
                    sample_id=sample.id,
                    scenario_id=scenario.id,
                    model=backend.model_id,
                    method=method_name,
                    evaluator=evaluator_name,
                    request=request,
                    response=gen_result,
                    eval_result=eval_result,
                    status="failed" if error_code else "completed",
                    telemetry=telemetry,
                    metadata={"thought": aligned.thought, **aligned.metadata},
                )
            except Exception as exc:
                sr = SampleResult(
                    run_id=run_id,
                    sample_id=sample.id,
                    scenario_id=scenario.id,
                    model=backend.model_id,
                    method=method_name,
                    evaluator=evaluator_name,
                    request=sample.request,
                    status="failed",
                    telemetry=BackendTelemetry(
                        provider="unknown",
                        model_name=backend.model_id,
                        api_style="mock",
                        error_code=f"process_error:{type(exc).__name__}",
                        error_message=str(exc),
                    ),
                )

            store.append_sample_result(sr)
            return sr
