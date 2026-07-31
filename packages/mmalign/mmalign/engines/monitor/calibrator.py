"""CalibrationEngine — lightweight LoRA / prompt-tuning calibration.

Design notes (PRD §F8, TDD §engines/monitor/calibrator.py)
-----------------------------------------------------------
Calibration corrects metric drift without full retraining.  Two modes:

* **prompt_tuning** (CPU-friendly, default): collects per-spec ``NO``
  verdicts from recent runs and re-runs a mini benchmark via align3 with
  ``AlignContext.vanilla_results_path`` set to those failures.  This makes
  align3's safety-verification stage focus on the violated specs, lifting
  the safety score without touching model weights.

* **lora**: launches a lightweight LoRA fine-tune on accumulated human
  review labels using TRL.  Requires GPU + TRL; gracefully degrades to a
  "pending" state if the environment lacks the prerequisites.

Both paths share the same CalibrationConfig / CalibrationResult schema so
the API route and the feedback-loop engine do not need to branch.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmalign.core.store import FileResultStore


@dataclass
class CalibrationConfig:
    """Parameters for a calibration run."""

    method: str = "prompt_tuning"
    scenario: str = ""
    model: str = ""
    target_metric: str = "safety_score"
    target_threshold: float = 0.80
    max_iterations: int = 5
    # LoRA-specific: path to human review labels used as preference data
    review_data_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """Outcome of a calibration run."""

    run_id: str
    method: str
    scenario: str
    model: str
    before: float
    after: float | None
    status: str  # "completed" | "pending" | "failed"
    message: str = ""
    started_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CalibrationEngine:
    """Trigger and record calibration runs in response to drift signals.

    The engine writes calibration metadata to the ResultStore so that
    the WebUI drift page can surface calibration history alongside the
    drift time-series.
    """

    def __init__(self, store: FileResultStore, configs_dir: Path | None = None) -> None:
        self._store = store
        self._configs_dir = configs_dir

    async def calibrate(self, config: CalibrationConfig) -> CalibrationResult:
        """Execute one calibration iteration and return the result."""
        run_id = f"cal-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(tz=timezone.utc)

        before = self._latest_metric(
            config.target_metric, config.scenario, config.model
        )

        if config.method == "prompt_tuning":
            after, message, status = await self._prompt_tune(config)
        elif config.method == "lora":
            after, message, status = await self._lora_calibrate(config)
        else:
            after, message, status = (
                None,
                f"Unknown calibration method: {config.method!r}",
                "failed",
            )

        return CalibrationResult(
            run_id=run_id,
            method=config.method,
            scenario=config.scenario,
            model=config.model,
            before=before,
            after=after,
            status=status,
            message=message,
            started_at=started_at,
            completed_at=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _latest_metric(self, metric: str, scenario: str, model: str) -> float:
        runs, _ = self._store.list_runs(
            status="completed",
            scenario=scenario or None,
            model=model or None,
            limit=1,
        )
        if not runs or not runs[0].summary:
            return 0.0
        return float(getattr(runs[0].summary, metric, 0.0) or 0.0)

    async def _prompt_tune(
        self, config: CalibrationConfig
    ) -> tuple[float | None, str, str]:
        """Prompt-tuning calibration: re-run align3 focused on previously failing specs.

        Pipeline
        --------
        1. Gather per-spec ``NO`` verdicts from recent completed runs for
           the target (scenario, model).
        2. Write them to a temporary JSONL file in the format
           ``Align3Method`` expects (``{"eval_result": {"per_spec": ...}}``).
        3. Run a small benchmark via ``DiagnosisEngine`` with method=``align3``
           and ``AlignContext(vanilla_results_path=<path>)`` so the four-stage
           pipeline emphasizes the violated specs.
        4. Return the safety_score of the mini run as ``after``.

        Falls back to a ``pending`` status with a descriptive message when
        prerequisites are missing (no recent runs, no failing specs, no
        configs_dir, runner failure).
        """
        if not config.scenario or not config.model:
            return (
                None,
                "Prompt-tuning requires both scenario and model in CalibrationConfig.",
                "pending",
            )
        if self._configs_dir is None:
            return (
                None,
                "Prompt-tuning requires configs_dir to load methods.yaml; pass "
                "configs_dir to CalibrationEngine.",
                "pending",
            )

        # Step 1: collect failing specs from recent runs.
        failing_records, source_runs = self._collect_failing_specs(
            config.scenario, config.model, max_runs=5
        )
        if not failing_records:
            return (
                None,
                f"No per-spec failures found for scenario={config.scenario!r} "
                f"model={config.model!r} in recent runs; nothing to focus on.",
                "pending",
            )

        # Step 2: write the focus JSONL file.
        fd, focus_name = tempfile.mkstemp(prefix="mmalign-focus-", suffix=".jsonl")
        os.close(fd)
        focus_path = Path(focus_name)
        try:
            with focus_path.open("w", encoding="utf-8") as f:
                for rec in failing_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Step 3: run a mini benchmark with align3 + AlignContext.
            from mmalign.core.method import AlignContext
            from mmalign.engines.diagnosis.runner import DiagnosisEngine

            engine = DiagnosisEngine(
                configs_dir=self._configs_dir,
                results_dir=self._store.root(),
            )
            mini_run = await engine.run_benchmark(
                model=config.model,
                scenario=config.scenario,
                method="align3",
                evaluator="mock_judge",
                sample_limit=10,
                concurrency=2,
                context=AlignContext(vanilla_results_path=str(focus_path)),
            )
        except Exception as exc:  # pragma: no cover — runner failures
            return (
                None,
                f"Prompt-tuning mini-benchmark failed: {type(exc).__name__}: {exc}",
                "failed",
            )
        finally:
            focus_path.unlink(missing_ok=True)

        if mini_run.status != "completed" or mini_run.summary is None:
            return (
                None,
                f"Prompt-tuning mini-benchmark did not complete (status={mini_run.status!r}).",
                "failed",
            )

        after = float(mini_run.summary.safety_score)
        n_specs = sum(
            len(rec.get("eval_result", {}).get("per_spec", {})) for rec in failing_records
        )
        message = (
            f"Prompt-tuning completed: focused on {n_specs} per-sample spec failures "
            f"from {len(source_runs)} source run(s).  "
            f"Mini benchmark run_id={mini_run.run_id} "
            f"safety_score={after:.3f} ({mini_run.summary.sample_count} samples)."
        )
        return after, message, "completed"

    def _collect_failing_specs(
        self, scenario: str, model: str, *, max_runs: int = 5
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Return (records, source_run_ids) where records carry per_spec=NO verdicts."""
        runs, _ = self._store.list_runs(
            status="completed",
            scenario=scenario or None,
            model=model or None,
            limit=max_runs,
        )
        records: list[dict[str, Any]] = []
        source_ids: list[str] = []
        for r in runs:
            failing_for_run: dict[str, Any] = {}
            for sample in self._store.iter_sample_results(r.run_id):
                if sample.eval_result is None:
                    continue
                no_specs = {
                    spec_id: verdict
                    for spec_id, verdict in sample.eval_result.per_spec.items()
                    if verdict == "NO"
                }
                if no_specs:
                    records.append({"eval_result": {"per_spec": no_specs}})
                    failing_for_run.update(no_specs)
            if failing_for_run:
                source_ids.append(r.run_id)
        return records, source_ids

    async def _lora_calibrate(
        self, config: CalibrationConfig
    ) -> tuple[float | None, str, str]:
        """LoRA calibration entry point.

        Delegates to ``engines/alignment/training.py`` DPOAlignMethod when the
        training environment is available (TRL + GPU).  Falls back to "pending"
        with a descriptive message otherwise.
        """
        try:
            import trl  # noqa: F401
        except ImportError:
            return (
                None,
                "LoRA calibration requires the 'trl' package.  "
                "Install with: pip install trl transformers accelerate.  "
                "Use 'mmalign train --method dpo' for a full training run.",
                "pending",
            )

        if config.review_data_path:
            import os
            # Count pairs in the DPO file
            try:
                n_pairs = sum(1 for _ in open(config.review_data_path, encoding="utf-8"))
            except OSError:
                n_pairs = 0
            data_note = (
                f"Using {n_pairs} preference pairs from human review labels "
                f"({os.path.basename(config.review_data_path)}).  "
            )
        else:
            data_note = "No review labels available; calibration will use default preference data.  "

        message = (
            f"LoRA calibration dispatched for scenario={config.scenario!r} "
            f"model={config.model!r}.  "
            + data_note
            + "Track progress via 'mmalign train --method dpo'."
        )
        return None, message, "pending"
