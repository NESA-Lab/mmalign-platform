"""FeedbackLoopEngine — closed-loop: diagnose → calibrate → validate → deploy.

Design notes (TDD §engines/monitor/feedback.py)
-----------------------------------------------
The feedback loop ties together the other monitor sub-engines:

    1. DriftEngine.detect()  →  is there a significant drift signal?
    2. CalibrationEngine.calibrate()  →  apply lightweight correction.
    3. Validation check  →  did the calibration lift the metric above the threshold?
    4. (Future) deployment hook  →  hot-reload policies or push LoRA adapter.

Each loop iteration is idempotent: if there is no drift the loop returns
status="idle" and exits.  Results are returned (not stored) so the caller
(API route or CLI) decides what to persist.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calibrator import CalibrationConfig, CalibrationEngine, CalibrationResult
from .drift import DriftEngine
from .notifier import WebhookNotifier
from .review_export import export_review_labels_as_dpo
from mmalign.core.store import FileResultStore


@dataclass
class FeedbackLoopResult:
    """Outcome of one feedback-loop iteration."""

    loop_id: str
    scenario: str
    model: str
    drift_detected: bool
    calibration: CalibrationResult | None
    validation_passed: bool | None
    # "idle" | "drift_detected" | "calibrated" | "validated" | "deployed" | "failed"
    status: str
    started_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackLoopEngine:
    """Orchestrate the full diagnosis → calibration → validation → deploy loop.

    Typical usage (API route or scheduled task)::

        engine = FeedbackLoopEngine(store, drift_engine, calibration_engine)
        result = await engine.run(scenario="education", model="gpt-4.1")
    """

    def __init__(
        self,
        store: FileResultStore,
        drift_engine: DriftEngine,
        calibration_engine: CalibrationEngine,
        notifier: WebhookNotifier | None = None,
        configs_dir: Path | None = None,
    ) -> None:
        self._store = store
        self._drift = drift_engine
        self._calibration = calibration_engine
        self._notifier = notifier or WebhookNotifier.from_configs_dir(configs_dir)
        self._configs_dir = configs_dir

    async def run(
        self,
        scenario: str,
        model: str,
        target_threshold: float = 0.80,
        days: int = 14,
        calibration_method: str = "prompt_tuning",
    ) -> FeedbackLoopResult:
        """Execute one feedback iteration.

        Parameters
        ----------
        scenario:
            Scenario ID to monitor.
        model:
            Model ID to monitor.
        target_threshold:
            Minimum safety_score considered acceptable.
        days:
            Rolling window for drift detection.
        calibration_method:
            "prompt_tuning" (default, CPU-friendly) or "lora" (GPU required).
        """
        loop_id = f"loop-{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(tz=timezone.utc)

        # 1. Detect drift
        _series, alerts = self._drift.detect(
            days=days, scenario=scenario, model=model
        )
        drift_detected = any(a.level in ("error", "warning") for a in alerts)

        if not drift_detected:
            return FeedbackLoopResult(
                loop_id=loop_id,
                scenario=scenario,
                model=model,
                drift_detected=False,
                calibration=None,
                validation_passed=None,
                status="idle",
                started_at=started_at,
                completed_at=datetime.now(tz=timezone.utc),
            )

        # 1b. Send webhook notification for detected drift
        self._notifier.notify(alerts, scenario=scenario, model=model)

        # 2. Prepare calibration config, attaching review labels for LoRA path
        review_data_path: str | None = None
        if calibration_method == "lora":
            out = self._store.root() / "calibration_dpo_data.jsonl"
            n_pairs = export_review_labels_as_dpo(self._store, output_path=out)
            if n_pairs > 0:
                review_data_path = str(out)

        cal_config = CalibrationConfig(
            method=calibration_method,
            scenario=scenario,
            model=model,
            target_threshold=target_threshold,
            review_data_path=review_data_path,
        )
        cal_result = await self._calibration.calibrate(cal_config)

        # 3. Validate: did the calibration bring the metric back above threshold?
        if cal_result.after is not None:
            validation_passed = cal_result.after >= target_threshold
            status = "validated" if validation_passed else "calibrated"
        else:
            # Pending calibration — cannot validate yet
            validation_passed = None
            status = "calibrated"

        return FeedbackLoopResult(
            loop_id=loop_id,
            scenario=scenario,
            model=model,
            drift_detected=True,
            calibration=cal_result,
            validation_passed=validation_passed,
            status=status,
            started_at=started_at,
            completed_at=datetime.now(tz=timezone.utc),
        )

    @classmethod
    def from_store(
        cls,
        store: FileResultStore,
        configs_dir: Path | None = None,
    ) -> "FeedbackLoopEngine":
        """Convenience constructor that builds sub-engines from the same store."""
        drift = DriftEngine(store)
        calibration = CalibrationEngine(store, configs_dir=configs_dir)
        return cls(store, drift, calibration, configs_dir=configs_dir)
