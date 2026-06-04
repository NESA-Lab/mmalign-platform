"""DriftEngine — metric time-series analysis and degradation detection.

Drift is detected when a metric's daily average deviates more than
DRIFT_THRESHOLD (10 %) from its rolling 7-day baseline. Alerts are
classified as "error" (>20 % change) or "warning" (>10 %).

This engine is a pure computation layer over the ResultStore; the API
route and CLI both delegate to it rather than re-implementing detection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from mmalign.core.store import FileResultStore


@dataclass
class DriftPoint:
    date: str
    safety_score: float
    behavioral_score: float
    unsafe_rate: float
    capability_rate: float
    drift_detected: bool = False


@dataclass
class DriftAlert:
    id: str
    level: str        # "warning" | "error"
    scenario: str
    model: str
    metric: str
    message: str
    detected_at: str
    baseline: float
    current: float
    change_pct: float


class DriftEngine:
    """Computes daily metric time-series and emits drift alerts."""

    DRIFT_THRESHOLD = 0.10  # 10 % relative change from 7-day rolling baseline
    ERROR_THRESHOLD = 0.20  # 20 % → error level

    def __init__(self, store: FileResultStore) -> None:
        self._store = store

    def detect(
        self,
        days: int = 30,
        scenario: str | None = None,
        model: str | None = None,
    ) -> tuple[list[DriftPoint], list[DriftAlert]]:
        """Return (time_series, alerts) for the given filters.

        Parameters
        ----------
        days:
            Number of days to look back (7–180).
        scenario:
            Filter by scenario ID.  None = aggregate all.
        model:
            Filter by model ID.  None = aggregate all.
        """
        runs, _ = self._store.list_runs(status="completed", limit=500)
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(days=days)

        # Bucket daily metric values
        by_day: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {
                "safety": [],
                "behavioral": [],
                "unsafe": [],
                "capability": [],
            }
        )
        for run in runs:
            if scenario and run.scenario != scenario:
                continue
            if model and run.model != model:
                continue
            if run.started_at < cutoff or not run.summary:
                continue
            day = run.started_at.strftime("%Y-%m-%d")
            s = run.summary
            by_day[day]["safety"].append(s.safety_score)
            by_day[day]["behavioral"].append(s.behavioral_score)
            by_day[day]["unsafe"].append(s.unsafe_rate)
            if s.capability_rate is not None:
                by_day[day]["capability"].append(s.capability_rate)

        # Build daily series
        series: list[DriftPoint] = []
        daily_safety: list[float] = []

        for i in range(days):
            day = (now - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
            d = by_day.get(day, {})
            safety_avg = _mean(d.get("safety", []))
            behavioral_avg = _mean(d.get("behavioral", []))
            unsafe_avg = _mean(d.get("unsafe", []))
            cap_avg = _mean(d.get("capability", []))

            daily_safety.append(safety_avg)
            # 7-day rolling baseline: window excludes today
            baseline_window = daily_safety[max(0, len(daily_safety) - 8) : -1]
            baseline = _mean(baseline_window) if baseline_window else safety_avg
            drift_detected = (
                baseline > 0
                and abs(safety_avg - baseline) / baseline > self.DRIFT_THRESHOLD
                and len(daily_safety) > 7
            )

            series.append(
                DriftPoint(
                    date=day,
                    safety_score=round(safety_avg, 4),
                    behavioral_score=round(behavioral_avg, 4),
                    unsafe_rate=round(unsafe_avg, 4),
                    capability_rate=round(cap_avg, 4),
                    drift_detected=drift_detected,
                )
            )

        # Generate alerts from drift points
        alerts: list[DriftAlert] = []
        for i, point in enumerate(series):
            if not point.drift_detected:
                continue
            prev_window = [
                s.safety_score
                for s in series[max(0, i - 7) : i]
                if s.safety_score > 0
            ]
            bl = _mean(prev_window) if prev_window else 0.0
            change_pct = (
                round((point.safety_score - bl) / bl * 100, 1) if bl else 0.0
            )
            level = "error" if abs(change_pct) > self.ERROR_THRESHOLD * 100 else "warning"
            alerts.append(
                DriftAlert(
                    id=f"drift-{point.date}",
                    level=level,
                    scenario=scenario or "all",
                    model=model or "all",
                    metric="safety_score",
                    message=(
                        f"Safety score {'dropped' if change_pct < 0 else 'spiked'} "
                        f"{abs(change_pct):.1f}% on {point.date} "
                        f"(baseline={bl:.3f}, current={point.safety_score:.3f})"
                    ),
                    detected_at=f"{point.date}T00:00:00Z",
                    baseline=round(bl, 4),
                    current=point.safety_score,
                    change_pct=change_pct,
                )
            )

        return series, alerts

    def latest_metric(
        self,
        metric: str,
        scenario: str | None = None,
        model: str | None = None,
    ) -> float:
        """Return the most recent daily average for the given metric."""
        runs, _ = self._store.list_runs(
            status="completed",
            scenario=scenario or None,
            model=model or None,
            limit=1,
        )
        if not runs or not runs[0].summary:
            return 0.0
        s = runs[0].summary
        return float(getattr(s, metric, 0.0) or 0.0)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0
