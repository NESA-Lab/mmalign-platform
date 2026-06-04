"""AgreementCalculator — human-machine agreement and Cohen's kappa (metrics 1.6 / 1.7)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from mmalign.core.result import HumanReviewLabel, SpecLabel
from mmalign.core.store import FileResultStore
from mmalign.core.types import ReviewVerdict


@dataclass
class AgreementStatsResult:
    total_reviewed: int
    agreement_rate: float
    kappa: float
    hitl_accuracy: float
    by_scenario: list[dict[str, Any]]
    by_reviewer: list[dict[str, Any]]


def _human_safety_from_per_spec(per_spec: list[SpecLabel]) -> int:
    """Binary safety: 0 if any safety spec is NO."""
    for sl in per_spec:
        if sl.spec_id.startswith("safety") and sl.verdict == "NO":
            return 0
    return 1


class AgreementCalculator:
    """Compute agreement between human labels and auto-eval from ResultStore."""

    def __init__(self, store: FileResultStore) -> None:
        self._store = store

    def _auto_per_spec_for_sample(
        self,
        run_id: str,
        sample_id: str,
    ) -> dict[str, ReviewVerdict]:
        for sr in self._store.iter_sample_results(run_id):
            if sr.sample_id == sample_id and sr.eval_result:
                return sr.eval_result.per_spec or {}
        return {}

    def _auto_safety_score(self, run_id: str, sample_id: str) -> int | None:
        for sr in self._store.iter_sample_results(run_id):
            if sr.sample_id == sample_id and sr.eval_result:
                return sr.eval_result.safety_score
        return None

    def agreement_for_label(self, label: HumanReviewLabel) -> tuple[int, int, float]:
        """Return (agree_count, total_specs, per-sample rate) for one label."""
        auto_per_spec = self._auto_per_spec_for_sample(label.run_id, label.sample_id)
        if not auto_per_spec or not label.per_spec:
            return 0, 0, 0.0

        agree = 0
        total = 0
        for sl in label.per_spec:
            auto_v = auto_per_spec.get(sl.spec_id)
            if auto_v is not None:
                total += 1
                if sl.verdict == auto_v:
                    agree += 1
        rate = agree / total if total > 0 else 0.0
        return agree, total, rate

    def hitl_match_for_label(self, label: HumanReviewLabel) -> bool | None:
        """Whether human-derived safety matches auto safety (proxy for metric 1.7)."""
        auto_safety = self._auto_safety_score(label.run_id, label.sample_id)
        if auto_safety is None or not label.per_spec:
            return None
        human_safety = _human_safety_from_per_spec(label.per_spec)
        return human_safety == auto_safety

    def compute_stats(self) -> AgreementStatsResult:
        """Aggregate all review labels across completed runs."""
        runs, _ = self._store.list_runs(status="completed", limit=500)

        total_agree = 0
        total_specs = 0
        total_reviewed = 0
        hitl_correct = 0
        hitl_total = 0

        by_scenario: dict[str, list[float]] = defaultdict(list)
        by_reviewer: dict[str, list[float]] = defaultdict(list)

        # For kappa: count concordant YES/NO pairs
        concordant = 0
        discordant = 0
        auto_yes = 0
        human_yes = 0

        for run_summary in runs:
            try:
                for label in self._store.iter_review_labels(run_summary.run_id):
                    if not label.per_spec:
                        continue

                    agree, spec_total, rate = self.agreement_for_label(label)
                    if spec_total == 0:
                        continue

                    total_reviewed += 1
                    total_agree += agree
                    total_specs += spec_total
                    by_scenario[run_summary.scenario].append(rate)
                    by_reviewer[label.reviewer_id].append(rate)

                    auto_per_spec = self._auto_per_spec_for_sample(
                        label.run_id, label.sample_id
                    )
                    for sl in label.per_spec:
                        av = auto_per_spec.get(sl.spec_id)
                        if av not in ("YES", "NO") or sl.verdict not in ("YES", "NO"):
                            continue
                        if av == "YES":
                            auto_yes += 1
                        if sl.verdict == "YES":
                            human_yes += 1
                        if av == sl.verdict:
                            concordant += 1
                        else:
                            discordant += 1

                    hitl = self.hitl_match_for_label(label)
                    if hitl is not None:
                        hitl_total += 1
                        if hitl:
                            hitl_correct += 1
            except Exception:
                continue

        n_pairs = concordant + discordant
        overall_rate = total_agree / total_specs if total_specs > 0 else 0.0
        hitl_accuracy = hitl_correct / hitl_total if hitl_total > 0 else 0.0

        # Cohen's kappa (binary YES/NO)
        if n_pairs > 0:
            p_o = concordant / n_pairs
            p_auto = auto_yes / n_pairs
            p_human = human_yes / n_pairs
            p_e = p_auto * p_human + (1 - p_auto) * (1 - p_human)
            kappa = (p_o - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-9 else 0.0
            kappa = max(-1.0, min(1.0, kappa))
        else:
            kappa = 0.0

        return AgreementStatsResult(
            total_reviewed=total_reviewed,
            agreement_rate=round(overall_rate, 4),
            kappa=round(kappa, 4),
            hitl_accuracy=round(hitl_accuracy, 4),
            by_scenario=[
                {"scenario": s, "rate": round(sum(v) / len(v), 4)}
                for s, v in by_scenario.items()
            ],
            by_reviewer=[
                {"reviewer": r, "rate": round(sum(v) / len(v), 4), "count": len(v)}
                for r, v in by_reviewer.items()
            ],
        )

    def daily_metric_values(
        self,
        metric_id: str,
        days: int,
    ) -> list[tuple[str, float]]:
        """Daily agreement (1.6) or HITL accuracy (1.7) from review label timestamps."""
        from datetime import datetime, timedelta, timezone

        runs, _ = self._store.list_runs(status="completed", limit=500)
        now = datetime.now(tz=timezone.utc)
        cutoff = now - timedelta(days=days)

        by_day: dict[str, list[float]] = defaultdict(list)

        for run_summary in runs:
            try:
                for label in self._store.iter_review_labels(run_summary.run_id):
                    if label.created_at < cutoff or not label.per_spec:
                        continue
                    day = label.created_at.strftime("%Y-%m-%d")
                    if metric_id == "1.6":
                        _, _, rate = self.agreement_for_label(label)
                        by_day[day].append(rate * 100)
                    elif metric_id == "1.7":
                        hitl = self.hitl_match_for_label(label)
                        if hitl is not None:
                            by_day[day].append(100.0 if hitl else 0.0)
            except Exception:
                continue

        history: list[tuple[str, float]] = []
        for i in range(days):
            day = (now - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
            vals = by_day.get(day, [])
            val = round(sum(vals) / len(vals), 2) if vals else 0.0
            history.append((f"{day}T00:00:00Z", val))
        return history
