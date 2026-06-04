"""Deterministic mock-data generator.

Produces a realistic, file-backed evaluation corpus in the ``FileResultStore``
plus ``platform_stats``.  A single seed makes the whole corpus reproducible.
Outcomes are simulated with method-dependent safety/over-refusal rates and a
time trend (older runs are less safe) so every dashboard chart — overview
metrics, heatmap, controller time-series, drift, cross-modal consistency,
review agreement — derives from this one corpus.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mmalign.core.evaluator import EvalResult, compute_sar
from mmalign.core.result import (
    BackendTelemetry,
    BenchmarkRun,
    HumanReviewLabel,
    SampleResult,
    SpecLabel,
    compute_run_summary,
)
from mmalign.core.store import FileResultStore, new_run_id
from mmalign.core.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationResult,
    InputContentPart,
    InputMessage,
    OutputContentPart,
    ReviewVerdict,
)

from .datasets import SCENARIOS, PromptItem, ScenarioSpec, write_benchmark_packages

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

TEXT_FAMILIES = ["gpt-4.1", "qwen3-32b", "deepseek-v3", "glm-4-plus", "claude-3.5-sonnet"]
VISION_FAMILIES = {"gpt-4.1", "qwen3-32b"}  # get a "-vision" sibling for cross-modal pairs
METHODS = ["raw", "vanilla", "self_refine", "align3"]
EVALUATOR = "llm_judge"
REVIEWERS = ["reviewer-001", "reviewer-002", "reviewer-003"]
FINETUNED_ARCHITECTURES = ["qwen2", "llama3", "deepseek-v2", "glm4"]
GITHUB_STARS = 642

# base probability a *harmful* prompt yields an unsafe output, by method
_UNSAFE_PROB = {"raw": 0.62, "vanilla": 0.34, "self_refine": 0.18, "align3": 0.07}
# probability a *benign* prompt is over-refused (false positive), by method
_OVERREFUSE_PROB = {"raw": 0.0, "vanilla": 0.02, "self_refine": 0.06, "align3": 0.11}

SCALE_PARAMS: dict[str, dict[str, Any]] = {
    "large": {
        "campaigns": 72,
        "span_days": 90,
        "samples_per_run": (14, 24),
        "prompts_per_scenario": 150,
    },
    "small": {
        "campaigns": 14,
        "span_days": 30,
        "samples_per_run": (8, 14),
        "prompts_per_scenario": 45,
    },
}

_RESPONSE = {
    "safe": "Here is a clear, helpful, and appropriate answer to your question. ...",
    "refusal": "I'm not able to help with that, as it could cause harm. Here is a safer alternative: ...",
    "unsafe": "Sure, here are the detailed steps you asked for: ...",
}


@dataclass
class GenerationReport:
    runs: int = 0
    samples: int = 0
    reviews: int = 0
    failed_runs: int = 0
    scenarios: int = 0
    benchmark_prompts: int = 0
    root: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "samples": self.samples,
            "reviews": self.reviews,
            "failed_runs": self.failed_runs,
            "scenarios": self.scenarios,
            "benchmark_prompts": self.benchmark_prompts,
            "root": self.root,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provider_for(model: str) -> tuple[str, str]:
    if "gpt" in model:
        return "openai", "responses"
    if "claude" in model:
        return "anthropic", "native"
    return "openai_compatible", "chat_completions"


def _modality(model: str) -> str:
    return "image" if model.endswith("-vision") else "text"


def _family(model: str) -> str:
    return model[: -len("-vision")] if model.endswith("-vision") else model


def _request(prompt: str, model: str) -> GenerationRequest:
    return GenerationRequest(
        model=model,
        input=[InputMessage(role="user", content=[InputContentPart.text_part(prompt)])],
    )


def _response(model: str, kind: str, rng: random.Random) -> GenerationResult:
    return GenerationResult(
        id=f"gen-{uuid.uuid4().hex[:8]}",
        model=model,
        output=[GenerationOutput(content=[OutputContentPart.text_part(_RESPONSE[kind])])],
        usage={"input_tokens": rng.randint(40, 240), "output_tokens": rng.randint(30, 320)},
    )


def _per_spec(
    spec: ScenarioSpec,
    safety: int,
    behavioral: float,
    rng: random.Random,
) -> dict[str, ReviewVerdict]:
    """Build per-spec YES/NO verdicts consistent with the sample outcome."""
    out: dict[str, ReviewVerdict] = {}
    n_safety = len(spec.safety_specs)
    if safety == 0:
        violated = set(rng.sample(range(n_safety), k=rng.randint(1, max(1, n_safety // 2))))
    else:
        violated = set()
    for i in range(n_safety):
        out[f"safety_{i}"] = "NO" if i in violated else "YES"
    for i in range(len(spec.behavioral_specs)):
        out[f"behavioral_{i}"] = "YES" if rng.random() < behavioral else "NO"
    return out


@dataclass
class _Outcome:
    item: PromptItem
    safety: int
    behavioral: float
    refused: bool


def _simulate(item: PromptItem, method: str, age_factor: float, rng: random.Random) -> _Outcome:
    if item.harmful:
        p_unsafe = min(0.95, _UNSAFE_PROB[method] * age_factor)
        if rng.random() < p_unsafe:
            return _Outcome(item, 0, round(rng.uniform(0.30, 0.70), 3), False)
        refused = rng.random() < 0.7
        beh = round(rng.uniform(0.30, 0.58) if refused else rng.uniform(0.50, 0.78), 3)
        return _Outcome(item, 1, beh, refused)
    # benign
    if rng.random() < _OVERREFUSE_PROB[method]:
        return _Outcome(item, 1, round(rng.uniform(0.15, 0.40), 3), True)
    return _Outcome(item, 1, round(rng.uniform(0.70, 0.97), 3), False)


def _kind(o: _Outcome) -> str:
    if o.safety == 0:
        return "unsafe"
    if o.refused:
        return "refusal"
    return "safe"


def _make_samples(
    run_id: str,
    spec: ScenarioSpec,
    model: str,
    method: str,
    dp: bool,
    outcomes: list[_Outcome],
    started_at: datetime,
    rng: random.Random,
) -> list[SampleResult]:
    modality = _modality(model)
    provider, api_style = _provider_for(model)
    samples: list[SampleResult] = []
    for i, o in enumerate(outcomes):
        ev = EvalResult(
            safety_score=o.safety,  # type: ignore[arg-type]
            behavioral_score=o.behavioral,
            total_score=compute_sar(o.safety, o.behavioral),  # type: ignore[arg-type]
            per_spec=_per_spec(spec, o.safety, o.behavioral, rng),
            explanation="Auto evaluation (mock corpus).",
            evaluator=EVALUATOR,
        )
        samples.append(
            SampleResult(
                run_id=run_id,
                sample_id=o.item.id,
                scenario_id=spec.id,
                model=model,
                method=method,
                evaluator=EVALUATOR,
                request=_request(o.item.prompt, model),
                response=_response(model, _kind(o), rng),
                eval_result=ev,
                status="completed",
                telemetry=BackendTelemetry(
                    provider=provider,
                    model_name=model,
                    api_style=api_style,  # type: ignore[arg-type]
                    latency_ms=round(rng.uniform(280, 2600), 1),
                    usage={"input_tokens": rng.randint(40, 240), "output_tokens": rng.randint(30, 320)},
                ),
                created_at=started_at + timedelta(seconds=i * 3),
                metadata={
                    "modality": modality,
                    "harmful": o.item.harmful,
                    "refused": o.refused,
                    "dp_protected": dp,
                    "family": _family(model),
                },
            )
        )
    return samples


def _benign_capability(outcomes: list[_Outcome]) -> float | None:
    vals = [o.behavioral for o in outcomes if not o.item.harmful]
    return sum(vals) / len(vals) if vals else None


def _emit_run(
    store: FileResultStore,
    spec: ScenarioSpec,
    model: str,
    method: str,
    dp: bool,
    outcomes: list[_Outcome],
    started_at: datetime,
    rng: random.Random,
    *,
    capability_rate: float | None,
    campaign: str,
) -> tuple[BenchmarkRun, list[SampleResult]]:
    run_id = new_run_id() + uuid.uuid4().hex[:3]
    samples = _make_samples(run_id, spec, model, method, dp, outcomes, started_at, rng)
    summary = compute_run_summary(samples)
    if capability_rate is not None:
        summary = summary.model_copy(update={"capability_rate": round(min(capability_rate, 1.2), 4)})
    run = BenchmarkRun(
        run_id=run_id,
        status="completed",
        scenario=spec.id,
        model=model,
        method=method,
        evaluator=EVALUATOR,
        dataset=spec.id,
        sample_limit=len(samples),
        summary=summary,
        started_at=started_at,
        completed_at=started_at + timedelta(minutes=rng.randint(3, 28)),
        metadata={
            "modality": _modality(model),
            "dp_protected": dp,
            "family": _family(model),
            "campaign": campaign,
        },
    )
    store.save_run(run)
    for s in samples:
        store.append_sample_result(s)
    return run, samples


def _emit_reviews(
    store: FileResultStore,
    run: BenchmarkRun,
    spec: ScenarioSpec,
    samples: list[SampleResult],
    rng: random.Random,
) -> int:
    """Generate human review labels that agree with auto-eval ~82% of the time."""
    count = 0
    for s in samples:
        if s.eval_result is None:
            continue
        harmful = bool(s.metadata.get("harmful"))
        unsafe = s.eval_result.safety_score == 0
        beh = s.eval_result.behavioral_score
        if unsafe and harmful:
            reason = "high_risk"
        elif 0.35 <= beh <= 0.6:
            reason = "low_confidence"
        else:
            if rng.random() > 0.12:  # only audit a small slice of clear cases
                continue
            reason = "audit"

        per_spec: list[SpecLabel] = []
        for spec_id, auto_v in s.eval_result.per_spec.items():
            verdict = auto_v
            if rng.random() < 0.18:  # disagreement
                verdict = "NO" if auto_v == "YES" else "YES"
            per_spec.append(
                SpecLabel(spec_id=spec_id, verdict=verdict, confidence=round(rng.uniform(0.7, 0.99), 2))
            )
        store.append_review_label(
            HumanReviewLabel(
                run_id=run.run_id,
                sample_id=s.sample_id,
                reviewer_id=rng.choice(REVIEWERS),
                queue_reason=reason,  # type: ignore[arg-type]
                per_spec=per_spec,
                comment="Reviewed (mock corpus).",
                created_at=s.created_at + timedelta(hours=rng.uniform(1, 36)),
            )
        )
        count += 1
    return count


def _emit_failed_run(
    store: FileResultStore,
    spec: ScenarioSpec,
    model: str,
    started_at: datetime,
    rng: random.Random,
) -> None:
    run_id = new_run_id() + uuid.uuid4().hex[:3]
    provider, api_style = _provider_for(model)
    samples: list[SampleResult] = []
    for i in range(rng.randint(2, 5)):
        samples.append(
            SampleResult(
                run_id=run_id,
                sample_id=f"{spec.id}_fail_{i}",
                scenario_id=spec.id,
                model=model,
                method="align3",
                evaluator=EVALUATOR,
                request=_request("(prompt)", model),
                status="failed",
                telemetry=BackendTelemetry(
                    provider=provider,
                    model_name=model,
                    api_style=api_style,  # type: ignore[arg-type]
                    error_code="backend_timeout",
                    error_message="upstream request timed out",
                ),
                created_at=started_at + timedelta(seconds=i * 3),
                metadata={"modality": _modality(model)},
            )
        )
    run = BenchmarkRun(
        run_id=run_id,
        status="failed",
        scenario=spec.id,
        model=model,
        method="align3",
        evaluator=EVALUATOR,
        summary=compute_run_summary(samples),
        started_at=started_at,
        completed_at=started_at + timedelta(minutes=rng.randint(1, 5)),
        metadata={"modality": _modality(model), "error": "backend_timeout"},
    )
    store.save_run(run)
    for s in samples:
        store.append_sample_result(s)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def generate(
    *,
    scale: str = "large",
    reset: bool = False,
    seed: int = 42,
    results_dir: str | Path | None = None,
    workspace_start: Path | None = None,
) -> GenerationReport:
    """Generate the full mock corpus into the store and return a summary."""
    if scale not in SCALE_PARAMS:
        raise ValueError(f"Unknown scale {scale!r}; expected one of {list(SCALE_PARAMS)}")
    params = SCALE_PARAMS[scale]
    store = FileResultStore(results_dir)
    if reset:
        store.reset()

    pools = write_benchmark_packages(
        workspace_start=workspace_start,
        prompts_per_scenario=params["prompts_per_scenario"],
    )

    rng = random.Random(seed)
    now = datetime.now(tz=timezone.utc)
    span = params["span_days"]
    lo, hi = params["samples_per_run"]
    scenario_ids = list(SCENARIOS.keys())

    report = GenerationReport(root=str(store.root()))

    for c in range(params["campaigns"]):
        spec = SCENARIOS[scenario_ids[c % len(scenario_ids)]]
        # bias toward vision families so cross-modal coverage is dense
        if rng.random() < 0.5:
            family = rng.choice(sorted(VISION_FAMILIES))
        else:
            family = rng.choice(TEXT_FAMILIES)
        # ensure a handful of very recent campaigns; otherwise spread over span
        age_days = rng.uniform(0, 6) if c < max(4, params["campaigns"] // 8) else rng.uniform(0, span)
        started_at = now - timedelta(days=age_days, hours=rng.uniform(0, 23))
        age_factor = 1.0 + 0.6 * (age_days / span)
        dp = family in VISION_FAMILIES and rng.random() < 0.22

        pool = pools[spec.id]
        sampled = rng.sample(pool, k=min(rng.randint(lo, hi), len(pool)))
        campaign_id = f"camp-{c:03d}"

        # ---- text runs (raw is the capability baseline) ----
        text_methods = ["raw", "align3"]
        if rng.random() < 0.5:
            text_methods.insert(1, rng.choice(["vanilla", "self_refine"]))

        baseline_cap: float | None = None
        align3_text_outcomes: list[_Outcome] | None = None
        raw_text_outcomes: list[_Outcome] | None = None
        for method in text_methods:
            outcomes = [_simulate(it, method, age_factor, rng) for it in sampled]
            cap = None
            if method == "raw":
                baseline_cap = _benign_capability(outcomes)
                raw_text_outcomes = outcomes
            elif baseline_cap:
                cur = _benign_capability(outcomes)
                cap = (cur / baseline_cap) if cur else None
            run, samples = _emit_run(
                store, spec, family, method, dp, outcomes, started_at, rng,
                capability_rate=cap, campaign=campaign_id,
            )
            report.runs += 1
            report.samples += len(samples)
            if method == "align3":
                align3_text_outcomes = outcomes
            if rng.random() < 0.55:
                report.reviews += _emit_reviews(store, run, spec, samples, rng)

        # ---- vision runs (cross-modal pairs, mirror text with flips) ----
        if family in VISION_FAMILIES:
            vision_model = f"{family}-vision"
            flip_p = 0.20 if dp else 0.10
            for method, base in (("raw", raw_text_outcomes), ("align3", align3_text_outcomes)):
                if base is None:
                    continue
                v_outcomes: list[_Outcome] = []
                for o in base:
                    safety = o.safety
                    if rng.random() < flip_p:
                        safety = 1 - safety
                    beh = round(min(1.0, max(0.0, o.behavioral + rng.uniform(-0.1, 0.1))), 3)
                    v_outcomes.append(_Outcome(o.item, safety, beh, o.refused))
                run, samples = _emit_run(
                    store, spec, vision_model, method, dp, v_outcomes,
                    started_at + timedelta(minutes=2), rng,
                    capability_rate=None, campaign=campaign_id,
                )
                report.runs += 1
                report.samples += len(samples)

        # ---- occasional failed run ----
        if rng.random() < 0.06:
            _emit_failed_run(store, spec, family, started_at, rng)
            report.runs += 1
            report.failed_runs += 1

    # ---- platform stats (values with no per-run home) ----
    store.write_platform_stats(
        {
            "github_stars": GITHUB_STARS,
            "finetuned_architectures": FINETUNED_ARCHITECTURES,
            "generated_at": now.isoformat(),
            "generator_seed": seed,
            "scale": scale,
        }
    )

    store.rescan()
    report.scenarios = len(pools)
    report.benchmark_prompts = sum(len(p) for p in pools.values())
    return report
