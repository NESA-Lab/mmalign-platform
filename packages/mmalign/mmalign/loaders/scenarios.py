"""Backward-compatible scenario API — delegates to self-contained benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mmalign.core.scenario import Sample, Scenario

from .benchmarks import (
    BenchmarkNotFoundError,
    BenchmarkOrigin,
    discover_packages,
    find_package,
    load_benchmark,
    list_benchmark_descriptors,
)

# Legacy alias
ScenarioNotFoundError = BenchmarkNotFoundError


def load_scenario(
    scenario_id: str,
    config_path: str | Path | None = None,  # noqa: ARG001 — deprecated
    *,
    origin: BenchmarkOrigin | None = None,
    model: str = "placeholder",
    use_adversarial: bool = True,
    sample_limit: int | None = None,
    workspace_start: Path | None = None,
) -> tuple[Scenario, list[Sample]]:
    ws = workspace_start
    if ws is None and config_path is not None:
        ws = Path(config_path).parent.parent
    return load_benchmark(
        scenario_id,
        origin=origin,
        model=model,
        use_adversarial=use_adversarial,
        sample_limit=sample_limit,
        workspace_start=ws,
    )


def list_scenarios(
    config_path: str | Path | None = None,  # noqa: ARG001 — deprecated
    *,
    origin: BenchmarkOrigin | None = None,
    workspace_start: Path | None = None,
) -> list[dict[str, Any]]:
    ws = workspace_start
    if ws is None and config_path is not None:
        ws = Path(config_path).parent.parent
    return list_benchmark_descriptors(origin=origin, workspace_start=ws)


__all__ = [
    "ScenarioNotFoundError",
    "load_scenario",
    "list_scenarios",
    "load_benchmark",
    "list_benchmark_descriptors",
    "discover_packages",
    "find_package",
]
