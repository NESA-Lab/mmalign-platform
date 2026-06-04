"""Policy loader — loads ControlPolicy from configs/policies.yaml.

The same YAML file is read/written by the WebUI Settings page and consumed
here by the SDK / API path.  Missing scenario sections fall back to the
``ControlPolicy`` defaults so this is always safe to call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mmalign.core.control import ControlPolicy, RiskLevel, StrategyMapping

_POLICIES_FILE = "policies.yaml"

_VALID_LEVELS: tuple[RiskLevel, ...] = ("safe", "low", "medium", "high")


def load_policy_from_yaml(
    scenario_id: str,
    configs_dir: Path | str | None,
) -> ControlPolicy:
    """Load the ControlPolicy for a scenario from policies.yaml.

    Behavior
    --------
    - If ``configs_dir`` is None, the file is missing, or the scenario has
      no entry, the default ``ControlPolicy`` is returned with the requested
      ``scenario_id``.
    - Unknown keys in the YAML are ignored; bad values fall back to defaults.
    - Float coercion is applied to thresholds; non-numeric values are dropped.
    """
    if configs_dir is None:
        return ControlPolicy(scenario_id=scenario_id)

    path = Path(configs_dir) / _POLICIES_FILE
    if not path.exists():
        return ControlPolicy(scenario_id=scenario_id)

    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return ControlPolicy(scenario_id=scenario_id)

    section: dict[str, Any] = raw.get(scenario_id) or {}
    if not isinstance(section, dict):
        return ControlPolicy(scenario_id=scenario_id)

    thresholds = _parse_thresholds(section.get("risk_thresholds"))
    mapping = _parse_mapping(section.get("strategy_mapping"))

    kwargs: dict[str, Any] = {"scenario_id": scenario_id}
    if thresholds:
        kwargs["risk_thresholds"] = thresholds
    if mapping is not None:
        kwargs["strategy_mapping"] = mapping
    if isinstance(section.get("max_attempts"), int) and section["max_attempts"] > 0:
        kwargs["max_attempts"] = section["max_attempts"]

    return ControlPolicy(**kwargs)


def _parse_thresholds(value: Any) -> dict[RiskLevel, float] | None:
    if not isinstance(value, dict):
        return None
    out: dict[RiskLevel, float] = {
        "safe": 0.2,
        "low": 0.5,
        "medium": 0.8,
        "high": 1.0,
    }
    found = False
    for level in _VALID_LEVELS:
        v = value.get(level)
        try:
            if v is not None:
                out[level] = float(v)
                found = True
        except (TypeError, ValueError):
            continue
    return out if found else None


def _parse_mapping(value: Any) -> StrategyMapping | None:
    if not isinstance(value, dict):
        return None
    kwargs: dict[str, str] = {}
    for level in _VALID_LEVELS:
        name = value.get(level)
        if isinstance(name, str) and name:
            kwargs[level] = name
    if not kwargs:
        return None
    return StrategyMapping(**kwargs)
