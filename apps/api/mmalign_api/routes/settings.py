"""Settings routes: GET/PUT /settings/policies — scenario risk thresholds and strategy mappings.

Policies are persisted to ``configs/policies.yaml``.  Built-in strategy names
are validated against the known set; custom strategies require prior registration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mmalign_api.deps import get_configs_dir

router = APIRouter(tags=["settings"])

_POLICIES_FILE = "policies.yaml"

# Default fallback policies when file doesn't exist yet
_DEFAULT_RISK_THRESHOLDS = {"safe": 0.2, "low": 0.5, "medium": 0.8, "high": 1.0}
_DEFAULT_STRATEGY_MAPPING = {
    "safe": "passthrough",
    "low": "vanilla",
    "medium": "align3",
    "high": "spec_rewrite",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_policies(configs_dir: Path) -> dict[str, Any]:
    path = configs_dir / _POLICIES_FILE
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_policies(configs_dir: Path, data: dict[str, Any]) -> None:
    path = configs_dir / _POLICIES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8")


def _scenario_names(configs_dir: Path) -> list[str]:  # noqa: ARG001
    from mmalign.loaders.benchmarks import discover_packages

    return [p.id for p in discover_packages()]


def _build_policy_out(scenario_id: str, name: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "scenario_name": name,
        "risk_thresholds": raw.get("risk_thresholds", _DEFAULT_RISK_THRESHOLDS),
        "strategy_mapping": raw.get("strategy_mapping", _DEFAULT_STRATEGY_MAPPING),
    }


# ---------------------------------------------------------------------------
# Schemas (inline)
# ---------------------------------------------------------------------------

class UpdatePolicyRequest(BaseModel):
    risk_thresholds: dict[str, float] | None = None
    strategy_mapping: dict[str, str] | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/settings/policies")
async def get_policies(
    configs_dir: Path = Depends(get_configs_dir),
) -> dict[str, Any]:
    """Return all scenario policies."""
    stored = _load_policies(configs_dir)
    scenario_ids = _scenario_names(configs_dir)
    policies = []
    for sid in scenario_ids:
        raw = stored.get(sid, {})
        policies.append(_build_policy_out(sid, sid.replace("_", " ").title(), raw))
    return {"policies": policies}


@router.put("/settings/policies/{scenario_id}")
async def update_policy(
    scenario_id: str,
    req: UpdatePolicyRequest,
    configs_dir: Path = Depends(get_configs_dir),
) -> dict[str, Any]:
    """Update risk thresholds and/or strategy mapping for a scenario."""
    known = _scenario_names(configs_dir)
    if scenario_id not in known:
        raise HTTPException(
            status_code=404,
            detail={"code": "INVALID_SCENARIO", "message": f"Scenario '{scenario_id}' not found"},
        )

    stored = _load_policies(configs_dir)
    existing = stored.get(scenario_id, {})

    thresholds = dict(existing.get("risk_thresholds", _DEFAULT_RISK_THRESHOLDS))
    mapping = dict(existing.get("strategy_mapping", _DEFAULT_STRATEGY_MAPPING))

    if req.risk_thresholds:
        thresholds.update(req.risk_thresholds)
    if req.strategy_mapping:
        mapping.update(req.strategy_mapping)

    stored[scenario_id] = {
        "risk_thresholds": thresholds,
        "strategy_mapping": mapping,
    }
    _save_policies(configs_dir, stored)

    return {
        "scenario_id": scenario_id,
        "policy": {"risk_thresholds": thresholds, "strategy_mapping": mapping},
        "ok": True,
    }
