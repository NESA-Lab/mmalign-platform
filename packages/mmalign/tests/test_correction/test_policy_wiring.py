"""Tests for policy loading + ComplianceController policy wiring (Fix 1, US-14/15)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from mmalign.core.control import ControlPolicy, StrategyMapping
from mmalign.core.scenario import Scenario, Specification
from mmalign.loaders.policies import load_policy_from_yaml


def _make_scenario(scenario_id: str = "education") -> Scenario:
    return Scenario(
        id=scenario_id,
        name=scenario_id,
        description="Test scenario",
        modality="text",
        language="en",
        safety_specs=[
            Specification(spec_id="s0", text="Do not cause harm.", category="safety"),
        ],
        behavioral_specs=[
            Specification(spec_id="b0", text="Be helpful.", category="behavioral"),
        ],
    )


# ---------------------------------------------------------------------------
# load_policy_from_yaml
# ---------------------------------------------------------------------------

class TestLoadPolicyFromYaml:
    def test_default_when_configs_dir_missing(self) -> None:
        policy = load_policy_from_yaml("education", configs_dir=None)
        assert policy.scenario_id == "education"
        assert policy.risk_thresholds == {"safe": 0.2, "low": 0.5, "medium": 0.8, "high": 1.0}
        assert policy.strategy_mapping.medium == "align3"

    def test_default_when_file_missing(self, tmp_path: Path) -> None:
        policy = load_policy_from_yaml("education", configs_dir=tmp_path)
        assert policy.scenario_id == "education"
        assert policy.strategy_mapping.medium == "align3"

    def test_loads_thresholds_and_mapping(self, tmp_path: Path) -> None:
        (tmp_path / "policies.yaml").write_text(
            yaml.dump({
                "education": {
                    "risk_thresholds": {"safe": 0.1, "low": 0.3, "medium": 0.6, "high": 1.0},
                    "strategy_mapping": {
                        "safe": "passthrough",
                        "low": "vanilla",
                        "medium": "align3",
                        "high": "refusal",
                    },
                }
            }),
            encoding="utf-8",
        )
        policy = load_policy_from_yaml("education", configs_dir=tmp_path)
        assert policy.risk_thresholds == {"safe": 0.1, "low": 0.3, "medium": 0.6, "high": 1.0}
        assert policy.strategy_mapping.high == "refusal"

    def test_unknown_scenario_returns_default(self, tmp_path: Path) -> None:
        (tmp_path / "policies.yaml").write_text(
            yaml.dump({"education": {"risk_thresholds": {"safe": 0.1}}}),
            encoding="utf-8",
        )
        policy = load_policy_from_yaml("unknown", configs_dir=tmp_path)
        assert policy.scenario_id == "unknown"
        # Falls back to the built-in defaults.
        assert policy.risk_thresholds == {"safe": 0.2, "low": 0.5, "medium": 0.8, "high": 1.0}

    def test_malformed_yaml_returns_default(self, tmp_path: Path) -> None:
        (tmp_path / "policies.yaml").write_text("::: not valid yaml :::", encoding="utf-8")
        policy = load_policy_from_yaml("education", configs_dir=tmp_path)
        assert policy.scenario_id == "education"
        assert policy.strategy_mapping.medium == "align3"

    def test_partial_thresholds_keep_defaults_for_unspecified(self, tmp_path: Path) -> None:
        (tmp_path / "policies.yaml").write_text(
            yaml.dump({"education": {"risk_thresholds": {"safe": 0.05}}}),
            encoding="utf-8",
        )
        policy = load_policy_from_yaml("education", configs_dir=tmp_path)
        # Custom: safe=0.05; defaults preserved for low/medium/high.
        assert policy.risk_thresholds["safe"] == 0.05
        assert policy.risk_thresholds["low"] == 0.5
        assert policy.risk_thresholds["medium"] == 0.8


# ---------------------------------------------------------------------------
# ComplianceController policy wiring
# ---------------------------------------------------------------------------

class TestControllerPolicyWiring:
    @pytest.mark.asyncio
    async def test_policy_threshold_changes_strategy(self) -> None:
        """A custom low threshold should re-bucket a low-score request to 'low'.

        With the *default* thresholds, score 0.3 would land in 'low' anyway.
        With a tightened policy of safe<=0.1, score 0.3 should still be 'low'.
        With a loose policy of safe<=0.5, score 0.3 should be 'safe'.
        We assert the loose case: passthrough strategy fires when re-bucketed to safe.
        """
        from mmalign.backends.mock import MockBackend
        from mmalign.core.types import GenerationRequest
        from mmalign.engines.correction.controller import ComplianceController

        scenario = _make_scenario()
        backend = MockBackend()

        # Loose policy: any score below 0.5 is 'safe' → passthrough.
        loose = ControlPolicy(
            scenario_id="education",
            risk_thresholds={"safe": 0.5, "low": 0.7, "medium": 0.9, "high": 1.0},
            strategy_mapping=StrategyMapping(
                safe="passthrough",
                low="vanilla",
                medium="align3",
                high="refusal",
            ),
        )
        controller = ComplianceController(backend=backend, scenario=scenario, policy=loose)
        # Use a borderline LOW prompt (avoid_tax score 0.35).
        result = await controller.generate(GenerationRequest(model="mock", input="如何避税"))
        # Re-bucketed to 'safe' under the loose policy → strategy must be passthrough.
        assert result.risk.level == "safe"
        assert result.strategy == "passthrough"

    @pytest.mark.asyncio
    async def test_strict_policy_promotes_low_score_to_high(self) -> None:
        """A strict policy should promote a small score into high-risk bucket."""
        from mmalign.backends.mock import MockBackend
        from mmalign.core.types import GenerationRequest
        from mmalign.engines.correction.controller import ComplianceController

        scenario = _make_scenario()
        backend = MockBackend()

        strict = ControlPolicy(
            scenario_id="education",
            risk_thresholds={"safe": 0.05, "low": 0.1, "medium": 0.2, "high": 1.0},
            strategy_mapping=StrategyMapping(
                safe="passthrough",
                low="vanilla",
                medium="align3",
                high="refusal",
            ),
        )
        controller = ComplianceController(backend=backend, scenario=scenario, policy=strict)
        # avoid_tax has score 0.35 → with strict thresholds this is 'high'.
        result = await controller.generate(GenerationRequest(model="mock", input="如何避税"))
        assert result.risk.level == "high"
        assert result.strategy == "refusal"

    @pytest.mark.asyncio
    async def test_default_policy_preserves_detector_level(self) -> None:
        """Without an override policy the detector's bucketing is honored."""
        from mmalign.backends.mock import MockBackend
        from mmalign.core.types import GenerationRequest
        from mmalign.engines.correction.controller import ComplianceController

        scenario = _make_scenario()
        backend = MockBackend()
        # No policy → uses default thresholds 0.2/0.5/0.8/1.0
        controller = ComplianceController(backend=backend, scenario=scenario)
        # avoid_tax score 0.35 → low under defaults.
        result = await controller.generate(GenerationRequest(model="mock", input="如何避税"))
        assert result.risk.level == "low"


# ---------------------------------------------------------------------------
# load_controller_from_config end-to-end
# ---------------------------------------------------------------------------

class TestLoadControllerFromConfig:
    def test_load_controller_uses_policy(self, tmp_path: Path) -> None:
        configs = tmp_path / "configs"
        configs.mkdir()
        (configs / "models.yaml").write_text(
            yaml.dump({"models": {"mock": {"provider": "mock"}}}),
            encoding="utf-8",
        )
        (configs / "policies.yaml").write_text(
            yaml.dump({
                "test_domain": {
                    "risk_thresholds": {"safe": 0.05, "low": 0.1, "medium": 0.2, "high": 1.0},
                    "strategy_mapping": {
                        "safe": "passthrough",
                        "low": "vanilla",
                        "medium": "align3",
                        "high": "refusal",
                    },
                }
            }),
            encoding="utf-8",
        )

        # Create a minimal scenario package the loader can discover.
        import json
        pkg = tmp_path / "data" / "benchmarks" / "builtin" / "test_domain"
        pkg.mkdir(parents=True)
        (pkg / "metadata.yaml").write_text(yaml.dump({
            "schema_version": 1,
            "id": "test_domain",
            "name": "Test Domain",
            "description": "",
            "origin": "builtin",
            "source": "native",
            "language": "en",
            "modality": "text",
            "enabled": True,
        }), encoding="utf-8")
        (pkg / "specifications.json").write_text(
            json.dumps({"safety_specifications": ["s"], "behavioral_specifications": ["b"]}),
            encoding="utf-8",
        )
        (pkg / "prompts.json").write_text("[]", encoding="utf-8")

        # Point benchmark discovery at this tmp path via env var.
        import os
        prev = os.environ.get("MMALIGN_DATA_DIR")
        os.environ["MMALIGN_DATA_DIR"] = str(tmp_path / "data")
        try:
            from mmalign.engines.correction.controller import load_controller_from_config
            controller = load_controller_from_config("mock", "test_domain", configs)
            assert controller._policy.risk_thresholds["safe"] == 0.05
            assert controller._policy.strategy_mapping.high == "refusal"
        finally:
            if prev is None:
                os.environ.pop("MMALIGN_DATA_DIR", None)
            else:
                os.environ["MMALIGN_DATA_DIR"] = prev
