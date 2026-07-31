"""Tests for the global registries (Fix 4, US-6)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from mmalign.core.registry import BuiltinOverrideError
from mmalign.core.registries import (
    control_strategy_registry,
    evaluator_registry,
    inference_align_registry,
    training_align_registry,
)


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------

class TestBuiltinRegistration:
    def test_inference_builtins_present(self) -> None:
        assert "raw" in inference_align_registry
        assert "vanilla" in inference_align_registry
        assert "align3" in inference_align_registry
        assert "self_refine" in inference_align_registry
        assert "vanilla_specbench" in inference_align_registry
        assert "align3_specbench" in inference_align_registry
        assert "zero_think" in inference_align_registry
        assert "more_think" in inference_align_registry
        assert "best_of_n" in inference_align_registry
        assert "tpo" in inference_align_registry
        assert "ipr_prompt_refine" in inference_align_registry
        assert "vlm_safety_guard" in inference_align_registry
        assert "overrefusal_check" in inference_align_registry
        assert "user_conditioned_safety_critic" in inference_align_registry
        assert "critic_revise_agent" in inference_align_registry
        assert "risk_adaptive_agent" in inference_align_registry
        assert "multi_agent_consensus" in inference_align_registry
        assert set(inference_align_registry.list_builtins()) >= {
            "raw", "vanilla", "align3", "self_refine",
            "vanilla_specbench", "align3_specbench", "zero_think",
            "more_think", "best_of_n", "tpo", "ipr_prompt_refine",
            "vlm_safety_guard", "overrefusal_check",
            "user_conditioned_safety_critic", "critic_revise_agent",
            "risk_adaptive_agent", "multi_agent_consensus",
        }

    def test_evaluator_builtins_present(self) -> None:
        assert "mock_judge" in evaluator_registry
        assert "rule_based" in evaluator_registry

    def test_strategy_builtins_present(self) -> None:
        names = set(control_strategy_registry.list_builtins())
        assert {"passthrough", "vanilla", "align3", "spec_rewrite", "refusal"}.issubset(names)

    def test_training_builtins_present(self) -> None:
        assert "dpo" in training_align_registry
        assert "sft" in training_align_registry

    def test_builtin_cannot_be_overwritten(self) -> None:
        with pytest.raises(BuiltinOverrideError):
            inference_align_registry.register("align3", lambda cfg: None)

    def test_factory_produces_real_instances(self) -> None:
        factory = inference_align_registry.get("vanilla")
        instance = factory({"include_safety_specs": True})
        from mmalign.methods.vanilla import VanillaMethod
        assert isinstance(instance, VanillaMethod)


# ---------------------------------------------------------------------------
# load_method consults the registry
# ---------------------------------------------------------------------------

class TestLoadMethodConsultsRegistry:
    def test_custom_plugin_method_resolves_via_registry(self, tmp_path: Path) -> None:
        """A method registered into inference_align_registry should be
        resolvable by ``load_method`` even if its ``type`` is not in the
        legacy if/elif chain."""
        from mmalign.loaders.methods import load_method

        # Define a stub method that satisfies the InferenceAlignMethod-ish shape.
        class _StubMethod:
            name = "custom_stub"
            supported_modalities = {"text"}

            def validate(self, scenario, backend) -> None:  # noqa: ARG002
                return None

            async def align(self, request, backend, scenario, context=None):  # noqa: ARG002
                from mmalign.core.method import AlignedResponse
                return AlignedResponse(output=[], method=self.name)

        called: dict[str, int] = {"count": 0}

        def factory(cfg):
            called["count"] += 1
            return _StubMethod()

        try:
            inference_align_registry.register("my_org.custom_stub", factory)

            cfg_path = tmp_path / "methods.yaml"
            cfg_path.write_text(
                yaml.dump({
                    "methods": {
                        "my_method": {"type": "my_org.custom_stub"}
                    }
                }),
                encoding="utf-8",
            )
            instance = load_method("my_method", cfg_path)
            assert isinstance(instance, _StubMethod)
            assert called["count"] == 1
        finally:
            inference_align_registry.unregister("my_org.custom_stub")

    def test_builtin_method_still_works(self, tmp_path: Path) -> None:
        """Built-in 'align3' should resolve via the registry (since we registered it)."""
        from mmalign.loaders.methods import load_method
        from mmalign.methods.align3 import Align3Method

        cfg_path = tmp_path / "methods.yaml"
        cfg_path.write_text(
            yaml.dump({"methods": {"align3": {"type": "align3", "max_stage_tokens": 256}}}),
            encoding="utf-8",
        )
        instance = load_method("align3", cfg_path)
        assert isinstance(instance, Align3Method)

    def test_specbench_ttd_builtin_method_loads(self, tmp_path: Path) -> None:
        from mmalign.loaders.methods import load_method
        from mmalign.methods.specbench_ttd import Align3SpecBenchMethod, BestOfNMethod

        cfg_path = tmp_path / "methods.yaml"
        cfg_path.write_text(
            yaml.dump({
                "methods": {
                    "best_of_n": {"type": "best_of_n", "n": 2},
                    "align3_specbench": {"type": "align3_specbench"},
                }
            }),
            encoding="utf-8",
        )
        assert isinstance(load_method("best_of_n", cfg_path), BestOfNMethod)
        assert isinstance(load_method("align3_specbench", cfg_path), Align3SpecBenchMethod)

    def test_unknown_type_raises_with_helpful_message(self, tmp_path: Path) -> None:
        from mmalign.loaders.methods import UnknownMethodError, load_method

        cfg_path = tmp_path / "methods.yaml"
        cfg_path.write_text(
            yaml.dump({"methods": {"bogus": {"type": "no_such_type"}}}),
            encoding="utf-8",
        )
        with pytest.raises(UnknownMethodError) as exc:
            load_method("bogus", cfg_path)
        assert "registries.inference_align_registry" in str(exc.value)


# ---------------------------------------------------------------------------
# Plugin entry-point discovery (idempotent + safe under errors)
# ---------------------------------------------------------------------------

class TestPluginDiscovery:
    def test_load_plugin_methods_is_idempotent(self) -> None:
        from mmalign.core import registries
        # Reset for the test, call twice; second call must be a no-op.
        registries._plugins_loaded = False
        registries.load_plugin_methods()
        registries.load_plugin_methods()  # must not raise
