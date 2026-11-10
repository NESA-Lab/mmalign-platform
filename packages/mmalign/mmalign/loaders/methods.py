"""Method loader — instantiates InferenceAlignMethod from configs/methods.yaml.

Resolution order
----------------
1. Consult ``inference_align_registry`` (built-ins + plugins).  If a
   factory is registered under ``cfg['type']`` (or the method id when no
   type is specified) it is invoked with the config dict.
2. Fall back to the legacy ``_build_method`` if/elif chain so older code
   paths keep working.

Plugins register themselves at import time via the ``mmalign.plugins``
entry point group; see ``core/registries.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from mmalign.core.method import InferenceAlignMethod


class UnknownMethodError(ValueError):
    pass


def load_method(method_id: str, config_path: str | Path) -> InferenceAlignMethod:
    """Load and instantiate an InferenceAlignMethod from methods.yaml."""
    raw: dict[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    methods: dict[str, Any] = raw.get("methods", {})

    if method_id not in methods:
        raise UnknownMethodError(
            f"Method '{method_id}' not found in {config_path}. "
            f"Available: {list(methods.keys())}"
        )

    cfg = dict(methods[method_id])
    method_type: str = cfg.pop("type", method_id)

    # Trigger plugin discovery once per process so third-party method
    # types appear in the registry before lookup.
    from mmalign.core.registries import inference_align_registry, load_plugin_methods

    load_plugin_methods()

    factory = inference_align_registry.get_or_none(method_type)
    if factory is not None:
        return cast(InferenceAlignMethod, factory(cfg))

    return _build_method(method_id, method_type, cfg)


def _build_method(method_id: str, method_type: str, cfg: dict[str, Any]) -> InferenceAlignMethod:
    if method_type == "raw":
        from mmalign.methods.raw import RawMethod
        return RawMethod()

    if method_type == "vanilla":
        from mmalign.methods.vanilla import VanillaMethod
        return VanillaMethod(
            include_safety_specs=cfg.get("include_safety_specs", True),
            include_behavioral_specs=cfg.get("include_behavioral_specs", True),
        )

    if method_type == "align3":
        from mmalign.methods.align3 import Align3Method
        return Align3Method(
            stages=cfg.get("stages", ["scenario_analysis", "behavior_planning",
                                       "safety_verification", "synthesis_reflection"]),
            focus_violated_specs=cfg.get("focus_violated_specs", True),
            max_stage_tokens=cfg.get("max_stage_tokens", 1024),
            temperature=cfg.get("temperature", 0.2),
        )

    if method_type == "self_refine":
        from mmalign.methods.self_refine import SelfRefineMethod
        return SelfRefineMethod(
            max_iterations=cfg.get("max_iterations", 3),
            temperature=cfg.get("temperature", 0.3),
        )

    if method_type == "zero_think":
        from mmalign.methods.specbench_ttd import ZeroThinkMethod
        return ZeroThinkMethod(temperature=cfg.get("temperature", 0.2))

    if method_type == "vanilla_specbench":
        from mmalign.methods.specbench_ttd import VanillaSpecBenchMethod
        return VanillaSpecBenchMethod(temperature=cfg.get("temperature", 0.2))

    if method_type == "align3_specbench":
        from mmalign.methods.specbench_ttd import Align3SpecBenchMethod
        return Align3SpecBenchMethod(
            temperature=cfg.get("temperature", 0.2),
            max_stage_tokens=cfg.get("max_stage_tokens", 1024),
        )

    if method_type == "more_think":
        from mmalign.methods.specbench_ttd import MoreThinkMethod
        return MoreThinkMethod(
            rethink_count=cfg.get("rethink_count", 3),
            temperature=cfg.get("temperature", 0.2),
        )

    if method_type == "best_of_n":
        from mmalign.methods.specbench_ttd import BestOfNMethod
        return BestOfNMethod(
            n=cfg.get("n", 4),
            temperature=cfg.get("temperature", 0.7),
        )

    if method_type == "tpo":
        from mmalign.methods.specbench_ttd import TPOMethod
        return TPOMethod(
            iterations=cfg.get("iterations", 2),
            sample_size=cfg.get("sample_size", 3),
            temperature=cfg.get("temperature", 0.7),
        )

    if method_type == "ipr_prompt_refine":
        from mmalign.methods.specbench_ttd import IPRPromptRefineMethod
        return IPRPromptRefineMethod(
            iterations=cfg.get("iterations", 3),
            temperature=cfg.get("temperature", 0.2),
        )

    if method_type == "vlm_safety_guard":
        from mmalign.methods.specbench_ttd import VLMSafetyGuardMethod
        return VLMSafetyGuardMethod(temperature=cfg.get("temperature", 0.2))

    if method_type == "overrefusal_check":
        from mmalign.methods.specbench_ttd import OverRefusalCheckMethod
        return OverRefusalCheckMethod(temperature=cfg.get("temperature", 0.2))

    if method_type == "user_conditioned_safety_critic":
        from mmalign.methods.specbench_ttd import UserConditionedSafetyCriticMethod
        return UserConditionedSafetyCriticMethod(temperature=cfg.get("temperature", 0.2))

    if method_type == "critic_revise_agent":
        from mmalign.methods.specbench_ttd import CriticReviseAgentMethod
        return CriticReviseAgentMethod(temperature=cfg.get("temperature", 0.2))

    if method_type == "risk_adaptive_agent":
        from mmalign.methods.specbench_ttd import RiskAdaptiveAgentMethod
        return RiskAdaptiveAgentMethod(temperature=cfg.get("temperature", 0.2))

    if method_type == "multi_agent_consensus":
        from mmalign.methods.specbench_ttd import MultiAgentConsensusMethod
        return MultiAgentConsensusMethod(temperature=cfg.get("temperature", 0.2))

    from mmalign.methods.safety_defenses import SAFETY_DEFENSES, build_safety_defense

    if method_type in SAFETY_DEFENSES:
        return build_safety_defense(method_type, cfg)

    raise UnknownMethodError(
        f"Unknown method type '{method_type}' for '{method_id}'. "
        "Supported: raw, vanilla, align3, self_refine, vanilla_specbench, "
        "align3_specbench, zero_think, more_think, best_of_n, tpo, "
        "ipr_prompt_refine, vlm_safety_guard, overrefusal_check, "
        "user_conditioned_safety_critic, critic_revise_agent, "
        "risk_adaptive_agent, multi_agent_consensus "
        "(plus plugins registered via core.registries.inference_align_registry)."
    )
