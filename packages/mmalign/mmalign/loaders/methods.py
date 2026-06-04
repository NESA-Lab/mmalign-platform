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

    raise UnknownMethodError(
        f"Unknown method type '{method_type}' for '{method_id}'. "
        "Supported: raw, vanilla, align3, self_refine "
        "(plus plugins registered via core.registries.inference_align_registry)."
    )
