"""Global registries for the six pluggable component types.

These are the canonical lookup tables that ``load_method``,
``load_evaluator``, and the strategy router consult to resolve a name to
an implementation.  Built-in entries are registered at import time below;
external plugins must register under namespaced names (e.g.
``my_org.align3``) to avoid colliding with the protected built-ins.

Why factories?
--------------
Methods and evaluators receive per-instance config (temperature, stages,
judge model, ...) so the registry stores **factory callables** that take
a config ``dict`` and return a fresh instance.  Scenarios and backends
are loaded from on-disk artifacts and are not represented here as
factories — they retain their own loader paths but the registries below
are reserved for future wiring.

Plugin entry points
-------------------
Third-party packages can register on import via the standard ``mmalign``
entry-point group.  At runtime call :func:`load_plugin_methods` (invoked
implicitly by ``load_method``) to walk ``importlib.metadata.entry_points``
and trigger registration.  The discovery is idempotent and cached.
"""

from __future__ import annotations

from typing import Any, Callable

from .registry import Registry

InferenceFactory = Callable[[dict[str, Any]], Any]
TrainingFactory = Callable[[dict[str, Any]], Any]
EvaluatorFactory = Callable[[dict[str, Any]], Any]
StrategyFactory = Callable[[dict[str, Any]], Any]
BackendFactory = Callable[[dict[str, Any]], Any]


inference_align_registry: Registry[InferenceFactory] = Registry("inference_align")
"""Factories for InferenceAlignMethod implementations, keyed by ``type``."""

training_align_registry: Registry[TrainingFactory] = Registry("training_align")
"""Factories for TrainingAlignMethod implementations."""

control_strategy_registry: Registry[StrategyFactory] = Registry("control_strategy")
"""Factories for ControlStrategy implementations.  The built-in strategy
router still owns its own immutable singleton table for back-compat; this
registry is the discovery surface for plugins."""

evaluator_registry: Registry[EvaluatorFactory] = Registry("evaluator")
"""Factories for Evaluator implementations.  Factory signature:
``(cfg: dict) -> Evaluator``.  Evaluators that need a judge backend
(``llm_judge``) are intentionally NOT registered here — they resolve
through ``loaders.evaluators._build_evaluator`` which has access to the
backend loader."""

scenario_registry: Registry[Any] = Registry("scenario")
"""Reserved for in-memory Scenario overrides / synthetic scenarios."""

backend_registry: Registry[BackendFactory] = Registry("backend")
"""Reserved for runtime ModelBackend factories.  Production usage flows
through ``backends/loader.py`` for YAML-defined backends."""


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    """Register all built-in implementations.  Called once at import time."""
    # InferenceAlignMethod
    def _make_raw(cfg: dict[str, Any]) -> Any:
        from mmalign.methods.raw import RawMethod
        return RawMethod()

    def _make_vanilla(cfg: dict[str, Any]) -> Any:
        from mmalign.methods.vanilla import VanillaMethod
        return VanillaMethod(
            include_safety_specs=cfg.get("include_safety_specs", True),
            include_behavioral_specs=cfg.get("include_behavioral_specs", True),
        )

    def _make_align3(cfg: dict[str, Any]) -> Any:
        from mmalign.methods.align3 import Align3Method
        return Align3Method(
            stages=cfg.get(
                "stages",
                [
                    "scenario_analysis",
                    "behavior_planning",
                    "safety_verification",
                    "synthesis_reflection",
                ],
            ),
            focus_violated_specs=cfg.get("focus_violated_specs", True),
            max_stage_tokens=cfg.get("max_stage_tokens", 1024),
            temperature=cfg.get("temperature", 0.2),
        )

    def _make_self_refine(cfg: dict[str, Any]) -> Any:
        from mmalign.methods.self_refine import SelfRefineMethod
        return SelfRefineMethod(
            max_iterations=cfg.get("max_iterations", 3),
            temperature=cfg.get("temperature", 0.3),
        )

    inference_align_registry.register("raw", _make_raw, builtin=True)
    inference_align_registry.register("vanilla", _make_vanilla, builtin=True)
    inference_align_registry.register("align3", _make_align3, builtin=True)
    inference_align_registry.register("self_refine", _make_self_refine, builtin=True)

    # Evaluators — only register those that don't need a judge backend.
    # ``llm_judge`` resolves through the legacy loader path because it
    # needs ``backend_loader``/``models_config_path`` to wire its judge
    # model.
    def _make_mock_judge(cfg: dict[str, Any]) -> Any:
        from mmalign.evaluators.mock_judge import MockJudgeEvaluator
        return MockJudgeEvaluator(deterministic=cfg.get("deterministic", True))

    def _make_rule_based(_cfg: dict[str, Any]) -> Any:
        from mmalign.evaluators.rule_based import RuleBasedEvaluator
        return RuleBasedEvaluator()

    evaluator_registry.register("mock_judge", _make_mock_judge, builtin=True)
    evaluator_registry.register("rule_based", _make_rule_based, builtin=True)

    # ControlStrategy — names mirror strategy_router._BUILTIN.  These are
    # discovery-only entries; the router still uses its own singleton table.
    def _make_passthrough(_cfg: dict[str, Any]) -> Any:
        from mmalign.engines.correction.strategies import PassthroughStrategy
        return PassthroughStrategy()

    def _make_vanilla_strategy(_cfg: dict[str, Any]) -> Any:
        from mmalign.engines.correction.strategies import VanillaStrategy
        return VanillaStrategy()

    def _make_ttd_wrapper(cfg: dict[str, Any]) -> Any:
        from mmalign.engines.correction.strategies import TTDWrapperStrategy
        return TTDWrapperStrategy(method_name=cfg.get("method_name", "align3"))

    def _make_semantic_rewrite(_cfg: dict[str, Any]) -> Any:
        from mmalign.engines.correction.strategies import SemanticRewriteStrategy
        return SemanticRewriteStrategy()

    def _make_refusal(_cfg: dict[str, Any]) -> Any:
        from mmalign.engines.correction.strategies import RefusalStrategy
        return RefusalStrategy()

    control_strategy_registry.register("passthrough", _make_passthrough, builtin=True)
    control_strategy_registry.register("vanilla", _make_vanilla_strategy, builtin=True)
    control_strategy_registry.register("ttd_wrapper", _make_ttd_wrapper, builtin=True)
    control_strategy_registry.register("align3", _make_ttd_wrapper, builtin=True)
    control_strategy_registry.register("self_refine", _make_ttd_wrapper, builtin=True)
    control_strategy_registry.register("spec_rewrite", _make_semantic_rewrite, builtin=True)
    control_strategy_registry.register("semantic_rewrite", _make_semantic_rewrite, builtin=True)
    control_strategy_registry.register("refusal", _make_refusal, builtin=True)

    # TrainingAlignMethod — DPO / SFT live in engines/alignment/training.py
    # and accept their config at ``train(config=...)`` rather than at
    # construction, so the factories take no kwargs from cfg.
    def _make_dpo(_cfg: dict[str, Any]) -> Any:
        from mmalign.engines.alignment.training import DPOAlignMethod
        return DPOAlignMethod()

    def _make_sft(_cfg: dict[str, Any]) -> Any:
        from mmalign.engines.alignment.training import SFTAlignMethod
        return SFTAlignMethod()

    training_align_registry.register("dpo", _make_dpo, builtin=True)
    training_align_registry.register("sft", _make_sft, builtin=True)


# ---------------------------------------------------------------------------
# Plugin entry-point discovery
# ---------------------------------------------------------------------------

_plugins_loaded = False


def load_plugin_methods() -> None:
    """Discover and load third-party registry contributions via entry points.

    Plugins declare an entry point under the ``mmalign.plugins`` group whose
    target is a callable.  The callable is invoked with no arguments and is
    expected to register its implementations into the global registries.

    Discovery runs at most once per process; subsequent calls are no-ops.
    Errors in plugin loading are logged but never raised, so a broken
    plugin cannot brick the runtime.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover — Python < 3.10 not supported
        return

    try:
        eps = entry_points(group="mmalign.plugins")
    except Exception:  # pragma: no cover — defensive
        return

    for ep in eps:
        try:
            target = ep.load()
            if callable(target):
                target()
        except Exception as exc:  # pragma: no cover — never let plugins crash us
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load mmalign plugin %r: %s", ep.name, exc
            )


_register_builtins()
