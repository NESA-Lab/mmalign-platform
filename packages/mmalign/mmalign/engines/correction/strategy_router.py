"""StrategyRouter — maps risk level + ControlPolicy to a ControlStrategy instance.

Built-in strategy registry
--------------------------
Built-in names cannot be overwritten; operators can only change the
``strategy_mapping`` in a ``ControlPolicy`` to point to custom names that
have been registered with ``register()``.
"""

from __future__ import annotations

from typing import Any

from mmalign.core.control import ControlPolicy, RiskLevel

from .strategies import (
    PassthroughStrategy,
    RefusalStrategy,
    SemanticRewriteStrategy,
    TTDWrapperStrategy,
    VanillaStrategy,
)

# Built-in strategies — immutable
_BUILTIN: dict[str, Any] = {
    "passthrough": PassthroughStrategy(),
    "vanilla": VanillaStrategy(),
    "ttd_wrapper": TTDWrapperStrategy(method_name="align3"),
    "align3": TTDWrapperStrategy(method_name="align3"),
    "self_refine": TTDWrapperStrategy(method_name="self_refine"),
    "spec_rewrite": SemanticRewriteStrategy(),
    "semantic_rewrite": SemanticRewriteStrategy(),
    "refusal": RefusalStrategy(),
}

_CUSTOM: dict[str, Any] = {}


def register(name: str, strategy: Any) -> None:
    """Register a custom ControlStrategy.  Built-in names are protected."""
    if name in _BUILTIN:
        raise ValueError(
            f"Cannot overwrite built-in strategy '{name}'. "
            "Register under a new name and update the policy mapping."
        )
    _CUSTOM[name] = strategy


def get(name: str) -> Any:
    """Return strategy by name; raises KeyError if not found."""
    if name in _BUILTIN:
        return _BUILTIN[name]
    if name in _CUSTOM:
        return _CUSTOM[name]
    raise KeyError(f"Unknown strategy: '{name}'")


class StrategyRouter:
    """Resolve the correct strategy for a given risk level and policy."""

    def route(self, level: RiskLevel, policy: ControlPolicy) -> Any:
        """Return the ControlStrategy instance for the given risk level."""
        strategy_name = policy.strategy_for(level)
        try:
            return get(strategy_name)
        except KeyError:
            # Fallback: refusal if configured strategy is missing
            return _BUILTIN["refusal"]

    def escalate(self, current_strategy: str, policy: ControlPolicy) -> Any:
        """Return the next escalation strategy when the current one fails."""
        _ladder = ["passthrough", "vanilla", "align3", "spec_rewrite", "refusal"]
        try:
            idx = _ladder.index(current_strategy)
            next_name = _ladder[min(idx + 1, len(_ladder) - 1)]
        except ValueError:
            next_name = "refusal"
        return get(next_name)
