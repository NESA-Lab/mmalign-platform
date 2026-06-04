"""Evaluator loader — instantiates Evaluator from configs/evaluators.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from mmalign.core.evaluator import Evaluator


class UnknownEvaluatorError(ValueError):
    pass


def load_evaluator(
    evaluator_id: str,
    config_path: str | Path,
    *,
    backend_loader: Any | None = None,
    models_config_path: str | Path | None = None,
) -> Evaluator:
    """Load and instantiate an Evaluator from evaluators.yaml.

    For ``llm_judge``, the judge model backend is resolved via
    ``backend_loader`` (a dict or callable).  If ``models_config_path``
    is given and ``backend_loader`` is None, backends are auto-loaded.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    evaluators: dict[str, Any] = raw.get("evaluators", {})

    if evaluator_id not in evaluators:
        raise UnknownEvaluatorError(
            f"Evaluator '{evaluator_id}' not found in {config_path}. "
            f"Available: {list(evaluators.keys())}"
        )

    cfg = dict(evaluators[evaluator_id])
    evaluator_type: str = cfg.pop("type", evaluator_id)

    # Consult the global registry first so plugins win for types that don't
    # need a backend resolver (mock_judge, rule_based, custom plugins).
    # ``llm_judge`` is intentionally not registered: it needs a judge backend
    # which only the legacy path knows how to resolve.
    from mmalign.core.registries import evaluator_registry, load_plugin_methods

    load_plugin_methods()
    factory = evaluator_registry.get_or_none(evaluator_type)
    if factory is not None:
        return cast(Evaluator, factory(cfg))

    return _build_evaluator(evaluator_id, evaluator_type, cfg,
                             backend_loader=backend_loader,
                             models_config_path=models_config_path)


def _build_evaluator(
    evaluator_id: str,
    evaluator_type: str,
    cfg: dict[str, Any],
    *,
    backend_loader: Any | None,
    models_config_path: str | Path | None,
) -> Evaluator:
    if evaluator_type == "mock_judge":
        from mmalign.evaluators.mock_judge import MockJudgeEvaluator
        return MockJudgeEvaluator(deterministic=cfg.get("deterministic", True))

    if evaluator_type == "rule_based":
        from mmalign.evaluators.rule_based import RuleBasedEvaluator
        return RuleBasedEvaluator()

    if evaluator_type == "llm_judge":
        judge_model_id: str = cfg.get("judge_model", "gpt-4.1")

        # Resolve judge backend
        if isinstance(backend_loader, dict):
            judge_backend = backend_loader[judge_model_id]
        elif callable(backend_loader):
            judge_backend = backend_loader(judge_model_id)
        elif models_config_path is not None:
            from mmalign.backends.loader import load_backends
            backends = load_backends(models_config_path, only=[judge_model_id])
            judge_backend = backends[judge_model_id]
        else:
            raise UnknownEvaluatorError(
                f"llm_judge evaluator '{evaluator_id}' requires models_config_path "
                "or backend_loader to resolve judge_model backend."
            )

        from mmalign.evaluators.llm_judge import LLMJudgeEvaluator
        return LLMJudgeEvaluator(
            judge_backend=judge_backend,
            mode=cfg.get("mode", "joint"),
            temperature=cfg.get("temperature", 0),
            max_output_tokens=cfg.get("max_output_tokens", 2048),
            retries=cfg.get("retries", 2),
            strict_json=cfg.get("strict_json", True),
        )

    raise UnknownEvaluatorError(
        f"Unknown evaluator type '{evaluator_type}' for '{evaluator_id}'. "
        "Supported: llm_judge, mock_judge, rule_based."
    )
