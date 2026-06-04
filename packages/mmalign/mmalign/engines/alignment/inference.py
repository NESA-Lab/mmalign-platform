"""InferenceAlignmentEngine — thin orchestration layer over InferenceAlignMethod.

Responsibilities
----------------
* Load the specified method and backend from config.
* Call ``method.validate(scenario, backend)`` pre-flight.
* Call ``method.align(request, backend, scenario, context)`` for a single sample.
* Normalise the result to a ``GenerationResult`` suitable for an Evaluator.

This engine is intentionally thin: the complex multi-sample orchestration
(concurrency, retry, ResultStore writes) lives in DiagnosisEngine.runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmalign.core.method import AlignContext, AlignedResponse, InferenceAlignMethod
from mmalign.core.scenario import Scenario
from mmalign.core.types import GenerationRequest


class InferenceAlignmentEngine:
    """Run a single inference-time alignment call.

    Parameters
    ----------
    configs_dir:
        Path to the project configs directory (contains methods.yaml, models.yaml).
    """

    def __init__(self, configs_dir: Path) -> None:
        self._configs_dir = configs_dir

    async def align(
        self,
        request: GenerationRequest,
        model_id: str,
        scenario: Scenario,
        method_name: str = "align3",
        context: AlignContext | None = None,
    ) -> AlignedResponse:
        """Run inference-time alignment and return the aligned response.

        Parameters
        ----------
        request:
            The generation request (may include multi-modal input).
        model_id:
            Model ID as registered in models.yaml.
        scenario:
            The scenario providing safety / behavioral specs.
        method_name:
            Name of the InferenceAlignMethod to use (default: ``align3``).
        context:
            Optional AlignContext for method-private dependencies (e.g.
            ``vanilla_results_path`` for align3 focus mode).
        """
        from mmalign.backends.loader import load_backends
        from mmalign.loaders.methods import load_method

        backends = load_backends(
            self._configs_dir / "models.yaml", only=[model_id]
        )
        backend = backends[model_id]
        method = load_method(method_name, self._configs_dir / "methods.yaml")
        method.validate(scenario, backend)
        return await method.align(request, backend, scenario, context)

    def list_methods(self) -> list[str]:
        """Return registered method names from methods.yaml."""
        import yaml

        path = self._configs_dir / "methods.yaml"
        if not path.exists():
            return []
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return list((raw or {}).get("methods", {}).keys())
