"""TTD wrapper strategy — delegates to an InferenceAlignMethod (e.g. align3).

This strategy wraps the configured alignment method (default: align3) so that
the Correction Engine can use the same deliberative reasoning pipeline that
the DiagnosisEngine uses for offline evaluation.
"""

from __future__ import annotations

from typing import Any

from mmalign.core.control import ControlAttempt, ControlPolicy, RiskAssessment
from mmalign.core.method import AlignContext
from mmalign.core.result import BackendTelemetry
from mmalign.core.scenario import Scenario
from mmalign.core.types import GenerationRequest, GenerationOutput, GenerationResult, OutputContentPart
import uuid


class TTDWrapperStrategy:
    """Medium-risk bucket: run the configured InferenceAlignMethod."""

    name = "ttd_wrapper"
    supported_modalities = {"text", "image"}

    def __init__(self, method_name: str = "align3") -> None:
        self._method_name = method_name
        self._method: Any = None  # lazy-loaded

    def _get_method(self) -> Any:
        if self._method is None:
            from mmalign.methods.align3 import Align3Method
            from mmalign.methods.self_refine import SelfRefineMethod
            if self._method_name == "self_refine":
                self._method = SelfRefineMethod()
            else:
                self._method = Align3Method()
        return self._method

    async def generate(
        self,
        request: GenerationRequest,
        backend: object,
        scenario: Scenario,
        risk: RiskAssessment,
        policy: ControlPolicy,
    ) -> ControlAttempt:
        from mmalign.core.backend import ModelBackend
        assert isinstance(backend, ModelBackend)

        method = self._get_method()
        context = AlignContext(
            metadata={"triggered_specs": risk.triggered_specs}
        )
        aligned = await method.align(request, backend, scenario, context)

        # Wrap AlignedResponse into a GenerationResult
        result = GenerationResult(
            id=str(uuid.uuid4()),
            model=request.model,
            output=[GenerationOutput(content=aligned.output)],
            usage=aligned.usage,
            metadata={"thought": aligned.thought, "method": aligned.method, "attempts": aligned.attempts},
        )
        return ControlAttempt(
            strategy=self.name,
            request=request,
            response=result,
            risk_level=risk.level,
            passed=not aligned.is_refusal(),
            metadata={"thought": aligned.thought, "attempts": aligned.attempts},
        )
