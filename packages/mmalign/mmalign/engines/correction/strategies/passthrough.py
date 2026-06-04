"""Passthrough strategy — forward request to backend without modification."""

from __future__ import annotations

from mmalign.core.control import ControlAttempt, ControlPolicy, RiskAssessment
from mmalign.core.scenario import Scenario
from mmalign.core.types import GenerationRequest


class PassthroughStrategy:
    """Safe-bucket strategy: forward the original request unchanged."""

    name = "passthrough"
    supported_modalities = {"text", "image", "video", "audio"}

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
        response = await backend.generate(request)
        return ControlAttempt(
            strategy=self.name,
            request=request,
            response=response,
            risk_level=risk.level,
            passed=True,
        )
