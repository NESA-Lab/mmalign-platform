"""Refusal fallback strategy — return a safe refusal without calling backend."""

from __future__ import annotations

import uuid

from mmalign.core.control import ControlAttempt, ControlPolicy, RiskAssessment
from mmalign.core.scenario import Scenario
from mmalign.core.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationResult,
    OutputContentPart,
)

_REFUSAL_MESSAGE = (
    "I'm sorry, but I'm unable to assist with this request as it may violate "
    "safety guidelines. If you have other questions or need help with something "
    "else, I'm happy to help."
)

_REFUSAL_MESSAGE_ZH = (
    "抱歉，我无法协助完成此请求，因为它可能违反安全准则。"
    "如果您有其他问题或需要其他方面的帮助，我很乐意为您服务。"
)


class RefusalStrategy:
    """Last-resort fallback: return a pre-written refusal without any model call."""

    name = "refusal"
    supported_modalities = {"text", "image", "video", "audio"}

    async def generate(
        self,
        request: GenerationRequest,
        backend: object,
        scenario: Scenario,
        risk: RiskAssessment,
        policy: ControlPolicy,
    ) -> ControlAttempt:
        # Detect language from input text
        inp = request.input
        text_sample = inp if isinstance(inp, str) else ""
        is_chinese = any("\u4e00" <= c <= "\u9fff" for c in text_sample[:100])
        refusal_text = _REFUSAL_MESSAGE_ZH if is_chinese else _REFUSAL_MESSAGE

        response = GenerationResult(
            id=str(uuid.uuid4()),
            model=request.model,
            output=[
                GenerationOutput(
                    content=[OutputContentPart.refusal_part(refusal_text)]
                )
            ],
            metadata={"strategy": self.name, "risk_level": risk.level},
        )
        return ControlAttempt(
            strategy=self.name,
            request=request,
            response=response,
            risk_level=risk.level,
            passed=True,
            metadata={"refusal_reason": risk.reason},
        )
