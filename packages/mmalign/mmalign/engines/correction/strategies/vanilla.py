"""Vanilla (spec-prompting) strategy — prepend scenario specifications."""

from __future__ import annotations

from mmalign.core.control import ControlAttempt, ControlPolicy, RiskAssessment
from mmalign.core.scenario import Scenario
from mmalign.core.types import GenerationRequest, InputContentPart, InputMessage


def _build_spec_system_message(scenario: Scenario) -> str:
    lines = [f"You are operating under the '{scenario.name}' safety scenario."]
    lines.append("You MUST comply with ALL of the following specifications:\n")
    for spec in scenario.safety_specs:
        lines.append(f"[SAFETY] {spec.text}")
    for spec in scenario.behavioral_specs:
        lines.append(f"[BEHAVIORAL] {spec.text}")
    return "\n".join(lines)


class VanillaStrategy:
    """Low-risk bucket: inject all scenario specs into the system message."""

    name = "vanilla"
    supported_modalities = {"text", "image"}

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

        spec_text = _build_spec_system_message(scenario)
        messages = request.input_messages()

        # Prepend spec system message before any existing system message
        augmented = [InputMessage.system(spec_text)] + [
            m for m in messages if m.role != "system"
        ]
        new_request = request.model_copy(update={"input": augmented})

        response = await backend.generate(new_request)
        return ControlAttempt(
            strategy=self.name,
            request=new_request,
            response=response,
            risk_level=risk.level,
            passed=True,
        )
