"""VanillaMethod — simplest spec-prompting: prepend all specs to the prompt."""

from __future__ import annotations

from typing import Any

from mmalign.core.backend import ModelBackend
from mmalign.core.method import AlignContext, AlignedResponse, IncompatibleMethodError
from mmalign.core.scenario import Scenario
from mmalign.core.types import (
    GenerationRequest,
    InputContentPart,
    InputMessage,
    Modality,
)


def _build_spec_system_prompt(scenario: Scenario,
                               include_safety: bool,
                               include_behavioral: bool) -> str:
    parts: list[str] = [
        "You must follow the specifications below when generating your response.\n"
    ]
    if include_safety and scenario.safety_specs:
        parts.append("== Safety Specifications ==")
        for spec in scenario.safety_specs:
            parts.append(f"- [{spec.spec_id}] {spec.text}")
    if include_behavioral and scenario.behavioral_specs:
        parts.append("\n== Behavioral Specifications ==")
        for spec in scenario.behavioral_specs:
            parts.append(f"- [{spec.spec_id}] {spec.text}")
    return "\n".join(parts)


class VanillaMethod:
    """Spec prompting: inject all scenario specs as a system message prefix."""

    name = "vanilla"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(
        self,
        include_safety_specs: bool = True,
        include_behavioral_specs: bool = True,
    ) -> None:
        self._include_safety = include_safety_specs
        self._include_behavioral = include_behavioral_specs

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        modality = scenario.modality
        if modality not in ("text", "image", "multimodal"):
            raise IncompatibleMethodError(
                f"vanilla: unsupported scenario modality '{modality}'."
            )
        if "text" not in backend.capabilities.output_modalities:
            raise IncompatibleMethodError(
                f"vanilla: backend '{backend.model_id}' does not produce text output."
            )

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        spec_text = _build_spec_system_prompt(
            scenario, self._include_safety, self._include_behavioral
        )

        # Build modified request: prepend spec system message
        original_messages = request.input_messages()
        spec_message = InputMessage(
            role="system",
            content=[InputContentPart.text_part(spec_text)],
        )

        # If there's already a system message, merge; otherwise prepend
        has_system = original_messages and original_messages[0].role == "system"
        if has_system:
            existing_sys = original_messages[0]
            if isinstance(existing_sys.content, str):
                merged = f"{spec_text}\n\n{existing_sys.content}"
                merged_msg = InputMessage(
                    role="system",
                    content=[InputContentPart.text_part(merged)],
                )
            else:
                merged_msg = InputMessage(
                    role="system",
                    content=[InputContentPart.text_part(spec_text)] + list(existing_sys.content),
                )
            new_messages = [merged_msg] + original_messages[1:]
        else:
            if model.capabilities.supports_system_message:
                new_messages = [spec_message] + original_messages
            else:
                # Prepend to first user message content
                user_msgs = list(original_messages)
                if user_msgs:
                    first = user_msgs[0]
                    prefix = InputContentPart.text_part(
                        f"{spec_text}\n\nUser request:\n"
                    )
                    if isinstance(first.content, str):
                        new_content = [prefix, InputContentPart.text_part(first.content)]
                    else:
                        new_content = [prefix] + list(first.content)
                    user_msgs[0] = InputMessage(role="user", content=new_content)
                new_messages = user_msgs

        modified_request = request.model_copy(update={"input": new_messages})
        result = await model.generate(modified_request, **kwargs)
        return AlignedResponse(
            output=result.output[0].content if result.output else [],
            method=self.name,
            usage=result.usage,
        )
