"""SelfRefineMethod — iterative self-correction alignment.

Process
-------
1. Generate initial response (like raw method).
2. Build a self-critique prompt using the scenario specs.
3. Ask the model to identify which specs the response might violate.
4. If violations found, ask the model to revise the response.
5. Repeat up to max_iterations or until the self-critique confirms compliance.
"""

from __future__ import annotations

from typing import Any

from mmalign.core.backend import ModelBackend
from mmalign.core.method import AlignContext, AlignedResponse, IncompatibleMethodError
from mmalign.core.scenario import Scenario
from mmalign.core.types import (
    GenerationRequest,
    InputMessage,
    Modality,
    OutputContentPart,
)
from .align3 import _extract_user_text, _spec_list, _sum_usage

_SYSTEM = "You are a safety-aware AI assistant. Follow the user's instructions carefully."


class SelfRefineMethod:
    """Iterative self-correction: generate → critique → revise, up to N rounds."""

    name = "self_refine"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(
        self,
        max_iterations: int = 3,
        temperature: float = 0.3,
        spec_aware_initial: bool = True,
    ) -> None:
        self._max_iterations = max_iterations
        self._temperature = temperature
        # The first draft sees the scenario specifications too (as in SpecBench's Self-Refine,
        # which starts from the spec-augmented prompt); False restores the old spec-free draft.
        self._spec_aware_initial = spec_aware_initial

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        if scenario.modality not in self.supported_modalities | {"multimodal"}:
            raise IncompatibleMethodError(
                f"self_refine: unsupported scenario modality '{scenario.modality}'."
            )
        if "text" not in backend.capabilities.output_modalities:
            raise IncompatibleMethodError(
                f"self_refine: backend '{backend.model_id}' does not produce text output."
            )

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        safety_list = _spec_list(scenario.safety_specs)
        behavioral_list = _spec_list(scenario.behavioral_specs)
        all_usages: list[dict[str, int]] = []
        iteration_history: list[dict[str, str]] = []

        # Initial generation
        initial_system = _SYSTEM
        if self._spec_aware_initial and (scenario.safety_specs or scenario.behavioral_specs):
            initial_system = (
                f"{_SYSTEM}\n\nFollow these specifications.\n\n"
                f"Safety specifications:\n{safety_list}\n\n"
                f"Behavioral specifications:\n{behavioral_list}"
            )
        initial_req = GenerationRequest(
            model=request.model,
            input=[InputMessage.system(initial_system), InputMessage.user(user_text)],
            temperature=self._temperature,
            max_output_tokens=request.max_output_tokens,
        )
        gr = await model.generate(initial_req)
        current_response = gr.text()
        all_usages.append(gr.usage)
        iteration_history.append({"iteration": "0", "response": current_response})

        for iteration in range(1, self._max_iterations + 1):
            # Self-critique
            critique_prompt = (
                f"User request:\n{user_text}\n\n"
                f"Current response:\n{current_response}\n\n"
                f"Safety specifications:\n{safety_list}\n\n"
                f"Behavioral specifications:\n{behavioral_list}\n\n"
                "Does the current response violate any specification? "
                "If YES, state which specs are violated and why. "
                "If NO, respond with exactly: COMPLIANT"
            )
            critique_req = GenerationRequest(
                model=request.model,
                input=[InputMessage.system(_SYSTEM), InputMessage.user(critique_prompt)],
                temperature=0,
                max_output_tokens=512,
            )
            critique_gr = await model.generate(critique_req)
            critique_text = critique_gr.text().strip()
            all_usages.append(critique_gr.usage)
            iteration_history.append({"iteration": f"critique_{iteration}", "response": critique_text})

            if critique_text.upper().startswith("COMPLIANT"):
                break

            # Refinement
            refine_prompt = (
                f"User request:\n{user_text}\n\n"
                f"Previous response:\n{current_response}\n\n"
                f"Issues identified:\n{critique_text}\n\n"
                f"Safety specifications:\n{safety_list}\n\n"
                f"Behavioral specifications:\n{behavioral_list}\n\n"
                "Revise the response to fix the identified issues while preserving "
                "all compliant parts. Output only the revised response."
            )
            refine_req = GenerationRequest(
                model=request.model,
                input=[InputMessage.system(_SYSTEM), InputMessage.user(refine_prompt)],
                temperature=self._temperature,
                max_output_tokens=request.max_output_tokens,
            )
            refine_gr = await model.generate(refine_req)
            current_response = refine_gr.text()
            all_usages.append(refine_gr.usage)
            iteration_history.append({"iteration": str(iteration), "response": current_response})

        thought = f"[self_refine] iterations={len([h for h in iteration_history if 'critique' not in h['iteration']])}"
        return AlignedResponse(
            output=[OutputContentPart.text_part(current_response)],
            method=self.name,
            thought=thought,
            usage=_sum_usage(*all_usages),
            attempts=len(iteration_history),
            metadata={"iterations": iteration_history},
        )
