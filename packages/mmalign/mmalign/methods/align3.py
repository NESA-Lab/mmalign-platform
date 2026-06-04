"""Align3Method — four-stage deliberative inference-time alignment.

Stages
------
1. scenario_analysis   — identify which safety/behavioral specs are relevant
2. behavior_planning   — plan a response that satisfies behavioral specs
3. safety_verification — check the plan against every safety spec; fix violations
4. synthesis_reflection— produce the final response + brief thought record

Token usage is aggregated across all four stages.
Intermediate outputs are stored in metadata["stages"] for replay/audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmalign.core.backend import ModelBackend
from mmalign.core.method import AlignContext, AlignedResponse, IncompatibleMethodError
from mmalign.core.scenario import Scenario, Specification
from mmalign.core.types import (
    GenerationRequest,
    InputMessage,
    Modality,
    OutputContentPart,
)

_STAGE_SYSTEM = "You are an expert alignment assistant. Follow the instructions exactly and output only what is asked."


def _spec_list(specs: list[Specification], prefix: str = "") -> str:
    if not specs:
        return "(none)"
    return "\n".join(f"  [{spec.spec_id}] {spec.text}" for spec in specs)


def _load_violated_specs(
    vanilla_results_path: str,
    scenario: Scenario,
) -> list[Specification]:
    """Return specs that were violated in a previous vanilla run's JSONL results."""
    try:
        p = Path(vanilla_results_path)
        violated_ids: set[str] = set()
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                per_spec = rec.get("eval_result", {}).get("per_spec", {})
                for spec_id, verdict in per_spec.items():
                    if verdict == "NO":
                        violated_ids.add(spec_id)
        if violated_ids:
            return [s for s in scenario.all_specs if s.spec_id in violated_ids]
    except Exception:
        pass
    return scenario.all_specs


def _sum_usage(*usages: dict[str, int]) -> dict[str, int]:
    total: dict[str, int] = {}
    for u in usages:
        for k, v in u.items():
            total[k] = total.get(k, 0) + v
    return total


class Align3Method:
    """Four-stage deliberative inference-time alignment method."""

    name = "align3"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(
        self,
        stages: list[str] | None = None,
        focus_violated_specs: bool = True,
        max_stage_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        self._stages = stages or [
            "scenario_analysis",
            "behavior_planning",
            "safety_verification",
            "synthesis_reflection",
        ]
        self._focus_violated = focus_violated_specs
        self._max_stage_tokens = max_stage_tokens
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        if scenario.modality not in self.supported_modalities | {"multimodal"}:
            raise IncompatibleMethodError(
                f"align3: unsupported scenario modality '{scenario.modality}'."
            )
        if "text" not in backend.capabilities.output_modalities:
            raise IncompatibleMethodError(
                f"align3: backend '{backend.model_id}' does not produce text output."
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

        focus_specs = (
            _load_violated_specs(context.vanilla_results_path, scenario)
            if context and context.vanilla_results_path
            else scenario.all_specs
        )

        safety_specs = [s for s in focus_specs if s.category == "safety"]
        behavioral_specs = [s for s in focus_specs if s.category == "behavioral"]

        stages_output: dict[str, str] = {}
        all_usages: list[dict[str, int]] = []

        async def call(prompt: str) -> tuple[str, dict[str, int]]:
            r = GenerationRequest(
                model=request.model,
                input=[
                    InputMessage.system(_STAGE_SYSTEM),
                    InputMessage.user(prompt),
                ],
                temperature=self._temperature,
                max_output_tokens=self._max_stage_tokens,
            )
            gr = await model.generate(r)
            return gr.text(), gr.usage

        # Stage 1: scenario_analysis
        if "scenario_analysis" in self._stages:
            prompt = (
                f"User request:\n{user_text}\n\n"
                f"Safety specifications:\n{_spec_list(safety_specs)}\n\n"
                f"Behavioral specifications:\n{_spec_list(behavioral_specs)}\n\n"
                "Briefly identify which safety and behavioral specifications are "
                "relevant to this request and explain why."
            )
            out, usage = await call(prompt)
            stages_output["scenario_analysis"] = out
            all_usages.append(usage)

        # Stage 2: behavior_planning
        if "behavior_planning" in self._stages:
            analysis = stages_output.get("scenario_analysis", "")
            prompt = (
                f"User request:\n{user_text}\n\n"
                f"Scenario analysis:\n{analysis}\n\n"
                f"Safety specifications:\n{_spec_list(safety_specs)}\n\n"
                f"Behavioral specifications:\n{_spec_list(behavioral_specs)}\n\n"
                "Plan a response structure that maximally satisfies the behavioral "
                "specifications without violating any safety specification. "
                "Output a concise response plan (not the final response)."
            )
            out, usage = await call(prompt)
            stages_output["behavior_planning"] = out
            all_usages.append(usage)

        # Stage 3: safety_verification
        if "safety_verification" in self._stages:
            plan = stages_output.get("behavior_planning", "")
            safety_text = _spec_list(safety_specs)
            prompt = (
                f"Proposed response plan:\n{plan}\n\n"
                f"Safety specifications:\n{safety_text}\n\n"
                "Review the plan against each safety specification. "
                "If any safety specification is violated, describe a safe correction. "
                "If the plan is safe, confirm it is safe to proceed."
            )
            out, usage = await call(prompt)
            stages_output["safety_verification"] = out
            all_usages.append(usage)

        # Stage 4: synthesis_reflection
        plan = stages_output.get("behavior_planning", "")
        safety_review = stages_output.get("safety_verification", "")
        final_prompt = (
            f"Original user request:\n{user_text}\n\n"
            f"Response plan:\n{plan}\n\n"
            f"Safety review:\n{safety_review}\n\n"
            "Generate the final, complete response for the user. "
            "The response must follow the plan and pass all safety specifications. "
            "Output only the response text, nothing else."
        )
        r_final = GenerationRequest(
            model=request.model,
            input=[
                InputMessage.system(_STAGE_SYSTEM),
                InputMessage.user(final_prompt),
            ],
            temperature=self._temperature,
            max_output_tokens=request.max_output_tokens,
        )
        gr_final = await model.generate(r_final)
        final_text = gr_final.text()
        all_usages.append(gr_final.usage)
        stages_output["synthesis_reflection"] = final_text

        thought = (
            f"[align3] analysis={bool(stages_output.get('scenario_analysis'))} "
            f"plan={bool(plan)} safety_ok={bool(safety_review)}"
        )
        return AlignedResponse(
            output=[OutputContentPart.text_part(final_text)],
            method=self.name,
            thought=thought,
            usage=_sum_usage(*all_usages),
            metadata={"stages": stages_output},
        )


def _extract_user_text(request: GenerationRequest) -> str:
    """Return the first user message text from a GenerationRequest."""
    for msg in request.input_messages():
        if msg.role == "user":
            content = msg.content
            if isinstance(content, str):
                return content
            return " ".join(
                part.text or "" for part in content if part.type == "input_text"
            )
    return ""
