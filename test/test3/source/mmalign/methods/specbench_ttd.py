"""SpecBench-style inference-time alignment methods.

The original SpecBench implementation contains useful test-time-deliberation
patterns, but its scripts are tied to local model paths and benchmark runners.
This module adapts the algorithmic pieces into MMAlign's native
InferenceAlignMethod interface without adding train-time dependencies.
"""

from __future__ import annotations

import uuid
import math
import re
from typing import Any

from mmalign.core.backend import IncompatibleBackendError, ModelBackend
from mmalign.core.method import AlignContext, AlignedResponse, IncompatibleMethodError
from mmalign.core.scenario import Scenario
from mmalign.core.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationResult,
    InputMessage,
    InputContentPart,
    Modality,
    OutputContentPart,
)

from .align3 import _extract_user_text, _spec_list, _sum_usage
from .multimodal_prompt import user_with_images

_SYSTEM = "You are an expert alignment assistant. Follow the user request while satisfying all specifications."
_RETHINK = "Wait, "


def _specbench_augmented_prompt(user_text: str, scenario: Scenario) -> str:
    return (
        "Your answer should satisfy all rules. If the user request conflicts "
        "with a safety rule, refuse or redirect safely. If it is feasible, "
        "answer helpfully and concisely.\n\n"
        f"Safety specifications:\n{_spec_list(scenario.safety_specs)}\n\n"
        f"Behavioral specifications:\n{_spec_list(scenario.behavioral_specs)}\n\n"
        f"User request:\n{user_text}"
    )


def _request_with_prompt(
    request: GenerationRequest,
    prompt: str,
    *,
    assistant_prefix: str | None = None,
    temperature: float | None = None,
) -> GenerationRequest:
    messages: list[InputMessage] = [
        InputMessage.system(_SYSTEM),
        user_with_images(request, prompt),
    ]
    if assistant_prefix:
        messages.append(InputMessage(role="assistant", content=assistant_prefix))
    return GenerationRequest(
        model=request.model,
        input=messages,
        modalities=request.modalities,
        audio=request.audio,
        temperature=temperature if temperature is not None else request.temperature,
        max_output_tokens=request.max_output_tokens,
        mmalign=request.mmalign,
        metadata=dict(request.metadata),
    )


def _final_audit_enabled(context: AlignContext | None) -> bool:
    return bool(context and context.metadata.get("enhanced_final_audit") is True)


def _final_specification_audit(prompt: str, scenario: Scenario, context: AlignContext | None) -> str:
    """Keep original requirements authoritative over generated reviews."""
    if not _final_audit_enabled(context):
        return prompt
    return (
        f"Safety specifications:\n{_spec_list(scenario.safety_specs)}\n\n"
        f"Behavioral specifications:\n{_spec_list(scenario.behavioral_specs)}\n\n"
        f"{prompt}\n\n"
        "Before answering, independently recheck the original image and every specification. "
        "Drafts and reviews may contain factual errors; do not copy them blindly. "
        "Correct unsupported technical claims or omit details you cannot verify. "
        "Use relevant visible evidence without inventing observations. Distinguish safe general "
        "guidance from attempting an unsafe action in the pictured setting; preserve useful "
        "safe assistance instead of blanket refusal. Output only the final answer."
    )


def _validate_text_method(name: str, scenario: Scenario, backend: ModelBackend) -> None:
    if scenario.modality not in {"text", "image", "multimodal"}:
        raise IncompatibleMethodError(
            f"{name}: unsupported scenario modality '{scenario.modality}'."
        )
    if "text" not in backend.capabilities.output_modalities:
        raise IncompatibleMethodError(
            f"{name}: backend '{backend.model_id}' does not produce text output."
        )


def _heuristic_score(response: str) -> float:
    """Fallback scorer for demos when no reward backend is available."""
    text = response.strip()
    if not text:
        return -1.0
    lowered = text.lower()
    score = min(len(text), 2000) / 2000.0
    if any(token in lowered for token in ("i can't", "i cannot", "sorry", "cannot help")):
        score += 0.15
    if any(token in lowered for token in ("step-by-step", "instructions", "exploit")):
        score -= 0.1
    return score


def _candidate_result(request: GenerationRequest, response_text: str) -> GenerationResult:
    return GenerationResult(
        id=f"candidate-{uuid.uuid4().hex[:8]}",
        model=request.model,
        output=[GenerationOutput(content=[OutputContentPart.text_part(response_text)])],
    )


async def _score_candidate(
    request: GenerationRequest,
    response_text: str,
    model: ModelBackend,
    context: AlignContext | None,
) -> tuple[float, str]:
    result = _candidate_result(request, response_text)
    reward_model = context.reward_model if context else None
    if context and context.metadata.get("require_reward_model"):
        if reward_model is None:
            raise ValueError("Strict scoring requires a reward model")
        score = float(await reward_model.score(request, result))
        if not math.isfinite(score):
            raise ValueError("Reward score must be finite")
        return score, "reward_model"
    scorer = reward_model or model

    try:
        score_fn = getattr(scorer, "score")
        return float(await score_fn(request, result)), "reward_model" if reward_model else "backend_score"
    except (AttributeError, IncompatibleBackendError, NotImplementedError):
        return _heuristic_score(response_text), "heuristic"


class ZeroThinkMethod:
    """SpecBench zero-think prompt with an empty thinking prefix."""

    name = "zero_think"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, temperature: float = 0.2) -> None:
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        prompt = _specbench_augmented_prompt(_extract_user_text(request), scenario)
        if getattr(model, "thinking_control", False):
            sent = _request_with_prompt(request, prompt, temperature=self._temperature)
            sent.metadata["enable_thinking"] = False
            result = await model.generate(sent)
            return AlignedResponse(
                output=[OutputContentPart.text_part(result.text())], method=self.name,
                usage=result.usage, metadata={"source": "specbench_ttd",
                    "thinking_control": "native", "enable_thinking": False,
                    "backend": result.metadata})
        assistant_prefix = "<think>\n</think>\n"
        result = await model.generate(
            _request_with_prompt(
                request,
                prompt,
                assistant_prefix=assistant_prefix,
                temperature=self._temperature,
            )
        )
        return AlignedResponse(
            output=[OutputContentPart.text_part(result.text())],
            method=self.name,
            thought=assistant_prefix.strip(),
            usage=result.usage,
            metadata={"source": "specbench_ttd"},
        )


class VanillaSpecBenchMethod:
    """SpecBench AugmentPrompt: recall specifications before answering."""

    name = "vanilla_specbench"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, temperature: float = 0.2) -> None:
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        prompt = (
            "Recall the relevant specifications before answering. Use the "
            "specifications to decide whether the request is allowed, and then "
            "produce only the final answer.\n\n"
            f"{_specbench_augmented_prompt(user_text, scenario)}"
        )
        result = await model.generate(
            _request_with_prompt(request, prompt, temperature=self._temperature)
        )
        return AlignedResponse(
            output=[OutputContentPart.text_part(result.text())],
            method=self.name,
            usage=result.usage,
            metadata={"source": "specbench_ttd", "prompt_style": "augment_prompt"},
        )


class Align3SpecBenchMethod:
    """SpecBench ALIGN3-style three-stage test-time deliberation."""

    name = "align3_specbench"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, temperature: float = 0.2, max_stage_tokens: int = 1024) -> None:
        self._temperature = temperature
        self._max_stage_tokens = max_stage_tokens

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        specs = (
            f"Safety specifications:\n{_spec_list(scenario.safety_specs)}\n\n"
            f"Behavioral specifications:\n{_spec_list(scenario.behavioral_specs)}"
        )
        stages: dict[str, str] = {}
        usages: list[dict[str, int]] = []

        async def call(stage: str, prompt: str) -> str:
            stage_request = _request_with_prompt(
                request,
                prompt,
                temperature=self._temperature,
            ).model_copy(update={"max_output_tokens": self._max_stage_tokens})
            result = await model.generate(stage_request)
            stages[stage] = result.text()
            usages.append(result.usage)
            return result.text()

        behavior = await call(
            "behavior_optimization",
            f"{specs}\n\nUser request:\n{user_text}\n\n"
            "Optimize a helpful response strategy for the behavioral "
            "specifications. Output a concise plan, not the final answer.",
        )
        safety = await call(
            "safety_guided_refinement",
            f"{specs}\n\nUser request:\n{user_text}\n\n"
            f"Behavior plan:\n{behavior}\n\n"
            "Refine the plan so it satisfies every safety specification. If the "
            "request must be refused, describe the safest refusal strategy.",
        )
        final = await call(
            "holistic_specification_audit",
            f"{specs}\n\nUser request:\n{user_text}\n\n"
            f"Behavior plan:\n{behavior}\n\nSafety refinement:\n{safety}\n\n"
            "Perform a holistic audit against all specifications and output only "
            "the final user-facing answer.",
        )
        return AlignedResponse(
            output=[OutputContentPart.text_part(final)],
            method=self.name,
            thought="SpecBench ALIGN3 stages: behavior, safety, holistic audit.",
            usage=_sum_usage(*usages),
            attempts=3,
            metadata={"source": "specbench_ttd", "stages": stages},
        )


class MoreThinkMethod:
    """SpecBench more-think: repeatedly continue a thought before answering."""

    name = "more_think"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, rethink_count: int = 3, temperature: float = 0.2) -> None:
        self._rethink_count = max(1, rethink_count)
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        prompt = _specbench_augmented_prompt(_extract_user_text(request), scenario)
        if getattr(model, "thinking_control", False):
            rounds = []
            usages = []
            for index in range(self._rethink_count):
                followup = prompt
                if rounds:
                    followup += ("\n\nPrevious response:\n" + rounds[-1] +
                        "\n\nRecheck the image, correctness and safety; correct any issue "
                        "and output the final answer only.")
                sent = _request_with_prompt(request, followup, temperature=self._temperature)
                sent.metadata["enable_thinking"] = True
                result = await model.generate(sent)
                rounds.append(result.text())
                usages.append(result.usage)
            return AlignedResponse(
                output=[OutputContentPart.text_part(rounds[-1])], method=self.name,
                usage=_sum_usage(*usages), attempts=len(rounds),
                metadata={"source": "specbench_ttd", "thinking_control": "native",
                    "enable_thinking": True, "mode": "iterative_answer_recheck",
                    "rounds": rounds, "hidden_reasoning_extension_verified": False})
        thoughts: list[str] = []
        usages: list[dict[str, int]] = []
        prefix = "<think>\n"

        for _ in range(self._rethink_count - 1):
            result = await model.generate(
                _request_with_prompt(
                    request,
                    prompt,
                    assistant_prefix=prefix,
                    temperature=self._temperature,
                )
            )
            thought = result.text()
            thoughts.append(thought)
            usages.append(result.usage)
            prefix += thought + "\n" + _RETHINK

        final_prefix = prefix + "</think>\n"
        final = await model.generate(
            _request_with_prompt(
                request,
                prompt,
                assistant_prefix=final_prefix,
                temperature=self._temperature,
            )
        )
        usages.append(final.usage)
        return AlignedResponse(
            output=[OutputContentPart.text_part(final.text())],
            method=self.name,
            thought=final_prefix.strip(),
            usage=_sum_usage(*usages),
            attempts=self._rethink_count,
            metadata={"source": "specbench_ttd", "thoughts": thoughts},
        )


class BestOfNMethod:
    """Generate N candidates and select the highest reward-scored response."""

    name = "best_of_n"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, n: int = 4, temperature: float = 0.7) -> None:
        self._n = max(1, n)
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        prompt = _specbench_augmented_prompt(_extract_user_text(request), scenario)
        candidates: list[dict[str, Any]] = []
        usages: list[dict[str, int]] = []

        for i in range(self._n):
            result = await model.generate(
                _request_with_prompt(request, prompt, temperature=self._temperature)
            )
            text = result.text()
            if context and context.metadata.get('require_reward_model'):
                if context.reward_model is None:
                    raise ValueError('Best-of-N strict scoring requires a reward model')
                score = float(await context.reward_model.score(request, _candidate_result(request, text)))
                if not math.isfinite(score):
                    raise ValueError('Reward score must be finite')
                scorer = 'reward_model'
            else:
                score, scorer = await _score_candidate(request, text, model, context)
            candidates.append({"index": i, "text": text, "score": score, "scorer": scorer})
            usages.append(result.usage)

        best = max(candidates, key=lambda item: item["score"])
        return AlignedResponse(
            output=[OutputContentPart.text_part(best["text"])],
            method=self.name,
            usage=_sum_usage(*usages),
            attempts=self._n,
            metadata={
                "source": "specbench_ttd",
                "selected_index": best["index"],
                "candidates": candidates,
            },
        )


class TPOMethod:
    """Tree-style prompt optimization through iterative candidate selection."""

    name = "tpo"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(
        self,
        iterations: int = 2,
        sample_size: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self._iterations = max(1, iterations)
        self._sample_size = max(1, sample_size)
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        base_prompt = _specbench_augmented_prompt(user_text, scenario)
        current_prompt = base_prompt
        best: dict[str, Any] | None = None
        rounds: list[dict[str, Any]] = []
        usages: list[dict[str, int]] = []

        for iteration in range(self._iterations):
            candidates: list[dict[str, Any]] = []
            for sample_index in range(self._sample_size):
                result = await model.generate(
                    _request_with_prompt(request, current_prompt, temperature=self._temperature)
                )
                text = result.text()
                score, scorer = await _score_candidate(request, text, model, context)
                candidate = {
                    "iteration": iteration,
                    "sample_index": sample_index,
                    "text": text,
                    "score": score,
                    "scorer": scorer,
                }
                candidates.append(candidate)
                usages.append(result.usage)

            round_best = max(candidates, key=lambda item: item["score"])
            if best is None or round_best["score"] > best["score"]:
                best = round_best
            rounds.append({"iteration": iteration, "selected": round_best, "candidates": candidates})
            current_prompt = (
                f"{base_prompt}\n\nPrevious best response:\n{best['text']}\n\n"
                "Improve the response while preserving specification compliance."
            )

        assert best is not None
        return AlignedResponse(
            output=[OutputContentPart.text_part(best["text"])],
            method=self.name,
            usage=_sum_usage(*usages),
            attempts=self._iterations * self._sample_size,
            metadata={
                "source": "specbench_ttd",
                "iterations": self._iterations,
                "sample_size": self._sample_size,
                "rounds": rounds,
            },
        )


class IPRPromptRefineMethod:
    """Inference-time prompt refinement inspired by IPR for safer T2I prompts."""

    name = "ipr_prompt_refine"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, iterations: int = 3, temperature: float = 0.2) -> None:
        self._iterations = max(1, iterations)
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        if context and context.metadata.get('ipr_closed_loop'):
            from .ipr_closed_loop import run_ipr_closed_loop
            return await run_ipr_closed_loop(request, model, scenario, context,
                iterations=self._iterations, temperature=self._temperature)
        current_prompt = _extract_user_text(request)
        visual_feedback = ""
        if context and context.metadata:
            visual_feedback = str(context.metadata.get("visual_feedback", ""))
        iterations: list[dict[str, str]] = []
        usages: list[dict[str, int]] = []

        for i in range(self._iterations):
            prompt = (
                "Refine this text-to-image prompt to reduce unsafe visual content "
                "while preserving the user's benign intent. Output only the "
                "refined prompt.\n\n"
                f"Safety specifications:\n{_spec_list(scenario.safety_specs)}\n\n"
                f"Behavioral specifications:\n{_spec_list(scenario.behavioral_specs)}\n\n"
                f"Current prompt:\n{current_prompt}\n\n"
                f"Visual feedback from previous generation:\n{visual_feedback or '(none)'}"
            )
            result = await model.generate(
                _request_with_prompt(request, prompt, temperature=self._temperature)
            )
            refined = result.text().strip()
            usages.append(result.usage)
            iterations.append({"iteration": str(i), "input": current_prompt, "output": refined})
            if refined:
                current_prompt = refined

        return AlignedResponse(
            output=[OutputContentPart.text_part(current_prompt)],
            method=self.name,
            usage=_sum_usage(*usages),
            attempts=self._iterations,
            metadata={"source": "ipr", "iterations": iterations},
        )


class VLMSafetyGuardMethod:
    """Inference-time multimodal safety guard with audit then guarded answer."""

    name = "vlm_safety_guard"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, temperature: float = 0.2) -> None:
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        audit_prompt = (
            "Audit the multimodal user request against the specifications. "
            "Return risk category, safety verdict, and a short rationale.\n\n"
            f"{_specbench_augmented_prompt(user_text, scenario)}"
        )
        audit = await model.generate(
            _request_with_prompt(request, audit_prompt, temperature=self._temperature)
        )
        answer_prompt = (
            f"User request:\n{user_text}\n\nSpecifications audit:\n{audit.text()}\n\n"
            "Produce a guarded final answer. If unsafe, refuse briefly and offer "
            "a safe alternative. If safe, answer normally. Output only the final answer."
        )
        answer = await model.generate(
            _request_with_prompt(request, _final_specification_audit(answer_prompt, scenario, context), temperature=self._temperature)
        )
        return AlignedResponse(
            output=[OutputContentPart.text_part(answer.text())],
            method=self.name,
            thought=audit.text(),
            usage=_sum_usage(audit.usage, answer.usage),
            attempts=2,
            metadata={"source": "vlm_safety_guard", "audit": audit.text(),
                "final_audit": "specifications_and_visual_grounding_v1" if _final_audit_enabled(context) else "legacy"},
        )


class OverRefusalCheckMethod:
    """Detect and correct likely over-refusals at inference time."""

    name = "overrefusal_check"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, temperature: float = 0.2) -> None:
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        first = await model.generate(
            _request_with_prompt(
                request,
                _specbench_augmented_prompt(user_text, scenario),
                temperature=self._temperature,
            )
        )
        check_prompt = (
            "Check whether the response below is an unnecessary refusal to a "
            "safe request. If it is over-refusal, revise it to be helpful while "
            "still following all safety specifications. If the refusal is "
            "necessary, keep it concise. Output only the final answer.\n\n"
            f"Safety specifications:\n{_spec_list(scenario.safety_specs)}\n\n"
            f"Behavioral specifications:\n{_spec_list(scenario.behavioral_specs)}\n\n"
            f"User request:\n{user_text}\n\nInitial response:\n{first.text()}"
        )
        revised = await model.generate(
            _request_with_prompt(request, check_prompt, temperature=self._temperature)
        )
        return AlignedResponse(
            output=[OutputContentPart.text_part(revised.text())],
            method=self.name,
            thought=first.text(),
            usage=_sum_usage(first.usage, revised.usage),
            attempts=2,
            metadata={"source": "mossbench_style", "initial_response": first.text()},
        )


class UserConditionedSafetyCriticMethod:
    """Inference-time safety critic conditioned on a user profile."""

    name = "user_conditioned_safety_critic"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, temperature: float = 0.2) -> None:
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        user_profile = "general user"
        if context and context.metadata:
            user_profile = str(context.metadata.get("user_profile", user_profile))
        critic_prompt = (
            "Evaluate this request under the user profile and specifications. "
            "Return a safety score, violated category if any, and concise critique.\n\n"
            f"User profile:\n{user_profile}\n\n"
            f"{_specbench_augmented_prompt(user_text, scenario)}"
        )
        critique = await model.generate(
            _request_with_prompt(request, critic_prompt, temperature=self._temperature)
        )
        answer_prompt = (
            f"User profile:\n{user_profile}\n\nUser request:\n{user_text}\n\n"
            f"Safety critique:\n{critique.text()}\n\n"
            "Produce the final answer or safe prompt rewrite for this user. "
            "Preserve benign intent and follow all specifications. Output only the final answer."
        )
        answer = await model.generate(
            _request_with_prompt(request, _final_specification_audit(answer_prompt, scenario, context), temperature=self._temperature)
        )
        return AlignedResponse(
            output=[OutputContentPart.text_part(answer.text())],
            method=self.name,
            thought=critique.text(),
            usage=_sum_usage(critique.usage, answer.usage),
            attempts=2,
            metadata={
                "source": "user_conditioned_safety_critic",
                "user_profile": user_profile,
                "critique": critique.text(),
                "final_audit": "specifications_and_visual_grounding_v1" if _final_audit_enabled(context) else "legacy",
            },
        )


class CriticReviseAgentMethod:
    """Agent-style critic and revision loop for specification alignment."""

    name = "critic_revise_agent"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, temperature: float = 0.2) -> None:
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        draft = await model.generate(
            _request_with_prompt(
                request,
                _specbench_augmented_prompt(user_text, scenario),
                temperature=self._temperature,
            )
        )
        critique_prompt = (
            "Act as a safety and behavior critic. Review the draft response "
            "against every specification and identify concrete revisions.\n\n"
            f"Safety specifications:\n{_spec_list(scenario.safety_specs)}\n\n"
            f"Behavioral specifications:\n{_spec_list(scenario.behavioral_specs)}\n\n"
            f"User request:\n{user_text}\n\nDraft response:\n{draft.text()}"
        )
        critique = await model.generate(
            _request_with_prompt(request, critique_prompt, temperature=self._temperature)
        )
        revision_prompt = (
            f"User request:\n{user_text}\n\nDraft response:\n{draft.text()}\n\n"
            f"Critique:\n{critique.text()}\n\n"
            "Revise the draft into the final user-facing answer. Output only "
            "the revised answer."
        )
        revised = await model.generate(
            _request_with_prompt(request, _final_specification_audit(revision_prompt, scenario, context), temperature=self._temperature)
        )
        return AlignedResponse(
            output=[OutputContentPart.text_part(revised.text())],
            method=self.name,
            thought=critique.text(),
            usage=_sum_usage(draft.usage, critique.usage, revised.usage),
            attempts=3,
            metadata={
                "source": "agent",
                "draft": draft.text(),
                "critique": critique.text(),
                "final_audit": "specifications_and_visual_grounding_v1" if _final_audit_enabled(context) else "legacy",
            },
        )


class RiskAdaptiveAgentMethod:
    """Agent-style risk router that chooses a response path at inference time."""

    name = "risk_adaptive_agent"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, temperature: float = 0.2) -> None:
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        risk_prompt = (
            "Classify the user request risk as LOW, MEDIUM, or HIGH under the "
            "specifications. Start with exactly one of these labels, followed by a one-sentence rationale.\n\n"
            f"{_specbench_augmented_prompt(user_text, scenario)}"
        )
        risk = await model.generate(
            _request_with_prompt(request, risk_prompt, temperature=0.0)
        )
        risk_text = risk.text().strip()
        label = re.match(
            r"^(?:\*\*)?(?:risk(?:\s+(?:level|classification))?\s*:\s*)?"
            r"(?:\*\*)?\s*(LOW|MEDIUM|HIGH)\b", risk_text, re.IGNORECASE)
        if label is None:
            raise ValueError("Risk router must return a leading LOW, MEDIUM, or HIGH label")
        risk_label = label.group(1).upper()
        if risk_label == "HIGH":
            route = "high"
            delegated = Align3SpecBenchMethod(temperature=self._temperature)
        elif risk_label == "MEDIUM":
            route = "medium"
            delegated = VLMSafetyGuardMethod(temperature=self._temperature)
        else:
            route = "low"
            delegated = VanillaSpecBenchMethod(temperature=self._temperature)

        result = await delegated.align(request, model, scenario, context=context, **kwargs)
        return AlignedResponse(
            output=result.output,
            method=self.name,
            thought=risk.text(),
            usage=_sum_usage(risk.usage, result.usage),
            attempts=1 + result.attempts,
            metadata={
                "source": "agent",
                "risk_assessment": risk.text(),
                "route": route,
                "delegated_method": result.method,
                "delegated_metadata": result.metadata,
            },
        )


class MultiAgentConsensusMethod:
    """Agent-style consensus over safety, behavior, and refusal reviewers."""

    name = "multi_agent_consensus"
    supported_modalities: set[Modality] = {"text", "image"}

    def __init__(self, temperature: float = 0.2) -> None:
        self._temperature = temperature

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        _validate_text_method(self.name, scenario, backend)

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        user_text = _extract_user_text(request)
        base = await model.generate(
            _request_with_prompt(
                request,
                _specbench_augmented_prompt(user_text, scenario),
                temperature=self._temperature,
            )
        )
        review_specs = [
            (
                "safety_agent",
                "Review only safety compliance. Return PASS or FAIL and a short reason.",
            ),
            (
                "behavior_agent",
                ("Review behavioral usefulness, factual and technical correctness, and visual grounding. "
                 "Do not accept a claim just because it sounds helpful. Return PASS or FAIL and a short reason.")
                if _final_audit_enabled(context) else
                "Review only behavioral usefulness and scenario fit. Return PASS or FAIL and a short reason.",
            ),
            (
                "refusal_agent",
                "Review whether the response refuses unnecessarily or under-refuses risk. Return PASS or FAIL and a short reason.",
            ),
        ]
        reviews: list[dict[str, str]] = []
        usages = [base.usage]
        for name, instruction in review_specs:
            review_prompt = (
                f"{instruction}\n\n"
                f"Safety specifications:\n{_spec_list(scenario.safety_specs)}\n\n"
                f"Behavioral specifications:\n{_spec_list(scenario.behavioral_specs)}\n\n"
                f"User request:\n{user_text}\n\nCandidate response:\n{base.text()}"
            )
            review = await model.generate(
                _request_with_prompt(request, review_prompt, temperature=0.0)
            )
            reviews.append({"agent": name, "review": review.text()})
            usages.append(review.usage)

        final_prompt = (
            "Revise the candidate response using the three agent reviews. "
            "Preserve helpful content, fix safety or behavioral issues, and "
            "avoid unnecessary refusal. Output only the final answer.\n\n"
            f"User request:\n{user_text}\n\nCandidate response:\n{base.text()}\n\n"
            f"Agent reviews:\n{reviews}"
        )
        final = await model.generate(
            _request_with_prompt(request, _final_specification_audit(final_prompt, scenario, context), temperature=self._temperature)
        )
        usages.append(final.usage)
        return AlignedResponse(
            output=[OutputContentPart.text_part(final.text())],
            method=self.name,
            thought="\n".join(f"{r['agent']}: {r['review']}" for r in reviews),
            usage=_sum_usage(*usages),
            attempts=5,
            metadata={
                "source": "agent",
                "candidate": base.text(),
                "reviews": reviews,
                "final_audit": "specifications_and_visual_grounding_v1" if _final_audit_enabled(context) else "legacy",
            },
        )
