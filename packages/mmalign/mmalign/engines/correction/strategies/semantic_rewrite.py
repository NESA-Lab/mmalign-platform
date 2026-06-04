"""SemanticRewrite strategy — generate raw response, diagnose, then rewrite.

Flow
----
1. Generate a raw response from the backend.
2. Identify which safety/behavioral specs are likely violated in the raw output.
3. Build a rewrite prompt: original request + raw response + list of violations
   + instruction to produce a compliant rewrite.
4. Call backend again with the rewrite prompt.
"""

from __future__ import annotations

import uuid

from mmalign.core.control import ControlAttempt, ControlPolicy, RiskAssessment
from mmalign.core.scenario import Scenario, Specification
from mmalign.core.types import (
    GenerationOutput,
    GenerationRequest,
    GenerationResult,
    InputMessage,
)


def _build_rewrite_prompt(
    original_text: str,
    raw_response: str,
    violated_specs: list[Specification],
    scenario_name: str,
) -> str:
    lines = [
        f"You previously answered the following request in the '{scenario_name}' scenario:",
        f"\n--- ORIGINAL REQUEST ---\n{original_text}\n",
        f"--- YOUR PREVIOUS RESPONSE ---\n{raw_response}\n",
        "--- PROBLEM ---",
        "Your previous response violates the following compliance specifications:",
    ]
    for spec in violated_specs:
        lines.append(f"  [{spec.category.upper()}] {spec.text}")
    lines += [
        "\n--- TASK ---",
        "Please rewrite your response to FULLY comply with ALL specifications above.",
        "Keep the same intent and helpfulness, but eliminate any violations.",
        "Do NOT repeat this instruction or the original response — output only the rewritten response.",
    ]
    return "\n".join(lines)


class SemanticRewriteStrategy:
    """High-risk bucket: generate raw then semantically rewrite to fix violations."""

    name = "semantic_rewrite"
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
        from mmalign.evaluators.rule_based import RuleBasedEvaluator
        assert isinstance(backend, ModelBackend)

        # Step 1: raw generation
        raw_response = await backend.generate(request)
        raw_text = raw_response.text()

        # Step 2: quick rule-based evaluation to find violated specs
        evaluator = RuleBasedEvaluator()
        eval_result = await evaluator.evaluate(request, raw_response, scenario)
        violated_specs: list[Specification] = [
            spec
            for spec in (scenario.safety_specs + scenario.behavioral_specs)
            if eval_result.per_spec.get(spec.spec_id) == "NO"
        ]

        if not violated_specs:
            # Raw response already passes — return it
            return ControlAttempt(
                strategy=self.name,
                request=request,
                response=raw_response,
                risk_level=risk.level,
                passed=True,
                metadata={"rewrite_needed": False},
            )

        # Step 3: build rewrite prompt
        original_text: str
        inp = request.input
        if isinstance(inp, str):
            original_text = inp
        else:
            original_text = " ".join(
                p.text for m in inp for p in (m.content if isinstance(m.content, list) else [])
                if getattr(p, "type", None) == "input_text" and p.text
            )

        rewrite_text = _build_rewrite_prompt(
            original_text, raw_text, violated_specs, scenario.name
        )
        rewrite_request = request.model_copy(update={"input": rewrite_text})

        # Step 4: rewrite call
        rewritten = await backend.generate(rewrite_request)
        return ControlAttempt(
            strategy=self.name,
            request=rewrite_request,
            response=rewritten,
            risk_level=risk.level,
            passed=True,
            metadata={
                "rewrite_needed": True,
                "violated_specs": [s.spec_id for s in violated_specs],
                "raw_response": raw_text,
            },
        )
