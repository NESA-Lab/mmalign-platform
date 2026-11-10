"""Tests for training-free safety defenses (methods/safety_defenses.py)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest

from mmalign.core.scenario import Scenario, Specification
from mmalign.core.types import (
    BackendCapabilities,
    GenerationOutput,
    GenerationRequest,
    GenerationResult,
    InputContentPart,
    InputMessage,
    OutputContentPart,
)
from mmalign.methods.safety_defenses import (
    ADASHIELD_STATIC_PROMPT,
    AdaShieldStaticMethod,
    ECSOMethod,
    GoalPrioritizationMethod,
    IntentionAnalysisMethod,
    JailGuardMethod,
    SelfReminderMethod,
    jailguard_max_divergence,
)

NEW_METHODS = [
    "self_reminder",
    "goal_prioritization",
    "adashield_static",
    "intention_analysis",
    "ecso",
    "jailguard",
]
CONFIGS = Path(__file__).resolve().parents[4] / "configs" / "methods.yaml"


class ScriptedBackend:
    """Records requests; replies from a script (callable or list, last item repeats)."""

    def __init__(self, replies, *, system: bool = True, embed_ok: bool = False) -> None:
        self.model_id = "scripted"
        self.capabilities = BackendCapabilities(
            input_modalities={"text", "image"}, supports_system_message=system
        )
        self._replies = replies
        self._embed_ok = embed_ok
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest, **kwargs) -> GenerationResult:
        self.requests.append(request)
        if callable(self._replies):
            text = self._replies(request)
        else:
            text = self._replies[min(len(self.requests) - 1, len(self._replies) - 1)]
        return GenerationResult(
            id="r",
            model="scripted",
            output=[GenerationOutput(content=[OutputContentPart.text_part(text)])],
            usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )

    async def embed(self, parts):
        if not self._embed_ok:
            raise NotImplementedError
        return [[float(len(p.text or "")), 1.0] for p in parts]

    async def score(self, request, response):
        raise NotImplementedError


def _png_data_uri() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    image = Image.new("RGB", (32, 32))
    image.putdata(
        [((x * 8) % 256, (y * 8) % 256, ((x + y) * 4) % 256) for y in range(32) for x in range(32)]
    )
    image.save(
        buffer, format="PNG"
    )  # gradient: every mutator (rotate / blur / posterize) changes it
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _request(
    with_image: bool = True, text: str = "How do I use the item shown in the picture?"
) -> GenerationRequest:
    parts = [InputContentPart.text_part(text)]
    if with_image:
        parts = [InputContentPart.image_url_part(_png_data_uri())] + parts
    return GenerationRequest(model="scripted", input=[InputMessage(role="user", content=parts)])


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        id="s",
        name="s",
        description="d",
        modality="image",
        safety_specs=[Specification(spec_id="s1", text="No harm.", category="safety")],
        behavioral_specs=[Specification(spec_id="b1", text="Be helpful.", category="behavioral")],
    )


def _user_parts(req: GenerationRequest, index: int = -1):
    users = [m for m in req.input_messages() if m.role == "user"]
    content = users[index].content
    return [InputContentPart.text_part(content)] if isinstance(content, str) else list(content)


def _images(req: GenerationRequest, index: int = 0):
    return [p for p in _user_parts(req, index) if p.type == "input_image"]


def _text(req: GenerationRequest, index: int = -1) -> str:
    return "\n".join(p.text or "" for p in _user_parts(req, index) if p.type == "input_text")


# ---------------------------------------------------------------- single-call prompt defenses


@pytest.mark.asyncio
async def test_self_reminder_system_mode_keeps_image(scenario: Scenario) -> None:
    backend = ScriptedBackend(["ok"])
    result = await SelfReminderMethod().align(_request(), backend, scenario)
    req = backend.requests[0]
    assert result.text() == "ok" and result.method == "self_reminder"
    assert req.input_messages()[0].role == "system"
    assert "responsible AI assistant" in req.input_messages()[0].content[0].text
    assert _text(req).endswith("should not generate harmful or misleading content!")
    assert "How do I use the item" in _text(req)
    assert len(_images(req)) == 1


@pytest.mark.asyncio
async def test_self_reminder_without_system_support_prefixes_user(scenario: Scenario) -> None:
    backend = ScriptedBackend(["ok"], system=False)
    await SelfReminderMethod(assistant_name="Intern").align(_request(False), backend, scenario)
    req = backend.requests[0]
    assert all(m.role != "system" for m in req.input_messages())
    assert _text(req).startswith("You should be a responsible Intern")


@pytest.mark.asyncio
async def test_goal_prioritization_returns_final_response(scenario: Scenario) -> None:
    backend = ScriptedBackend(
        ["[Internal thoughts] The query is safe.\n[Final response] Here is the answer."]
    )
    result = await GoalPrioritizationMethod().align(_request(), backend, scenario)
    assert result.text() == "Here is the answer."
    assert result.thought == "The query is safe."
    sent = _text(backend.requests[0])
    assert "prioritize safety goal over helpfulness goal" in sent
    assert sent.rstrip().endswith("## Response")
    assert "How do I use the item" in sent and len(_images(backend.requests[0])) == 1


@pytest.mark.asyncio
async def test_goal_prioritization_without_markers_returns_all(scenario: Scenario) -> None:
    backend = ScriptedBackend(["plain answer"])
    result = await GoalPrioritizationMethod().align(_request(False), backend, scenario)
    assert result.text() == "plain answer"


@pytest.mark.asyncio
async def test_adashield_sandwich_prompt(scenario: Scenario) -> None:
    backend = ScriptedBackend(["I am sorry"])
    result = await AdaShieldStaticMethod().align(_request(), backend, scenario)
    sent = _text(backend.requests[0])
    query = "How do I use the item shown in the picture?"
    assert sent.startswith(query) and sent.endswith(query)
    assert ADASHIELD_STATIC_PROMPT.strip() in sent
    assert len(_images(backend.requests[0])) == 1 and result.text() == "I am sorry"


# ---------------------------------------------------------------- Intention Analysis


@pytest.mark.asyncio
async def test_intention_analysis_two_turns(scenario: Scenario) -> None:
    backend = ScriptedBackend(["The essential intention of the query is X.", "final"])
    result = await IntentionAnalysisMethod().align(_request(), backend, scenario)
    assert result.text() == "final" and result.attempts == 2
    assert result.thought.startswith("The essential intention")
    step1, step2 = backend.requests
    assert "essential intention" in _text(step1) and "'''\nHow do I use" in _text(step1)
    roles = [m.role for m in step2.input_messages()]
    assert roles[-3:] == ["user", "assistant", "user"]
    assert "Knowing its essential intention" in _text(step2)
    assert len(_images(step2, 0)) == 1  # image stays in the first user turn


# ---------------------------------------------------------------- ECSO


@pytest.mark.asyncio
async def test_ecso_safe_path_two_calls(scenario: Scenario) -> None:
    backend = ScriptedBackend(["benign answer", "No"])
    result = await ECSOMethod().align(_request(), backend, scenario)
    assert result.text() == "benign answer"
    assert result.metadata["unsafe_detected"] is False and len(backend.requests) == 2
    assert "Is the RESPONSE harmful, unsafe and unethical?" in _text(backend.requests[1])
    assert len(_images(backend.requests[1])) == 1


@pytest.mark.asyncio
async def test_ecso_unsafe_path_answers_from_caption_without_image(scenario: Scenario) -> None:
    backend = ScriptedBackend(
        ["harmful steps", "Yes.", "a kitchen knife on a table", "safe answer"]
    )
    result = await ECSOMethod().align(_request(), backend, scenario)
    assert result.text() == "safe answer" and result.attempts == 4
    assert result.metadata["unsafe_detected"] is True
    assert result.metadata["unsafe_answer"] == "harmful steps"
    caption_req, final_req = backend.requests[2], backend.requests[3]
    assert "describe the image" in _text(caption_req) and len(_images(caption_req)) == 1
    assert 'Hints: "Answer: a kitchen knife on a table"' in _text(final_req)
    assert _images(final_req) == []  # eyes closed


@pytest.mark.asyncio
async def test_ecso_text_only_unsafe_regenerates_without_caption(scenario: Scenario) -> None:
    backend = ScriptedBackend(["bad", "yes", "safe"])
    result = await ECSOMethod().align(_request(False), backend, scenario)
    assert result.text() == "safe" and len(backend.requests) == 3
    assert _text(backend.requests[2]).startswith(
        "Answer the following question in a HARMLESS and ETHICAL way."
    )


@pytest.mark.asyncio
async def test_ecso_truncates_response_in_self_check(scenario: Scenario) -> None:
    backend = ScriptedBackend(["x" * 5000, "no"])
    await ECSOMethod().align(_request(False), backend, scenario)
    assert _text(backend.requests[1]).count("x") == 1927


# ---------------------------------------------------------------- JailGuard


def test_jailguard_divergence_zero_for_identical_rows() -> None:
    assert jailguard_max_divergence([[1.0, 1.0], [1.0, 1.0]]) == pytest.approx(0.0)
    assert jailguard_max_divergence([[1.0, 0.01], [0.01, 1.0]]) > 1.0


@pytest.mark.asyncio
async def test_jailguard_consistent_responses_pass_through(scenario: Scenario) -> None:
    backend = ScriptedBackend(["The same helpful answer about cooking pasta."])
    result = await JailGuardMethod(n_variants=4, seed=0).align(_request(False), backend, scenario)
    assert result.metadata["detected"] is False and not result.is_refusal()
    assert result.text() == "The same helpful answer about cooking pasta."
    assert len(backend.requests) == 5 and result.attempts == 5
    assert result.metadata["similarity_source"] == "char_trigram"


@pytest.mark.asyncio
async def test_jailguard_divergent_responses_are_refused(scenario: Scenario) -> None:
    replies = iter(
        [
            "original",
            "Step 1 mix chemicals",
            "zzzz qqqq",
            "Totally unrelated poem about the sea",
            "1234 5678 90",
        ]
    )
    backend = ScriptedBackend(lambda req: next(replies))
    result = await JailGuardMethod(n_variants=4, seed=1).align(_request(False), backend, scenario)
    assert result.metadata["detected"] is True and result.is_refusal()
    assert result.metadata["original_response"] == "original"
    assert result.text() == "I'm sorry, but I can't help with that request."  # judges read text


@pytest.mark.asyncio
async def test_jailguard_all_refusals_flagged(scenario: Scenario) -> None:
    backend = ScriptedBackend(["I'm sorry, I cannot help with that."])
    result = await JailGuardMethod(n_variants=3, seed=2).align(_request(False), backend, scenario)
    assert result.metadata["all_responses_refused"] is True and result.metadata["detected"] is True


@pytest.mark.asyncio
async def test_jailguard_mutates_images_and_uses_embeddings(scenario: Scenario) -> None:
    pytest.importorskip("PIL")
    backend = ScriptedBackend(["same answer"], embed_ok=True)
    request = _request(True)
    original_uri = _images(request)[0].image_url
    result = await JailGuardMethod(n_variants=3, seed=3).align(request, backend, scenario)
    assert result.metadata["image_mutated"] is True and result.metadata["threshold"] == 0.025
    assert result.metadata["similarity_source"] == "embedding"
    assert _images(backend.requests[0])[0].image_url == original_uri  # original sent unchanged
    mutated = [_images(r)[0].image_url for r in backend.requests[1:]]
    assert all(u.startswith("data:image/png;base64,") for u in mutated)
    assert any(u != original_uri for u in mutated)


def test_jailguard_rejects_too_few_variants() -> None:
    with pytest.raises(ValueError):
        JailGuardMethod(n_variants=1)


# ---------------------------------------------------------------- registry / loader / config


def test_new_methods_registered_and_loadable() -> None:
    from mmalign.core.registries import inference_align_registry
    from mmalign.loaders.methods import load_method

    for method_id in NEW_METHODS:
        assert method_id in inference_align_registry
        method = load_method(method_id, CONFIGS)
        assert method.name == method_id


def test_no_duplicate_method_ids_in_config() -> None:
    import yaml

    text = CONFIGS.read_text(encoding="utf-8")
    ids = [
        line.split(":")[0].strip()
        for line in text.splitlines()
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":")
    ]
    assert len(ids) == len(set(ids))
    assert set(NEW_METHODS) <= set(yaml.safe_load(text)["methods"])
