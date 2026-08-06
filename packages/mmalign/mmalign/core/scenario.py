"""Scenario, Specification, and Sample data models.

A ``Scenario`` is the fundamental evaluation unit.  Every benchmark run,
alignment call, and evaluation result is anchored to a scenario.

SpecBench compatibility
-----------------------
SpecBench stores its data as three JSON files per scenario directory:

- ``scenario.json``      → ``Scenario.id`` / ``name`` / ``description``
- ``specifications.json``→ ``Scenario.safety_specs`` + ``behavioral_specs``
- ``prompts.json``       → ``list[Sample]``

``Scenario.from_specbench(path)`` loads all three files from a scenario
directory, maintaining full round-trip fidelity with SpecBench data.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .types import GenerationRequest, InputContentPart, InputMessage, ReviewVerdict

SpecCategory = Literal["safety", "behavioral"]


class Specification(BaseModel):
    """A single alignment specification (safety or behavioral rule).

    ``spec_id`` is assigned automatically when loading from SpecBench
    (e.g. ``safety_0``, ``behavioral_2``).
    """

    spec_id: str
    text: str
    category: SpecCategory
    tags: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    """An evaluation scenario with its associated specifications.

    ``source`` distinguishes native Chinese-value scenarios from SpecBench
    originals (``"specbench"``) or other external benchmarks.
    """

    id: str
    name: str
    description: str
    modality: Literal["text", "image", "video", "multimodal"]
    language: str = "en"
    safety_specs: list[Specification] = Field(default_factory=list)
    behavioral_specs: list[Specification] = Field(default_factory=list)
    source: str = "native"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def all_specs(self) -> list[Specification]:
        return self.safety_specs + self.behavioral_specs

    @property
    def safety_spec_count(self) -> int:
        return len(self.safety_specs)

    @property
    def behavioral_spec_count(self) -> int:
        return len(self.behavioral_specs)

    def get_spec(self, spec_id: str) -> Specification | None:
        for spec in self.all_specs:
            if spec.spec_id == spec_id:
                return spec
        return None

    def specification_dict(self) -> dict[str, list[str]]:
        """Return SpecBench-style specification dict for TTD adapter use."""
        return {
            "safety_specifications": [s.text for s in self.safety_specs],
            "behavioral_specifications": [s.text for s in self.behavioral_specs],
        }

    @classmethod
    def from_specbench(cls, path: str | Path) -> Scenario:
        """Load a scenario from a SpecBench data directory.

        Expected directory layout::

            <path>/
              scenario.json        {"name": "...", "description": "..."}
              specifications.json  {"safety_specifications": [...],
                                    "behavioral_specifications": [...]}
              prompts.json         [{...}, ...]   (used separately via Sample.from_specbench)

        The ``id`` is derived from the directory name.
        """
        base = Path(path)
        scenario_json = json.loads((base / "scenario.json").read_text(encoding="utf-8"))
        specs_json = json.loads((base / "specifications.json").read_text(encoding="utf-8"))

        safety_specs = [
            Specification(
                spec_id=f"safety_{i}",
                text=text,
                category="safety",
            )
            for i, text in enumerate(specs_json.get("safety_specifications", []))
        ]
        behavioral_specs = [
            Specification(
                spec_id=f"behavioral_{i}",
                text=text,
                category="behavioral",
            )
            for i, text in enumerate(specs_json.get("behavioral_specifications", []))
        ]

        modality: Literal["text", "image", "video", "multimodal"] = (
            "image" if "vlm" in str(base).lower() else "text"
        )

        return cls(
            id=base.name.lower(),
            name=scenario_json.get("name", base.name),
            description=scenario_json.get("description", ""),
            modality=modality,
            language="en",
            safety_specs=safety_specs,
            behavioral_specs=behavioral_specs,
            source="specbench",
        )


class SpecLabel(BaseModel):
    """Per-specification verdict from a reviewer or evaluator."""

    spec_id: str
    verdict: ReviewVerdict
    confidence: float | None = None
    note: str | None = None


class Sample(BaseModel):
    """A single evaluation sample (prompt + ground-truth metadata).

    ``request`` carries the full OpenAI-compatible prompt so that backends
    can consume it without re-wrapping.

    SpecBench compatibility
    -----------------------
    ``adversarial_prompt`` holds the WildTeaming-augmented prompt variant.
    ``label`` is SpecBench's ``"safe"`` / ``"unsafe"`` classification.
    ``safety_warning`` is the VLM-scenario multimodal safety context string.
    """

    id: str
    request: GenerationRequest
    scenario_id: str
    label: Literal["safe", "unsafe"] = "unsafe"
    adversarial_prompt: str | None = None
    answer: str | None = None
    safety_warning: str | None = None
    expected_specs: list[str] = Field(default_factory=list)
    labels: list[SpecLabel] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_specbench_entry(
        cls,
        entry: dict[str, Any],
        scenario_id: str,
        model: str = "placeholder",
        sample_id: str | None = None,
        *,
        use_adversarial: bool = True,
    ) -> Sample:
        """Construct a Sample from a single SpecBench prompts.json entry.

        By default uses the adversarial variant of the prompt when available,
        mirroring SpecBench's evaluation behaviour.
        """
        raw_prompt: str = entry.get("prompt", "")
        adversarial: str | None = entry.get("adversarial_prompt") or None
        effective_prompt = adversarial if (use_adversarial and adversarial) else raw_prompt

        sid = str(
            entry.get("id") or sample_id or f"{scenario_id}_{hash(effective_prompt) & 0xFFFF:04x}"
        )
        content = [InputContentPart.text_part(effective_prompt)]
        image_url = entry.get("image_url")
        if image_url:
            content.append(InputContentPart.image_url_part(str(image_url)))
        request = GenerationRequest(
            model=model,
            input=[
                InputMessage(
                    role="user",
                    content=content,
                )
            ],
        )
        return cls(
            id=sid,
            request=request,
            scenario_id=scenario_id,
            label=entry.get("label", "unsafe"),
            adversarial_prompt=adversarial,
            answer=entry.get("answer") or None,
            safety_warning=entry.get("safety_warning") or None,
            expected_specs=list(entry.get("expected_specs", [])),
            metadata={
                "source": entry.get("source", ""),
                "modality": entry.get("modality", "image" if image_url else "text"),
                "pair_id": entry.get("pair_id"),
                "variant": entry.get("variant"),
                **entry.get("metadata", {}),
            },
        )

    @classmethod
    def load_specbench_samples(
        cls,
        path: str | Path,
        scenario_id: str,
        model: str = "placeholder",
        *,
        use_adversarial: bool = True,
        limit: int | None = None,
    ) -> list[Sample]:
        """Load all samples from a SpecBench prompts.json file."""
        base = Path(path)
        data: list[dict[str, Any]] = json.loads((base / "prompts.json").read_text(encoding="utf-8"))
        if limit is not None:
            data = data[:limit]
        normalized: list[dict[str, Any]] = []
        for entry in data:
            item = dict(entry)
            image_path = item.pop("image_path", None)
            if image_path and not item.get("image_url"):
                asset = (base / str(image_path)).resolve()
                if not asset.is_relative_to(base.resolve()):
                    raise ValueError(f"image_path escapes benchmark package: {image_path}")
                mime = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
                item["image_url"] = f"data:{mime};base64,{encoded}"
            normalized.append(item)
        return [
            cls.from_specbench_entry(
                entry,
                scenario_id=scenario_id,
                model=model,
                sample_id=str(entry.get("id") or f"{scenario_id}_{i}"),
                use_adversarial=use_adversarial,
            )
            for i, entry in enumerate(normalized)
        ]
