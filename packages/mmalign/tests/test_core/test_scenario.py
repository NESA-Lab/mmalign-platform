"""Tests for mmalign.core.scenario — Scenario, Specification, Sample."""

import json
import tempfile
from pathlib import Path

import pytest
from mmalign.core.scenario import Sample, Scenario, Specification, SpecCategory
from mmalign.core.types import GenerationRequest


SPECBENCH_CHILD_PATH = Path(
    "/Users/helium7/projects/project-mm-align/refs/repos/SpecBench/data/Child"
)


class TestSpecification:
    def test_basic_fields(self):
        spec = Specification(
            spec_id="safety_0",
            text="Do not use hate speech",
            category="safety",
        )
        assert spec.category == "safety"
        assert spec.tags == []

    def test_serialisation(self):
        spec = Specification(spec_id="b_0", text="Be helpful", category="behavioral")
        data = spec.model_dump()
        assert data["spec_id"] == "b_0"
        assert Specification.model_validate(data) == spec


class TestScenario:
    def _make_scenario(self) -> Scenario:
        return Scenario(
            id="test",
            name="Test",
            description="A test scenario",
            modality="text",
            safety_specs=[
                Specification(spec_id="safety_0", text="Rule A", category="safety"),
            ],
            behavioral_specs=[
                Specification(spec_id="behavioral_0", text="Rule B", category="behavioral"),
            ],
        )

    def test_all_specs(self):
        s = self._make_scenario()
        assert len(s.all_specs) == 2

    def test_spec_counts(self):
        s = self._make_scenario()
        assert s.safety_spec_count == 1
        assert s.behavioral_spec_count == 1

    def test_get_spec_found(self):
        s = self._make_scenario()
        spec = s.get_spec("safety_0")
        assert spec is not None
        assert spec.text == "Rule A"

    def test_get_spec_missing(self):
        s = self._make_scenario()
        assert s.get_spec("nonexistent") is None

    def test_specification_dict(self):
        s = self._make_scenario()
        d = s.specification_dict()
        assert "safety_specifications" in d
        assert "behavioral_specifications" in d
        assert d["safety_specifications"] == ["Rule A"]

    def test_serialisation_round_trip(self):
        s = self._make_scenario()
        restored = Scenario.model_validate(s.model_dump())
        assert restored.id == s.id
        assert len(restored.safety_specs) == 1


class TestScenarioFromSpecbench:
    """Integration test: loads real SpecBench Child scenario data."""

    @pytest.mark.skipif(
        not SPECBENCH_CHILD_PATH.exists(),
        reason="SpecBench data not available at expected path",
    )
    def test_from_specbench_child(self):
        scenario = Scenario.from_specbench(SPECBENCH_CHILD_PATH)
        assert scenario.id == "child"
        assert scenario.source == "specbench"
        assert scenario.safety_spec_count > 0
        assert scenario.behavioral_spec_count > 0
        # All spec IDs follow expected pattern
        for i, spec in enumerate(scenario.safety_specs):
            assert spec.spec_id == f"safety_{i}"
            assert spec.category == "safety"
        for i, spec in enumerate(scenario.behavioral_specs):
            assert spec.spec_id == f"behavioral_{i}"
            assert spec.category == "behavioral"

    def test_from_specbench_with_temp_dir(self):
        """Unit test with synthetic SpecBench-format files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "scenario.json").write_text(
                json.dumps({"name": "Demo Scenario", "description": "Test desc"}),
                encoding="utf-8",
            )
            (base / "specifications.json").write_text(
                json.dumps({
                    "safety_specifications": ["No violence", "No hate speech"],
                    "behavioral_specifications": ["Be concise"],
                }),
                encoding="utf-8",
            )
            (base / "prompts.json").write_text(json.dumps([]), encoding="utf-8")

            scenario = Scenario.from_specbench(base)

        assert scenario.name == "Demo Scenario"
        assert scenario.source == "specbench"
        assert scenario.safety_spec_count == 2
        assert scenario.behavioral_spec_count == 1
        assert scenario.safety_specs[0].spec_id == "safety_0"
        assert scenario.safety_specs[1].text == "No hate speech"


class TestSample:
    def test_from_specbench_entry_uses_adversarial_by_default(self):
        entry = {
            "prompt": "original",
            "adversarial_prompt": "adversarial version",
            "label": "unsafe",
            "source": "GPT",
            "answer": "",
        }
        sample = Sample.from_specbench_entry(entry, scenario_id="child")
        # Should use adversarial prompt by default
        msgs = sample.request.input_messages()
        assert msgs[0].content[0].text == "adversarial version"  # type: ignore[index]
        assert sample.adversarial_prompt == "adversarial version"
        assert sample.label == "unsafe"

    def test_from_specbench_entry_safe_label(self):
        entry = {"prompt": "Tell me about stars", "label": "safe", "source": "manual"}
        sample = Sample.from_specbench_entry(entry, scenario_id="travel")
        assert sample.label == "safe"

    def test_from_specbench_entry_no_adversarial(self):
        entry = {"prompt": "original", "label": "unsafe"}
        sample = Sample.from_specbench_entry(entry, scenario_id="test")
        msgs = sample.request.input_messages()
        assert msgs[0].content[0].text == "original"  # type: ignore[index]

    @pytest.mark.skipif(
        not SPECBENCH_CHILD_PATH.exists(),
        reason="SpecBench data not available",
    )
    def test_load_specbench_samples_limit(self):
        samples = Sample.load_specbench_samples(
            SPECBENCH_CHILD_PATH, scenario_id="child", limit=5
        )
        assert len(samples) == 5
        assert all(s.scenario_id == "child" for s in samples)
