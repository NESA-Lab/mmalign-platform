"""Acceptance checks for the tracked multimodal value-safety corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmalign.loaders.benchmarks import BenchmarkPackage, package_to_scenario

ROOT = Path(__file__).resolve().parents[4]
BUILTIN = ROOT / "examples" / "benchmarks" / "builtin"
SCENARIOS = (
    "visual_privacy",
    "visual_health",
    "visual_physical_safety",
    "visual_minor_safety",
)


def _read_json(path: Path):  # type: ignore[no-untyped-def]
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_multimodal_package_acceptance(scenario_id: str) -> None:
    root = BUILTIN / scenario_id
    specs = _read_json(root / "specifications.json")
    prompts = _read_json(root / "prompts.json")

    safety = specs["safety_specifications"]
    behavioral = specs["behavioral_specifications"]
    spec_ids = {entry["id"] for entry in safety + behavioral}

    assert len(safety) == 12
    assert len(behavioral) == 12
    assert len(spec_ids) == 24
    assert len(prompts) == 300
    assert len({entry["id"] for entry in prompts}) == 300
    assert {entry["label"] for entry in prompts} == {"safe", "unsafe"}
    assert {entry["metadata"]["split"] for entry in prompts} == {
        "train", "validation", "test"
    }

    asset_splits: dict[str, set[str]] = {}
    for entry in prompts:
        assert (root / entry["image_path"]).is_file()
        assert set(entry["expected_specs"]) <= spec_ids
        asset_id = entry["metadata"]["asset_id"]
        asset_splits.setdefault(asset_id, set()).add(entry["metadata"]["split"])
    assert all(len(splits) == 1 for splits in asset_splits.values())
    assert len(asset_splits) == 60


def test_multimodal_corpus_meets_total_targets() -> None:
    total = sum(len(_read_json(BUILTIN / sid / "prompts.json")) for sid in SCENARIOS)
    assert len(SCENARIOS) >= 2
    assert total >= 1000


def test_structured_specs_load_with_stable_ids_and_tags() -> None:
    root = BUILTIN / "visual_privacy"
    pkg = BenchmarkPackage(
        root=root,
        origin="builtin",
        meta={"id": "visual_privacy", "modality": "image"},
    )
    scenario = package_to_scenario(pkg)
    assert scenario.safety_spec_count + scenario.behavioral_spec_count == 24
    assert scenario.safety_specs[0].spec_id == "VS-S01"
    assert scenario.behavioral_specs[-1].spec_id == "VS-B12"
    assert scenario.safety_specs[0].tags == ["privacy", "pii"]
