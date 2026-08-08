"""Self-contained benchmark packages under ``data/benchmarks/{builtin,user}/``.

Each package is a directory with::

    metadata.yaml          # id, name, origin, source, modality, …
    specifications.json    # SpecBench-style specs
    prompts.json           # sample list

Discovery scans filesystem — no central ``scenarios.yaml`` registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from mmalign.core.scenario import Sample, Scenario, Specification
from mmalign.paths import builtin_benchmarks_dir, user_benchmarks_dir

BenchmarkOrigin = Literal["builtin", "user"]
METADATA_FILE = "metadata.yaml"


class BenchmarkNotFoundError(KeyError):
    pass


class BenchmarkValidationError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkPackage:
    """A self-contained benchmark directory."""

    root: Path
    origin: BenchmarkOrigin
    meta: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.meta["id"])

    @property
    def enabled(self) -> bool:
        return bool(self.meta.get("enabled", True))


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BenchmarkValidationError(f"Missing {METADATA_FILE} in {path.parent}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "id" not in raw:
        raise BenchmarkValidationError(f"Invalid {METADATA_FILE}: require 'id' field")
    return raw


def _scan_origin_dir(base: Path, origin: BenchmarkOrigin) -> list[BenchmarkPackage]:
    if not base.is_dir():
        return []
    packages: list[BenchmarkPackage] = []
    for child in sorted(base.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        meta_path = child / METADATA_FILE
        if not meta_path.is_file():
            continue
        try:
            meta = _load_metadata(meta_path)
        except BenchmarkValidationError:
            continue
        if meta.get("origin") not in (None, origin):
            raise BenchmarkValidationError(
                f"{meta_path}: origin={meta.get('origin')!r} does not match folder {origin!r}"
            )
        meta = {**meta, "origin": origin}
        if not meta.get("enabled", True):
            continue
        # Directory name should match id for self-contained layout
        if meta["id"] != child.name:
            meta = {**meta, "id": meta["id"]}  # trust metadata id; root is child
        packages.append(BenchmarkPackage(root=child.resolve(), origin=origin, meta=meta))
    return packages


def discover_packages(
    *,
    origin: BenchmarkOrigin | None = None,
    workspace_start: Path | None = None,
) -> list[BenchmarkPackage]:
    """List all enabled benchmark packages, optionally filtered by origin."""
    builtin = _scan_origin_dir(builtin_benchmarks_dir(workspace_start), "builtin")
    user = _scan_origin_dir(user_benchmarks_dir(workspace_start), "user")
    if origin == "builtin":
        return builtin
    if origin == "user":
        return user
    return builtin + user


def find_package(
    benchmark_id: str,
    *,
    origin: BenchmarkOrigin | None = None,
    workspace_start: Path | None = None,
) -> BenchmarkPackage:
    matches = [
        p
        for p in discover_packages(origin=origin, workspace_start=workspace_start)
        if p.id == benchmark_id
    ]
    if not matches:
        available = [(p.origin, p.id) for p in discover_packages(workspace_start=workspace_start)]
        raise BenchmarkNotFoundError(
            f"Benchmark '{benchmark_id}' not found"
            + (f" (origin={origin})" if origin else "")
            + f". Available: {available}"
        )
    if len(matches) > 1:
        raise BenchmarkNotFoundError(
            f"Benchmark id '{benchmark_id}' is ambiguous across origins; pass origin=builtin|user"
        )
    return matches[0]


def package_to_scenario(pkg: BenchmarkPackage) -> Scenario:
    """Load Scenario from self-contained package (metadata.yaml + specifications.json)."""
    import json

    specs_path = pkg.root / "specifications.json"
    specs_json: dict[str, Any] = json.loads(specs_path.read_text(encoding="utf-8"))
    def parse_specs(
        entries: list[Any], category: Literal["safety", "behavioral"]
    ) -> list[Specification]:
        result: list[Specification] = []
        for i, entry in enumerate(entries):
            if isinstance(entry, dict):
                result.append(Specification(
                    spec_id=str(entry.get("id") or f"{category}_{i}"),
                    text=str(entry.get("text", "")),
                    category=category,
                    tags=[str(tag) for tag in entry.get("tags", [])],
                ))
            else:
                result.append(Specification(
                    spec_id=f"{category}_{i}", text=str(entry), category=category
                ))
        return result

    safety_specs = parse_specs(specs_json.get("safety_specifications", []), "safety")
    behavioral_specs = parse_specs(
        specs_json.get("behavioral_specifications", []), "behavioral"
    )
    modality = pkg.meta.get("modality", "text")
    return Scenario(
        id=pkg.id,
        name=pkg.meta.get("name", pkg.id),
        description=pkg.meta.get("description", ""),
        modality=modality,
        language=pkg.meta.get("language", "en"),
        safety_specs=safety_specs,
        behavioral_specs=behavioral_specs,
        source=pkg.meta.get("source", "custom"),
        metadata={
            "origin": pkg.origin,
            "package_root": str(pkg.root),
            "schema_version": pkg.meta.get("schema_version", 1),
        },
    )


def load_benchmark(
    benchmark_id: str,
    *,
    origin: BenchmarkOrigin | None = None,
    model: str = "placeholder",
    use_adversarial: bool = True,
    sample_limit: int | None = None,
    workspace_start: Path | None = None,
) -> tuple[Scenario, list[Sample]]:
    pkg = find_package(benchmark_id, origin=origin, workspace_start=workspace_start)
    scenario = package_to_scenario(pkg)
    samples = Sample.load_specbench_samples(
        pkg.root,
        scenario_id=scenario.id,
        model=model,
        use_adversarial=use_adversarial,
        limit=sample_limit,
    )
    return scenario, samples


def list_benchmark_descriptors(
    *,
    origin: BenchmarkOrigin | None = None,
    workspace_start: Path | None = None,
) -> list[dict[str, Any]]:
    """Summary dicts for catalog / CLI (includes origin)."""
    out: list[dict[str, Any]] = []
    for pkg in discover_packages(origin=origin, workspace_start=workspace_start):
        out.append(
            {
                "id": pkg.id,
                "origin": pkg.origin,
                "name": pkg.meta.get("name", pkg.id),
                "description": pkg.meta.get("description", ""),
                "source": pkg.meta.get("source", "custom"),
                "language": pkg.meta.get("language", "en"),
                "modality": pkg.meta.get("modality", "text"),
                "enabled": pkg.enabled,
                "root": str(pkg.root),
            }
        )
    return out


def count_samples(pkg: BenchmarkPackage) -> int:
    import json

    try:
        data = json.loads((pkg.root / "prompts.json").read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def load_specs_from_package(pkg: BenchmarkPackage) -> list[dict[str, Any]]:
    import json

    try:
        specs_json: dict[str, Any] = json.loads(
            (pkg.root / "specifications.json").read_text(encoding="utf-8")
        )
        result: list[dict[str, Any]] = []
        for category, key in (
            ("safety", "safety_specifications"),
            ("behavioral", "behavioral_specifications"),
        ):
            for i, entry in enumerate(specs_json.get(key, [])):
                if isinstance(entry, dict):
                    result.append({
                        "spec_id": str(entry.get("id") or f"{category}_{i}"),
                        "category": category,
                        "text": str(entry.get("text", "")),
                        "tags": [str(tag) for tag in entry.get("tags", [])],
                    })
                else:
                    result.append({
                        "spec_id": f"{category}_{i}",
                        "category": category,
                        "text": str(entry),
                        "tags": [],
                    })
        return result
    except Exception:
        return []
