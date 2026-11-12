"""Catalog routes: benchmarks (self-contained), methods, models, evaluators."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query

from mmalign.loaders.benchmarks import (
    BenchmarkNotFoundError,
    METADATA_FILE,
    count_samples,
    discover_packages,
    find_package,
    load_specs_from_package,
)
from mmalign.paths import user_benchmarks_dir
from mmalign_api.deps import get_configs_dir, get_workspace_dir
from mmalign_api.schemas import (
    CreateUserBenchmarkRequest,
    CreateUserBenchmarkResponse,
    EvaluatorOut,
    EvaluatorsResponse,
    MethodOut,
    MethodsResponse,
    ModelOut,
    ModelsResponse,
    ScenarioOut,
    ScenariosListResponse,
    SpecificationOut,
)
from pydantic import BaseModel as _BaseModel

class _AddSpecRequest(_BaseModel):
    category: str
    text: str
    tags: list[str] = []

router = APIRouter(tags=["catalog"])


def _pkg_to_scenario_out(pkg: Any, *, include_specs: bool = False) -> ScenarioOut:
    raw_specs = load_specs_from_package(pkg)
    specs = [
        SpecificationOut(
            spec_id=s["spec_id"],
            category=s["category"],
            text=s["text"],
            tags=s.get("tags", []),
        )
        for s in raw_specs
    ]
    return ScenarioOut(
        id=pkg.id,
        origin=pkg.origin,
        name=pkg.meta.get("name", pkg.id),
        description=pkg.meta.get("description", ""),
        modality=pkg.meta.get("modality", "text"),
        language=pkg.meta.get("language", "en"),
        safety_spec_count=sum(1 for s in specs if s.category == "safety"),
        behavioral_spec_count=sum(1 for s in specs if s.category == "behavioral"),
        sample_count=count_samples(pkg),
        source=pkg.meta.get("source", "custom"),
        specs=specs if include_specs else None,
    )


# ---------------------------------------------------------------------------
# Benchmarks / scenarios (discovered from data/benchmarks/{builtin,user}/)
# ---------------------------------------------------------------------------

@router.get("/scenarios", response_model=ScenariosListResponse)
async def list_scenarios(
    origin: Literal["builtin", "user"] | None = Query(default=None),
    workspace: Path = Depends(get_workspace_dir),
) -> ScenariosListResponse:
    packages = discover_packages(origin=origin, workspace_start=workspace)
    return ScenariosListResponse(scenarios=[_pkg_to_scenario_out(p) for p in packages])


@router.get("/scenarios/{scenario_id}", response_model=ScenarioOut)
async def get_scenario(
    scenario_id: str,
    origin: Literal["builtin", "user"] | None = Query(default=None),
    workspace: Path = Depends(get_workspace_dir),
) -> ScenarioOut:
    try:
        pkg = find_package(scenario_id, origin=origin, workspace_start=workspace)
    except BenchmarkNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "INVALID_SCENARIO", "message": str(exc)},
        ) from exc
    return _pkg_to_scenario_out(pkg, include_specs=True)


@router.post("/benchmarks/user", response_model=CreateUserBenchmarkResponse, status_code=201)
async def create_user_benchmark(
    req: CreateUserBenchmarkRequest,
    workspace: Path = Depends(get_workspace_dir),
) -> CreateUserBenchmarkResponse:
    """Register a user-uploaded self-contained benchmark (never written under builtin/)."""
    bid = req.id or f"user-{uuid.uuid4().hex[:12]}"
    root = user_benchmarks_dir(workspace) / bid
    if root.exists():
        raise HTTPException(
            status_code=409,
            detail={"code": "BENCHMARK_EXISTS", "message": f"User benchmark {bid} already exists"},
        )
    root.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "id": bid,
        "name": req.name,
        "description": req.description,
        "origin": "user",
        "source": req.source,
        "language": req.language,
        "modality": req.modality,
        "enabled": True,
    }
    (root / METADATA_FILE).write_text(
        yaml.dump(metadata, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    specs = {
        "safety_specifications": req.safety_specifications,
        "behavioral_specifications": req.behavioral_specifications,
    }
    (root / "specifications.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "prompts.json").write_text(
        json.dumps(req.prompts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return CreateUserBenchmarkResponse(id=bid)


@router.post("/scenarios/{scenario_id}/specs")
async def add_spec(
    scenario_id: str,
    req: _AddSpecRequest,
    origin: Literal["builtin", "user"] = Query(..., description="builtin or user"),
) -> dict:
    try:
        pkg = find_package(scenario_id, origin=origin)
    except BenchmarkNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "INVALID_SCENARIO", "message": str(exc)}) from exc
    if origin == "builtin":
        raise HTTPException(
            status_code=403,
            detail={"code": "READ_ONLY_BUILTIN", "message": "Cannot modify builtin benchmark specs via API"},
        )
    spec_path = pkg.root / "specifications.json"
    specs_data: dict = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else {}
    category_key = f"{req.category}_specifications"
    specs_data.setdefault(category_key, [])
    specs_data[category_key].append(req.text)
    spec_path.write_text(json.dumps(specs_data, ensure_ascii=False, indent=2), encoding="utf-8")
    spec_id = f"{req.category}_{len(specs_data[category_key]) - 1}"
    return {"spec_id": spec_id, "ok": True}


@router.delete("/scenarios/{scenario_id}/specs/{spec_id}")
async def delete_spec(
    scenario_id: str,
    spec_id: str,
    origin: Literal["builtin", "user"] = Query(...),
) -> dict:
    try:
        pkg = find_package(scenario_id, origin=origin)
    except BenchmarkNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "INVALID_SCENARIO", "message": str(exc)}) from exc
    if origin == "builtin":
        raise HTTPException(
            status_code=403,
            detail={"code": "READ_ONLY_BUILTIN", "message": "Cannot modify builtin benchmark specs via API"},
        )
    spec_path = pkg.root / "specifications.json"
    if not spec_path.exists():
        return {"ok": False, "message": "specifications.json not found"}
    specs_data: dict = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        parts = spec_id.rsplit("_", 1)
        category_key = f"{parts[0]}_specifications"
        idx = int(parts[1])
        lst = specs_data.get(category_key, [])
        if 0 <= idx < len(lst):
            lst.pop(idx)
            specs_data[category_key] = lst
            spec_path.write_text(json.dumps(specs_data, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"ok": True}
    except (IndexError, ValueError):
        pass
    return {"ok": False, "message": f"spec_id '{spec_id}' not found"}


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

_METHOD_DESCRIPTIONS = {
    "raw": "Baseline: forward request directly to the backend without any alignment.",
    "vanilla": "Spec prompting: prepend all scenario specifications to the system message.",
    "align3": "Four-stage deliberative alignment: scenario analysis → behavior planning → safety verification → synthesis.",
    "self_refine": "Iterative self-correction: generate → self-critique → revise, up to N rounds.",
}

_METHOD_MODALITIES = {
    "raw": ["text", "image", "video", "audio"],
    "vanilla": ["text", "image"],
    "align3": ["text", "image"],
    "self_refine": ["text", "image"],
}

_METHOD_DESCRIPTIONS.update({
    "vanilla_specbench": "SpecBench AugmentPrompt: recall specifications before answering.",
    "align3_specbench": "SpecBench ALIGN3: behavior optimization, safety refinement, and holistic audit.",
    "zero_think": "SpecBench TTD: use an empty thinking prefix before producing a specification-aware answer.",
    "more_think": "SpecBench TTD: continue internal thought several times, then answer under the same specifications.",
    "best_of_n": "SpecBench TTD: sample multiple candidate answers and select the highest-scored one.",
    "tpo": "SpecBench TTD: iteratively sample, score, and improve candidates across optimization rounds.",
    "ipr_prompt_refine": "IPR-style inference-time prompt refinement for safer text-to-image prompts.",
    "vlm_safety_guard": "VLM-style safety audit followed by a guarded final answer.",
    "overrefusal_check": "MOSSBench-style over-refusal detection and helpfulness-preserving correction.",
    "user_conditioned_safety_critic": "User-profile-conditioned safety critique and response correction.",
    "critic_revise_agent": "Agent workflow: draft, critique, and revise under scenario specifications.",
    "risk_adaptive_agent": "Agent workflow: assess risk and route to lightweight or stronger deliberation.",
    "multi_agent_consensus": "Agent workflow: safety, behavior, and refusal reviewers revise a candidate answer.",
    "self_reminder": (
        "Self-Reminder (Nature MI 2023): responsibility reminder in the system prompt and "
        "after the query."
    ),
    "goal_prioritization": (
        "Goal Prioritization (ACL 2024): few-shot prompt prioritising safety over "
        "helpfulness; returns the final response."
    ),
    "adashield_static": (
        "AdaShield-S (ECCV 2024): static defense prompt that checks the image and "
        "instruction for structure-based jailbreaks."
    ),
    "intention_analysis": (
        "Intention Analysis: first identify the query's essential intention, then answer "
        "in the same dialogue."
    ),
    "ecso": (
        "ECSO (ECCV 2024): self-check the answer; if harmful, caption the image and "
        "answer from the caption without the image."
    ),
    "jailguard": (
        "JailGuard: mutate the text/image input, compare response divergence, and refuse "
        "when an attack is detected."
    ),
})

_METHOD_MODALITIES.update({
    "vanilla_specbench": ["text", "image"],
    "align3_specbench": ["text", "image"],
    "zero_think": ["text", "image"],
    "more_think": ["text", "image"],
    "best_of_n": ["text", "image"],
    "tpo": ["text", "image"],
    "ipr_prompt_refine": ["text", "image"],
    "vlm_safety_guard": ["text", "image"],
    "overrefusal_check": ["text", "image"],
    "user_conditioned_safety_critic": ["text", "image"],
    "critic_revise_agent": ["text", "image"],
    "risk_adaptive_agent": ["text", "image"],
    "multi_agent_consensus": ["text", "image"],
    "self_reminder": ["text", "image"],
    "goal_prioritization": ["text", "image"],
    "adashield_static": ["text", "image"],
    "intention_analysis": ["text", "image"],
    "ecso": ["text", "image"],
    "jailguard": ["text", "image"],
})


@router.get("/methods", response_model=MethodsResponse)
async def list_methods(
    configs_dir: Path = Depends(get_configs_dir),
) -> MethodsResponse:
    raw: dict[str, Any] = yaml.safe_load(
        (configs_dir / "methods.yaml").read_text(encoding="utf-8")
    )
    methods_cfg: dict[str, Any] = raw.get("methods", {})
    out: list[MethodOut] = []
    for name, cfg in methods_cfg.items():
        out.append(MethodOut(
            name=name,
            type="inference",
            source="native",
            supported_modalities=_METHOD_MODALITIES.get(name, ["text"]),
            description=_METHOD_DESCRIPTIONS.get(name, ""),
        ))
    return MethodsResponse(methods=out)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@router.get("/models", response_model=ModelsResponse)
async def list_models(
    configs_dir: Path = Depends(get_configs_dir),
) -> ModelsResponse:
    raw: dict[str, Any] = yaml.safe_load(
        (configs_dir / "models.yaml").read_text(encoding="utf-8")
    )
    models_cfg: dict[str, Any] = raw.get("models", {})
    out: list[ModelOut] = []
    for model_id, cfg in models_cfg.items():
        caps = cfg.get("capabilities", {})
        provider = cfg.get("provider", "unknown")
        out.append(ModelOut(
            id=model_id,
            provider=provider,
            input_modalities=caps.get("input_modalities", ["text"]),
            output_modalities=caps.get("output_modalities", ["text"]),
            status="test" if provider == "mock" else "online",
        ))
    return ModelsResponse(models=out)


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

_EVALUATOR_DESCRIPTIONS: dict[str, str] = {
    "rule_based": "Deterministic keyword/pattern rules — no LLM or API key required.",
    "llm_judge": "LLM-as-Judge (joint or sequential mode) using the configured judge model.",
    "mock_judge": "Deterministic hash-based stub for automated tests only.",
}

_EVALUATOR_REQUIRES_KEY: dict[str, bool] = {
    "rule_based": False,
    "llm_judge": True,
    "mock_judge": False,
}


@router.get("/evaluators", response_model=EvaluatorsResponse)
async def list_evaluators(
    configs_dir: Path = Depends(get_configs_dir),
) -> EvaluatorsResponse:
    raw: dict[str, Any] = yaml.safe_load(
        (configs_dir / "evaluators.yaml").read_text(encoding="utf-8")
    )
    evaluators_cfg: dict[str, Any] = raw.get("evaluators", {})
    out: list[EvaluatorOut] = []
    for evaluator_id, cfg in evaluators_cfg.items():
        evaluator_type = cfg.get("type", evaluator_id)
        out.append(
            EvaluatorOut(
                id=evaluator_id,
                type=evaluator_type,
                description=_EVALUATOR_DESCRIPTIONS.get(evaluator_type, ""),
                requires_api_key=_EVALUATOR_REQUIRES_KEY.get(evaluator_type, False),
            )
        )
    return EvaluatorsResponse(evaluators=out)
