"""Controller route: POST /api/v1/controller/generate."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from mmalign_api.deps import get_configs_dir, get_workspace_dir
from mmalign_api.schemas import (
    ControlAttemptOut,
    ControllerGenerateRequest,
    ControllerGenerateResponse,
    RiskAssessmentOut,
)

router = APIRouter(tags=["controller"])


@router.post("/controller/generate", response_model=ControllerGenerateResponse)
async def controller_generate(
    req: ControllerGenerateRequest,
    configs_dir: Path = Depends(get_configs_dir),
    workspace: Path = Depends(get_workspace_dir),
) -> ControllerGenerateResponse:
    """Run ComplianceController on a single prompt.

    Returns:
    - ``raw_text``: the direct backend response without alignment
    - ``aligned_text``: the final compliant response after correction
    - ``risk``: risk assessment for the input
    - ``strategy``: which ControlStrategy was used
    - ``attempts``: list of correction attempt summaries
    - ``latency_ms``: end-to-end latency
    """
    from mmalign.engines.correction.controller import ComplianceController
    from mmalign.core.types import GenerationRequest
    from mmalign.backends.loader import load_backends
    from mmalign.loaders.policies import load_policy_from_yaml
    from mmalign.loaders.scenarios import load_scenario

    # Load backend + scenario
    backends = load_backends(configs_dir / "models.yaml")
    backend = backends.get(req.model)
    if backend is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail={"code": "UNKNOWN_MODEL", "message": f"Model '{req.model}' not found in models.yaml"})
    scenario, _ = load_scenario(
        req.scenario,
        origin=req.benchmark_origin,
        sample_limit=0,
        workspace_start=workspace,
    )

    # Load operator-configured policy (risk thresholds + strategy mapping)
    policy = load_policy_from_yaml(req.scenario, configs_dir)

    # Build request
    gen_req = GenerationRequest(model=req.model, input=req.input)

    # Raw generation is useful for demos, but doubles model cost and latency.
    raw_text = ""
    if req.raw_compare:
        raw_result = await backend.generate(gen_req)
        raw_text = raw_result.text()

    # Run controller with the loaded policy
    controller = ComplianceController(backend=backend, scenario=scenario, policy=policy)
    controlled = await controller.generate(gen_req)

    aligned_text = controlled.response.text()
    if not aligned_text:
        # Refusal case: extract refusal text
        for out in controlled.response.output:
            for part in out.content:
                if getattr(part, "type", None) == "refusal" and part.text:
                    aligned_text = part.text
                    break

    attempts_out = [
        ControlAttemptOut(
            strategy=a.strategy,
            risk_level=a.risk_level,
            passed=a.passed,
            attempt_number=a.attempt_number,
            response_text=a.response.text()[:300],
        )
        for a in controlled.attempts
    ]

    debug_trace = None
    if req.debug:
        debug_trace = {
            "attempts": [a.model_dump() for a in controlled.attempts],
            "risk": controlled.risk.model_dump(),
        }

    return ControllerGenerateResponse(
        raw_text=raw_text,
        aligned_text=aligned_text,
        risk=RiskAssessmentOut(
            level=controlled.risk.level,
            score=controlled.risk.score,
            reason=controlled.risk.reason,
            triggered_specs=controlled.risk.triggered_specs,
        ),
        strategy=controlled.strategy,
        attempts=attempts_out,
        latency_ms=controlled.latency_ms,
        debug_trace=debug_trace,
    )
