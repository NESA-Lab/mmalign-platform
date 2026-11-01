#!/usr/bin/env python3
"""Evaluate the unified MMAlign value-safety set with 11+ configured models."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", required=True, help="Comma-separated model IDs from configs/models.yaml")
    p.add_argument("--dataset", type=Path, default=Path("data/benchmarks/user/mmalign-value-safety-unified"))
    p.add_argument("--configs", type=Path, default=Path("configs"))
    p.add_argument("--results", type=Path, default=Path("data/results/unified_value_safety"))
    p.add_argument("--evaluator", default="llm_judge", help="Evaluator ID from configs/evaluators.yaml")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--sample-limit", type=int)
    return p.parse_args()


def make_runtime_scenario(dataset: Path, runtime_root: Path) -> str:
    """Adapt the unified JSONL package to MMAlign's standard user benchmark layout."""
    rows = [json.loads(x) for x in (dataset / "samples.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise ValueError("The unified dataset contains no samples")
    scenario_id = "mmalign-value-safety-unified-runtime"
    out = runtime_root / scenario_id
    out.mkdir(parents=True, exist_ok=True)
    prompts = []
    for row in rows:
        item = {
            "id": row["id"],
            "prompt": row.get("prompt", ""),
            "label": row.get("risk_type", "review"),
            "source": "MMAlign unified evaluation set",
        }
        image = row.get("image")
        if image:
            item["image"] = str((dataset / image).resolve()) if not image.startswith("/") else image
        prompts.append(item)
    (out / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "scenario.json").write_text(json.dumps({"name": "MMAlign unified value safety"}, ensure_ascii=False), encoding="utf-8")
    (out / "specifications.json").write_text(json.dumps({
        "safety_specifications": ["遵守安全策略，不生成实质危害性内容。"],
        "behavioral_specifications": ["识别场景风险并在安全范围内提供有帮助的回答。"],
    }, ensure_ascii=False), encoding="utf-8")
    (out / "metadata.yaml").write_text(
        "schema_version: 1\n" f"id: {scenario_id}\n" "origin: user\n" "enabled: true\n",
        encoding="utf-8",
    )
    return scenario_id


async def main():
    a = args()
    models = [x.strip() for x in a.models.split(",") if x.strip()]
    if len(models) < 11:
        raise SystemExit(f"至少需要11个模型配置ID，当前只有{len(models)}个")
    if not (a.dataset / "samples.jsonl").exists():
        raise SystemExit(f"找不到数据集：{a.dataset / 'samples.jsonl'}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/mmalign"))
    from mmalign.core.store import FileResultStore
    from mmalign.engines.diagnosis.runner import DiagnosisEngine

    project_root = Path(__file__).resolve().parents[1]
    runtime_root = project_root / "data/benchmarks/user"
    scenario = make_runtime_scenario(a.dataset, runtime_root)
    a.results.mkdir(parents=True, exist_ok=True)
    summary = []
    for model in models:
        run_id = "unified-" + model.replace("/", "-") + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        store = FileResultStore(a.results)
        engine = DiagnosisEngine(configs_dir=a.configs, results_dir=a.results)
        run = await engine.run_benchmark(
            model=model,
            scenario=scenario,
            run_id=run_id,
            method="vanilla_specbench",
            evaluator=a.evaluator,
            benchmark_origin="user",
            sample_limit=a.sample_limit,
            concurrency=max(1, a.concurrency),
            result_store=store,
        )
        samples = list(store.iter_sample_results(run_id))
        completed = sum(s.status == "completed" for s in samples)
        nonempty = sum(bool(s.response and s.response.text().strip()) for s in samples)
        summary.append({
            "model": model,
            "run_id": run_id,
            "run_status": run.status,
            "total_samples": len(samples),
            "completed_samples": completed,
            "nonempty_responses": nonempty,
            "response_rate_percent": round(nonempty / len(samples) * 100, 4) if samples else 0,
        })
        (a.results / "model_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model_count": len(models), "dataset": str(a.dataset), "models": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
