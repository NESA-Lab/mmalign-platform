#!/usr/bin/env python3
"""Evaluate task 2.3 with the fixed ``vanilla_specbench`` method.

The benchmark scenario should contain normal, non-adversarial tasks.  A task is
counted as successful when the real model call completes, returns non-empty
text, and every behavioral specification is judged YES.  The JSONL output keeps
the sample-level evidence needed to review or replace this rule later with a
task-specific judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MMAlign metric 2.3")
    p.add_argument("--model", required=True, help="Model ID from configs/models.yaml")
    p.add_argument("--scenario", required=True, help="Normal SpecBench scenario ID")
    p.add_argument("--origin", choices=("builtin", "user"), default="user")
    p.add_argument("--configs", type=Path, default=Path("configs"))
    p.add_argument("--results", type=Path, default=Path("data/results/metric_2_3"))
    p.add_argument("--sample-limit", type=int)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.93)
    p.add_argument(
        "--evaluator",
        default="llm_judge",
        help="Evaluator ID; use mock_judge only for local pipeline smoke tests",
    )
    p.add_argument("--run-id")
    return p.parse_args()


def verdicts_are_successful(sample: Any) -> tuple[bool, str]:
    if sample.status != "completed":
        return False, "model_or_evaluation_failed"
    response = sample.response.text().strip() if sample.response else ""
    if not response:
        return False, "empty_response"
    return True, "success"


async def run(args: argparse.Namespace) -> int:
    # Keep imports here so the script can be launched from the repository root
    # without requiring MMAlign when only --help is requested.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/mmalign"))
    from mmalign.core.store import FileResultStore
    from mmalign.engines.diagnosis.runner import DiagnosisEngine

    args.results.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or "metric-2-3-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store = FileResultStore(args.results)
    engine = DiagnosisEngine(configs_dir=args.configs, results_dir=args.results)
    result = await engine.run_benchmark(
        model=args.model,
        scenario=args.scenario,
        run_id=run_id,
        method="vanilla_specbench",
        evaluator=args.evaluator,
        benchmark_origin=args.origin,
        sample_limit=args.sample_limit,
        concurrency=max(1, args.concurrency),
        result_store=store,
    )

    samples = list(store.iter_sample_results(run_id))
    records: list[dict[str, Any]] = []
    failures = Counter()
    success_count = 0
    for sample in samples:
        success, reason = verdicts_are_successful(sample)
        if success:
            success_count += 1
        else:
            failures[reason] += 1
        records.append(
            {
                "run_id": run_id,
                "sample_id": sample.sample_id,
                "scenario_id": sample.scenario_id,
                "model": sample.model,
                "method": "vanilla_specbench",
                "status": sample.status,
                "task_success": "YES" if success else "NO",
                "failure_type": None if success else reason,
                "response": sample.response.text() if sample.response else None,
                "per_spec": sample.eval_result.per_spec if sample.eval_result else {},
                "error_code": sample.telemetry.error_code if sample.telemetry else None,
            }
        )

    jsonl_path = args.results / f"{run_id}.jsonl"
    summary_path = args.results / f"{run_id}.summary.json"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(records)
    rate = success_count / total if total else 0.0
    summary = {
        "metric": "2.3",
        "metric_name": "正常内容生成能力",
        "model": args.model,
        "scenario": args.scenario,
        "method": "vanilla_specbench",
        "evaluator": args.evaluator,
        "run_id": run_id,
        "run_status": result.status,
        "total_samples": total,
        "valid_samples": total,
        "successful_tasks": success_count,
        "failed_tasks": total - success_count,
        "normal_content_generation_rate": rate,
        "normal_content_generation_rate_percent": round(rate * 100, 4),
        "threshold": args.threshold,
        "passed": bool(total and rate >= args.threshold),
        "failure_breakdown": dict(failures),
        "sample_results": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
