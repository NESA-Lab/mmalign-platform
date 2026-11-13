"""Spec-aware evaluation of mmalign methods on Intern-S2 (test3 adaptation overlay).

Differences from eval_methods.py (everything else reused: package judge prompt, parsing, metrics):
- each sample's method call receives the Scenario built from the package's own criteria for that
  sample (specs_for) plus the scenario description, as in the SpecBench protocol (methods may see
  the scenario specification; the judge scores against the same specification; `raw` ignores it);
- explicit sample-id lists, so tuning happens on a dev split and the final comparison on a
  disjoint test split;
- align3 never receives previous judge results (no AlignContext.vanilla_results_path), so no
  test-label information reaches any method.

Run with PYTHONPATH pointing at the test3 overlay of packages/mmalign, e.g. (repo root):

  cp -R packages /tmp/ov && cp -R test/test3/source/mmalign/* /tmp/ov/packages/mmalign/mmalign/
  INTERN_API_KEY=... OPENAI_API_KEY=... PYTHONPATH=/tmp/ov/packages/mmalign \
  python test/test3/scripts/eval_spec_aware.py --split test --methods raw,align3 \
    --configs <dir with models.yaml: intern-s2-preview-35b + gpt-5.6-luna> \
    --methods-config test/test3/configs/methods.yaml \
    --ids-dir test/test3/results/intern_method_uplift --results /tmp/uplift
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from mmalign.core.scenario import Scenario, Specification
from mmalign.loaders.methods import load_method

sys.path.insert(0, str(Path(__file__).resolve().parent))  # eval_methods_common
import eval_methods_common as base  # noqa: E402

CURRENT = contextvars.ContextVar("current_sample")


def scenario_for(pkg: str, mod, sample: dict, criteria: list[dict], cache: dict) -> Scenario:
    specs = mod.specs_for(sample, criteria)
    key = tuple(c["id"] for c in specs)
    if key not in cache:
        if pkg == "spec_alignment":
            desc = mod.SCENARIOS.get(sample["scenario"], sample["scenario"])
        else:
            desc = f"{mod.SUBSET_NAMES.get(sample['subset'], sample['subset'])}" + (
                f" / {sample['scenario']}" if sample["subset"] == "scenario" else ""
            )
        is_scene = sample.get("subset", "scenario") == "scenario"
        scene = sample["scenario"] if is_scene else "*"
        cache[key] = Scenario(
            id=f"{pkg}:{sample.get('subset', '')}:{scene}",
            name=desc,
            description=desc,
            modality="multimodal",
            safety_specs=[
                Specification(spec_id=c["id"], text=c["text"], category="safety")
                for c in specs
                if c["category"] == "safety"
            ],
            behavioral_specs=[
                Specification(spec_id=c["id"], text=c["text"], category="behavioral")
                for c in specs
                if c["category"] == "behavioral"
            ],
        )
    return cache[key]


class ThinkingClientBackend(base.ClientBackend):
    """ClientBackend that honours request.metadata["enable_thinking"] like test3's InternBackend.

    Two package Clients are kept: the default config (thinking_mode false, as in the package
    baselines) and a copy with thinking_mode true and a larger output budget (16384, the
    InternBackend minimum) so reasoning does not truncate the final answer.
    """

    thinking_control = True

    def __init__(self, client_off, client_on, model_id: str) -> None:
        super().__init__(client_off, model_id)
        self.client_on = client_on

    async def generate(self, request, **kwargs):
        if request.metadata.get("enable_thinking") is True:
            off, self.client = self.client, self.client_on
            try:
                return await super().generate(request, **kwargs)
            finally:
                self.client = off
        return await super().generate(request, **kwargs)


class SpecAwareMethodClient(base.MethodClient):
    async def chat(self, messages, *, temperature=None, max_tokens=None) -> dict:
        self.scenario = CURRENT.get()
        return await super().chat(messages, temperature=temperature, max_tokens=max_tokens)


async def run_method(pkg, mod, samples, criteria, method_id, judge, client_factory, args, out_root):
    out_dir = out_root / pkg / method_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "responses.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["status"] not in ("api_error", "judge_error"):
                    done[r["id"]] = r
    todo = [s for s in samples if s["id"] not in done]
    method = load_method(method_id, args.methods_config)
    backend = ThinkingClientBackend(
        client_factory(False), client_factory(True), "intern-s2-preview-35b"
    )
    mclient = SpecAwareMethodClient(method, backend, None)
    cache: dict = {}
    print(
        f"[{pkg}/{method_id}] {len(samples)} samples, {len(done)} cached, {len(todo)} to run",
        flush=True,
    )
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    results = dict(done)
    with path.open("w", encoding="utf-8") as f:
        for r in done.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()

        async def worker(s):
            async with sem:
                CURRENT.set(scenario_for(pkg, mod, s, criteria, cache))
                rec = await mod.evaluate_one(s, mclient, judge, criteria, args)
                rec["method"] = method_id
                return rec

        n = 0
        for fut in asyncio.as_completed([worker(s) for s in todo]):
            r = await fut
            results[r["id"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0 or n == len(todo):
                print(f"[{pkg}/{method_id}] {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    records = [results[s["id"]] for s in samples]
    summary = {
        "package": pkg,
        "method": method_id,
        "samples": len(samples),
        **mod.summarize(records, criteria),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", required=True)
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--packages", default="value_safety,spec_alignment")
    p.add_argument("--configs", type=Path, required=True)
    p.add_argument("--methods-config", type=Path, required=True)
    p.add_argument(
        "--ids-dir",
        type=Path,
        help="dir with {dev,test}_{package}_sample_ids.json "
        "(e.g. test/test3/results/intern_method_uplift); overrides --dev-ids/--test-run",
    )
    p.add_argument(
        "--dev-ids", type=Path, help="jailguard_calib/divergences.jsonl (calibration split)"
    )
    p.add_argument("--test-run", type=Path, help="sampled_v1 dir (*_sample_ids.json)")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N samples per package")
    a = p.parse_args()
    out_root = a.results / a.split
    dev = {}
    for line in a.dev_ids.read_text().splitlines() if a.dev_ids and not a.ids_dir else []:
        r = json.loads(line)
        if r["split"] == "calibration":
            dev.setdefault(r["pkg"], set()).add(r["id"])
    for pkg in a.packages.split(","):
        path = base.PACKAGES[pkg]
        mod = base.load_module(f"evalpkg2_{pkg}", path / "evaluate.py")
        if hasattr(mod, "SCENARIOS") and (path / "scenarios.json").exists():
            mod.SCENARIOS = {
                k: f"{v['name']} — {v['description']}"
                for k, v in json.loads(
                    (path / "scenarios.json").read_text(encoding="utf-8")
                ).items()
            }
        models = mod.load_yaml(a.configs / "models.yaml")["models"]
        criteria = json.loads((path / "criteria.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(x)
            for x in (path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        if a.ids_dir:
            ids = set(json.loads((a.ids_dir / f"{a.split}_{pkg}_sample_ids.json").read_text()))
        else:
            ids = (
                dev[pkg]
                if a.split == "dev"
                else set(json.loads((a.test_run / f"{pkg}_sample_ids.json").read_text()))
            )
        samples = sorted((r for r in rows if r["id"] in ids), key=lambda r: r["id"])
        if a.limit:
            samples = samples[: a.limit]
        (out_root).mkdir(parents=True, exist_ok=True)
        (out_root / f"{pkg}_sample_ids.json").write_text(
            json.dumps([s["id"] for s in samples], indent=1)
        )
        margs = SimpleNamespace(
            temperature=0.0,
            max_tokens=4096,
            timeout=600,
            retries=8,
            judge="gpt-5.6-luna",
            concurrency=a.concurrency,
            methods_config=a.methods_config,
        )
        judge = mod.Client(mod.model_config(models, "gpt-5.6-luna"), 600, 8)
        mcfg = mod.model_config(models, "intern-s2-preview-35b")
        print(f"== {pkg} {a.split}: {len(samples)} samples", flush=True)
        for method_id in a.methods.split(","):

            def client_factory(thinking: bool, mod=mod, mcfg=mcfg):
                cfg = dict(mcfg)
                if thinking:
                    cfg["extra_body"] = {**(cfg.get("extra_body") or {}), "thinking_mode": True}
                    cfg["max_tokens"] = max(int(cfg.get("max_tokens") or 0), 16384)
                return mod.Client(cfg, 600, 8)

            await run_method(
                pkg, mod, samples, criteria, method_id, judge, client_factory, margs, out_root
            )
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
