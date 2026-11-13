"""Sampled evaluation of mmalign inference-time methods on the two MMAlign eval packages.

Reuses each package's own evaluate.py for everything except answering:
sample format, judge prompt (JUDGE_TEMPLATE), judge parsing, status classification
(blocked / api_error / judge_error) and metrics (summarize). The only change is that
the "model" is an mmalign InferenceAlignMethod running on top of the package's own
OpenAI-compatible Client, so results are directly comparable with the packages'
Intern-S2 baselines (same model config, same judge, same metrics).

Usage (from mmalign-platform repo root, with INTERN_API_KEY and OPENAI_API_KEY set):
  python <this file> --methods raw,self_reminder,... --per-cell 10 --run sampled_v1
Resumable: rerun the same command; finished (method, package, sample) records are kept.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from mmalign.core.scenario import Scenario
from mmalign.core.types import (
    BackendCapabilities,
    GenerationOutput,
    GenerationRequest,
    GenerationResult,
    InputContentPart,
    InputMessage,
    OutputContentPart,
)
from mmalign.loaders.methods import load_method

ROOT = Path.cwd()
PACKAGES = {
    "value_safety": ROOT / "test/multimodal_value_safety",
    "spec_alignment": ROOT / "test/multimodal_spec_alignment",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class ProviderError(RuntimeError):
    """Raised by the backend so the method aborts; message keeps the provider error text."""


class ClientBackend:
    """mmalign ModelBackend on top of an evaluate.py Client (keeps its extra_body / retries)."""

    def __init__(self, client, model_id: str, default_temperature: float = 0.0) -> None:
        self.client = client
        self.default_temperature = default_temperature  # package baselines ran at temperature 0
        self.model_id = model_id
        self.capabilities = BackendCapabilities(input_modalities={"text", "image"})
        self.last_finish_reason: str | None = None

    @staticmethod
    def _to_openai(request: GenerationRequest) -> list[dict]:
        out = []
        for m in request.input_messages():
            if isinstance(m.content, str):
                out.append({"role": m.role, "content": m.content})
                continue
            parts = []
            for p in m.content:
                if p.type == "input_text":
                    parts.append({"type": "text", "text": p.text or ""})
                elif p.type == "input_image" and p.image_url:
                    parts.append({"type": "image_url", "image_url": {"url": p.image_url}})
            if all(x["type"] == "text" for x in parts):
                out.append({"role": m.role, "content": "\n".join(x["text"] for x in parts)})
            else:
                out.append({"role": m.role, "content": parts})
        return out

    async def generate(self, request: GenerationRequest, **kwargs) -> GenerationResult:
        temperature = (
            self.default_temperature if request.temperature is None else request.temperature
        )
        r = await self.client.chat(
            self._to_openai(request), temperature=temperature, max_tokens=request.max_output_tokens
        )
        if r.get("error"):
            raise ProviderError(r["error"])
        if r.get("finish_reason") == "content_filter":
            raise ProviderError("content_filter")
        self.last_finish_reason = r.get("finish_reason")
        usage = {k: v for k, v in (r.get("usage") or {}).items() if isinstance(v, int)}
        return GenerationResult(
            id="g",
            model=self.model_id,
            usage=usage,
            output=[GenerationOutput(content=[OutputContentPart.text_part(r["text"])])],
        )

    async def embed(self, parts):
        raise NotImplementedError

    async def score(self, request, response):
        raise NotImplementedError


class MethodClient:
    """Duck-types evaluate.py's Client.chat(), answering through an mmalign method."""

    def __init__(self, method, backend: ClientBackend, scenario: Scenario) -> None:
        self.method, self.backend, self.scenario = method, backend, scenario
        self.retry_count = 0

    @staticmethod
    def _to_request(messages: list[dict]) -> GenerationRequest:
        msgs = []
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                msgs.append(InputMessage(role=m["role"], content=[InputContentPart.text_part(c)]))
                continue
            parts = []
            for p in c:
                if p["type"] == "text":
                    parts.append(InputContentPart.text_part(p["text"]))
                elif p["type"] == "image_url":
                    parts.append(InputContentPart.image_url_part(p["image_url"]["url"]))
            msgs.append(InputMessage(role=m["role"], content=parts))
        return GenerationRequest(model="intern-s2-preview-35b", input=msgs)

    async def chat(self, messages, *, temperature=None, max_tokens=None) -> dict:
        t0 = time.time()
        try:
            out = await self.method.align(self._to_request(messages), self.backend, self.scenario)
        except ProviderError as e:
            return {
                "text": "",
                "error": str(e)[:500],
                "finish_reason": None,
                "usage": None,
                "latency_s": round(time.time() - t0, 2),
            }
        except Exception as e:  # noqa: BLE001 - method bug: record, don't crash the run
            return {
                "text": "",
                "error": f"method_error: {type(e).__name__}: {e}"[:500],
                "finish_reason": None,
                "usage": None,
                "latency_s": round(time.time() - t0, 2),
            }
        return {
            "text": out.text().strip(),
            "finish_reason": "stop",
            "usage": out.usage,
            "latency_s": round(time.time() - t0, 2),
            "method_meta": {
                "attempts": out.attempts,
                "refusal": out.is_refusal(),
                **{
                    k: v
                    for k, v in out.metadata.items()
                    if k
                    in (
                        "unsafe_detected",
                        "detected",
                        "max_divergence",
                        "defense_applied",
                        "similarity_source",
                        "image_mutated",
                        "all_responses_refused",
                    )
                },
            },
        }


def stratified_sample(rows: list[dict], key, per_cell: int, seed: int) -> list[dict]:
    cells = defaultdict(list)
    for r in rows:
        cells[(key(r), r["label"])].append(r)
    rng = random.Random(seed)
    picked = []
    for cell in sorted(cells):
        pool = sorted(cells[cell], key=lambda r: r["id"])
        picked += rng.sample(pool, min(per_cell, len(pool)))
    return sorted(picked, key=lambda r: r["id"])


async def evaluate_package(
    pkg_name, mod, samples, criteria, methods, judge, client_factory, args, run_dir
):
    summaries = {}
    for method_id in methods:
        out_dir = run_dir / pkg_name / method_id
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
        method = load_method(method_id, ROOT / "configs/methods.yaml")
        backend = ClientBackend(client_factory(), "intern-s2-preview-35b")
        scenario = Scenario(
            id=pkg_name,
            name=pkg_name,
            description=pkg_name,
            modality="image",
            safety_specs=[],
            behavioral_specs=[],
        )
        mclient = MethodClient(method, backend, scenario)
        print(
            f"[{pkg_name}/{method_id}] {len(samples)} samples, "
            f"{len(done)} cached, {len(todo)} to run",
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
                    print(
                        f"[{pkg_name}/{method_id}] {n}/{len(todo)} {time.time() - t0:.0f}s",
                        flush=True,
                    )
        records = [results[s["id"]] for s in samples]
        summary = {
            "package": pkg_name,
            "method": method_id,
            "model": "intern-s2-preview-35b",
            "judge": args.judge,
            "samples": len(samples),
            "wall_time_s": round(time.time() - t0, 1),
            **mod.summarize(records, criteria),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summaries[method_id] = summary
    return summaries


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", required=True)
    p.add_argument("--packages", default="value_safety,spec_alignment")
    p.add_argument(
        "--per-cell", type=int, default=10, help="samples per (scenario/subset, label) cell"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="intern-s2-preview-35b")
    p.add_argument("--judge", default="gpt-5.6-luna")
    p.add_argument("--configs", type=Path, required=True, help="dir with models.yaml")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--run", required=True)
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    run_dir = a.results / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    methods = a.methods.split(",")
    all_summaries = {}
    for pkg_name in a.packages.split(","):
        pkg = PACKAGES[pkg_name]
        mod = load_module(f"evalpkg_{pkg_name}", pkg / "evaluate.py")
        if hasattr(mod, "SCENARIOS") and (pkg / "scenarios.json").exists():
            # same formatting as the package's own main()
            mod.SCENARIOS = {
                k: f"{v['name']} — {v['description']}"
                for k, v in json.loads((pkg / "scenarios.json").read_text(encoding="utf-8")).items()
            }
        models = mod.load_yaml(a.configs / "models.yaml")["models"]
        criteria = json.loads((pkg / "criteria.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(x)
            for x in (pkg / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        key = (
            (lambda r: r["subset"] if r["subset"] != "scenario" else "scenario/" + r["scenario"])
            if pkg_name == "value_safety"
            else (lambda r: r["scenario"])
        )
        samples = stratified_sample(rows, key, a.per_cell, a.seed)
        (run_dir / f"{pkg_name}_sample_ids.json").write_text(
            json.dumps([s["id"] for s in samples], indent=1)
        )
        margs = SimpleNamespace(
            temperature=0.0,
            max_tokens=4096,
            timeout=600,
            retries=8,
            judge=a.judge,
            concurrency=a.concurrency,
        )
        judge = mod.Client(mod.model_config(models, a.judge), margs.timeout, margs.retries)
        mcfg = mod.model_config(models, a.model)

        def client_factory(mod=mod, mcfg=mcfg):
            return mod.Client(mcfg, margs.timeout, margs.retries)

        print(f"== {pkg_name}: {len(samples)} sampled ({a.per_cell} per cell)", flush=True)
        all_summaries[pkg_name] = await evaluate_package(
            pkg_name, mod, samples, criteria, methods, judge, client_factory, margs, run_dir
        )
    (run_dir / "all_summaries.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
