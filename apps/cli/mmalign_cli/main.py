"""MMAlign CLI — typer-based entry point.

Commands
--------
mmalign benchmark   Run a full benchmark evaluation.
mmalign demo        Run a single-sample interactive alignment demo.
mmalign capability  List available models, methods, evaluators, and scenarios.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

app = typer.Typer(
    name="mmalign",
    help="MMAlign — multimodal value alignment evaluation platform.",
    add_completion=False,
)

def _find_configs_dir(start: Path | None = None) -> Path:
    from mmalign.paths import default_configs_dir

    return default_configs_dir(start)


def _print_comparison_table(runs: list[Any]) -> None:
    """Print a comparison table for multi-method benchmark runs."""
    col_w = [20, 10, 12, 12, 10, 10]
    headers = ["Method", "Status", "Safety", "Behavioral", "Total", "Samples"]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    def row(cells: list[str]) -> str:
        return "|" + "|".join(f" {c:<{col_w[i]}} " for i, c in enumerate(cells)) + "|"

    typer.echo("\n=== Multi-Method Comparison ===")
    typer.echo(sep)
    typer.echo(row(headers))
    typer.echo(sep)
    for run in runs:
        s = run.summary
        cells = [
            run.method,
            run.status,
            f"{s.safety_score:.1%}" if s else "—",
            f"{s.behavioral_score:.1%}" if s else "—",
            f"{s.total_score:.1%}" if s else "—",
            str(s.sample_count) if s else "—",
        ]
        typer.echo(row(cells))
    typer.echo(sep)


def _find_results_dir(configs_dir: Path) -> Path:
    from mmalign.paths import default_results_dir

    return default_results_dir(configs_dir.parent)


@app.command()
def benchmark(
    model: Annotated[str, typer.Option("--model", "-m", help="Model ID from models.yaml")],
    scenario: Annotated[str, typer.Option("--scenario", "-s", help="Benchmark ID (metadata.yaml package)")],
    benchmark_origin: Annotated[
        Optional[str],
        typer.Option("--origin", help="builtin | user (required if id exists in both)"),
    ] = None,
    method: Annotated[str, typer.Option(help="Alignment method")] = "align3",
    evaluator: Annotated[str, typer.Option(help="Evaluator")] = "llm_judge",
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="Max samples")] = None,
    concurrency: Annotated[int, typer.Option(help="Concurrent workers")] = 4,
    configs: Annotated[Optional[Path], typer.Option(help="Path to configs directory")] = None,
    results: Annotated[Optional[Path], typer.Option(help="Path to results directory")] = None,
    vanilla_path: Annotated[Optional[Path], typer.Option(
        help="Path to vanilla JSONL results for align3 focus mode"
    )] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress summary output")] = False,
) -> None:
    """Run a benchmark evaluation run for a given model and scenario."""
    configs_dir = configs or _find_configs_dir()
    results_dir = results or _find_results_dir(configs_dir)

    typer.echo(f"Starting benchmark: model={model} scenario={scenario} method={method}")
    typer.echo(f"Configs: {configs_dir}  Results: {results_dir}")

    from mmalign.core.method import AlignContext
    from mmalign.engines.diagnosis.runner import DiagnosisEngine
    from mmalign.engines.diagnosis.report import print_run_summary

    engine = DiagnosisEngine(configs_dir=configs_dir, results_dir=results_dir)
    context = AlignContext(
        vanilla_results_path=str(vanilla_path) if vanilla_path else None
    )

    methods = [m.strip() for m in method.split(",") if m.strip()]

    async def _run() -> None:
        if len(methods) == 1:
            run = await engine.run_benchmark(
                model=model,
                scenario=scenario,
                method=methods[0],
                evaluator=evaluator,
                benchmark_origin=benchmark_origin,
                sample_limit=limit,
                concurrency=concurrency,
                context=context,
            )
            if not quiet:
                from mmalign.core.store import FileResultStore
                store = FileResultStore(results_dir)
                samples = list(store.iter_sample_results(run.run_id))
                print_run_summary(run, samples)
            if run.status == "failed":
                raise typer.Exit(1)
        else:
            runs = []
            for m in methods:
                typer.echo(f"\n--- Method: {m} ---")
                run = await engine.run_benchmark(
                    model=model,
                    scenario=scenario,
                    method=m,
                    evaluator=evaluator,
                    benchmark_origin=benchmark_origin,
                    sample_limit=limit,
                    concurrency=concurrency,
                    context=context,
                )
                runs.append(run)
            if not quiet:
                _print_comparison_table(runs)

    asyncio.run(_run())


def _default_model_id(configs_dir: Path) -> str:
    """First non-mock model from models.yaml, or explicit error if only mock is configured."""
    import yaml

    raw = yaml.safe_load((configs_dir / "models.yaml").read_text(encoding="utf-8"))
    models: dict[str, Any] = raw.get("models", {})
    for model_id, cfg in models.items():
        if cfg.get("provider") != "mock":
            return model_id
    raise typer.BadParameter(
        "No production model in models.yaml. Register a real backend or pass --model explicitly."
    )


@app.command()
def demo(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model ID from models.yaml (default: first non-mock)"),
    ] = None,
    scenario: Annotated[str, typer.Option("--scenario", "-s")] = "education",
    benchmark_origin: Annotated[
        Optional[str],
        typer.Option("--origin", help="builtin | user"),
    ] = "builtin",
    method: Annotated[str, typer.Option()] = "align3",
    prompt: Annotated[Optional[str], typer.Option("--prompt", "-p", help="Custom user prompt")] = None,
    configs: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    """Run a single-sample alignment demo and print the result."""
    configs_dir = configs or _find_configs_dir()
    model_id = model or _default_model_id(configs_dir)

    typer.echo(f"Demo mode: model={model_id} scenario={scenario} method={method}")

    from mmalign.loaders.scenarios import load_scenario
    from mmalign.loaders.methods import load_method
    from mmalign.backends.loader import load_backends
    from mmalign.core.types import GenerationRequest, InputMessage
    from mmalign.core.method import AlignContext

    async def _run() -> None:
        origin = benchmark_origin if benchmark_origin in ("builtin", "user") else None
        loaded_scenario, samples = load_scenario(
            scenario, origin=origin, model=model_id, sample_limit=1
        )
        backends = load_backends(configs_dir / "models.yaml", only=[model_id])
        backend = backends[model_id]
        align_method = load_method(method, configs_dir / "methods.yaml")
        align_method.validate(loaded_scenario, backend)

        # Use first sample or custom prompt
        if prompt:
            request = GenerationRequest(
                model=model_id,
                input=[InputMessage.user(prompt)],
            )
        elif samples:
            request = samples[0].request
        else:
            typer.echo("[error] No samples found for scenario.", err=True)
            raise typer.Exit(1)

        typer.echo(f"\nPrompt: {_request_preview(request)}\n")

        # Raw baseline — backend output without any alignment.
        raw = await backend.generate(request)
        typer.echo("=== Raw (no alignment) ===")
        typer.echo(raw.text() or "(empty response)")

        # Aligned output — through the chosen InferenceAlignMethod.
        aligned = await align_method.align(request, backend, loaded_scenario)

        typer.echo("\n=== Aligned ({}) ===".format(method))
        typer.echo(aligned.text())
        if aligned.thought:
            typer.echo(f"\n[thought] {aligned.thought}")

    asyncio.run(_run())


@app.command()
def capability(
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model ID for capability eval (required with --run)")] = None,
    scenario: Annotated[str, typer.Option("--scenario", "-s", help="Scenario ID for capability eval")] = "education",
    run: Annotated[bool, typer.Option("--run/--list", help="Run a capability benchmark instead of listing resources")] = False,
    evaluator: Annotated[str, typer.Option(help="Evaluator for capability eval")] = "llm_judge",
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="Max samples for capability eval")] = None,
    configs: Annotated[Optional[Path], typer.Option()] = None,
    results: Annotated[Optional[Path], typer.Option()] = None,
) -> None:
    """List available resources, or run a capability benchmark (normal-set evaluation without alignment).

    Without --run: prints available models, methods, evaluators, and scenarios.

    With --run: evaluates the model on a scenario using the raw method (no alignment
    intervention) to measure baseline capability retention.
    """
    configs_dir = configs or _find_configs_dir()

    if not run:
        import yaml

        def _load(filename: str, key: str) -> list[str]:
            p = configs_dir / filename
            if not p.exists():
                return ["(file not found)"]
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            items = raw.get(key, {})
            return list(items.keys()) if isinstance(items, dict) else [e.get("id", "?") for e in items]

        typer.echo("=== MMAlign Capability Report ===\n")
        typer.echo("Models (models.yaml):")
        for m in _load("models.yaml", "models"):
            typer.echo(f"  {m}")
        typer.echo("\nMethods (methods.yaml):")
        for m in _load("methods.yaml", "methods"):
            typer.echo(f"  {m}")
        typer.echo("\nEvaluators (evaluators.yaml):")
        for e in _load("evaluators.yaml", "evaluators"):
            typer.echo(f"  {e}")
        typer.echo("\nBenchmarks (data/benchmarks/):")
        from mmalign.loaders.scenarios import list_scenarios
        for s in list_scenarios():
            typer.echo(f"  [{s['origin']}] {s['id']} — {s.get('name', s['id'])}")
        typer.echo("")
        return

    # --run mode: execute a raw benchmark to measure normal capability
    results_dir = results or _find_results_dir(configs_dir)
    model_id = model or _default_model_id(configs_dir)

    typer.echo(f"Capability eval: model={model_id} scenario={scenario} method=raw (no alignment)")
    typer.echo(f"Configs: {configs_dir}  Results: {results_dir}")

    from mmalign.engines.diagnosis.runner import DiagnosisEngine
    from mmalign.engines.diagnosis.report import print_run_summary

    engine = DiagnosisEngine(configs_dir=configs_dir, results_dir=results_dir)

    async def _run() -> None:
        run_result = await engine.run_benchmark(
            model=model_id,
            scenario=scenario,
            method="raw",
            evaluator=evaluator,
            sample_limit=limit,
        )
        from mmalign.core.store import FileResultStore
        store = FileResultStore(results_dir)
        samples = list(store.iter_sample_results(run_result.run_id))
        print_run_summary(run_result, samples)
        typer.echo("\n[capability] Use 'mmalign benchmark' with --method align3 to compare aligned vs raw.")
        if run_result.status == "failed":
            raise typer.Exit(1)

    asyncio.run(_run())


def _request_preview(request: "GenerationRequest") -> str:
    for msg in request.input_messages():
        if msg.role == "user":
            content = msg.content
            if isinstance(content, str):
                return content[:120]
            return " ".join(
                (p.text or "")[:60] for p in content if p.type == "input_text"
            )[:120]
    return "(no user message)"


@app.command()
def seed(
    scale: Annotated[str, typer.Option("--scale", help="large | small")] = "large",
    reset: Annotated[bool, typer.Option("--reset", help="Wipe the store before generating")] = False,
    seed_value: Annotated[int, typer.Option("--seed", help="RNG seed for reproducibility")] = 42,
    results: Annotated[Optional[Path], typer.Option(help="Results directory (default: workspace)")] = None,
    configs: Annotated[Optional[Path], typer.Option(help="Path to configs directory")] = None,
) -> None:
    """Generate a realistic mock evaluation corpus into the FileResultStore.

    Writes benchmark packages, runs, samples, review labels, and platform
    stats from a single seed.  Every dashboard chart derives from this corpus.
    """
    from mmalign.mock import generate

    configs_dir = configs or _find_configs_dir()
    results_dir = results or _find_results_dir(configs_dir)
    workspace_start = configs_dir.parent

    typer.echo(f"Generating mock data (scale={scale}, reset={reset}) → {results_dir}")
    report = generate(
        scale=scale,
        reset=reset,
        seed=seed_value,
        results_dir=results_dir,
        workspace_start=workspace_start,
    )
    d = report.as_dict()
    typer.echo(
        f"  ✓ {d['runs']} runs ({d['failed_runs']} failed), {d['samples']} sample results, "
        f"{d['reviews']} review labels"
    )
    typer.echo(
        f"  ✓ {d['scenarios']} benchmark scenarios, {d['benchmark_prompts']} prompts"
    )
    typer.echo(f"Done — corpus at {d['root']}")


@app.command()
def train(
    method: Annotated[str, typer.Option("--method", "-t", help="Training method: dpo | sft")] = "dpo",
    base_model: Annotated[str, typer.Option("--base-model", help="HuggingFace model ID or local path")] = "Qwen/Qwen2.5-7B-Instruct",
    data: Annotated[Optional[Path], typer.Option("--data", "-d", help="Path to JSONL training data")] = None,
    output_dir: Annotated[Optional[Path], typer.Option("--output-dir", "-o", help="Checkpoint output directory")] = None,
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="Number of training epochs")] = 1,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Per-device train batch size")] = 4,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate")] = 5e-5,
    lora_rank: Annotated[int, typer.Option("--lora-rank", help="LoRA rank")] = 16,
    validate: Annotated[bool, typer.Option("--validate/--no-validate", help="Run benchmark validation after training")] = True,
    validate_scenario: Annotated[Optional[str], typer.Option("--validate-scenario", help="Scenario for post-training validation")] = None,
    validate_model: Annotated[Optional[str], typer.Option("--validate-model", help="Model ID for validation benchmark")] = None,
    configs: Annotated[Optional[Path], typer.Option(help="Path to configs directory")] = None,
    results: Annotated[Optional[Path], typer.Option(help="Path to results directory")] = None,
) -> None:
    """Run a training-time alignment pipeline (DPO or SFT).

    Example usage::

        # DPO fine-tune with preference data
        mmalign train --method dpo --data reviews.jsonl --base-model Qwen/Qwen2.5-7B-Instruct

        # SFT fine-tune
        mmalign train --method sft --data sft_corpus.jsonl --epochs 3

    Required packages (not installed by default)::

        pip install trl transformers peft accelerate datasets

    The training run writes a LoRA checkpoint to the output directory and
    optionally runs ``mmalign benchmark`` to verify that safety is maintained.
    """
    import json

    configs_dir = configs or _find_configs_dir()
    results_dir = results or _find_results_dir(configs_dir)
    out_dir = output_dir or (configs_dir.parent / "checkpoints")

    typer.echo(f"Training: method={method} base_model={base_model}")

    from mmalign.engines.alignment.training import TrainingConfig, TrainingEngine, TrainingEnvironmentError

    train_config = TrainingConfig(
        base_model=base_model,
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        lora_rank=lora_rank,
    )

    # Load training data
    if data is None or not data.exists():
        typer.echo(
            "[error] --data is required and must point to a JSONL file containing "
            "training examples.\n"
            f"  DPO format: {{\"prompt\": \"...\", \"chosen\": \"...\", \"rejected\": \"...\"}}\n"
            f"  SFT format: {{\"text\": \"...\"}}\n",
            err=True,
        )
        raise typer.Exit(1)

    dataset: list[dict] = []
    with data.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    typer.echo(f"Loaded {len(dataset)} training examples from {data}")

    engine = TrainingEngine(configs_dir=configs_dir, results_dir=results_dir)

    async def _run() -> None:
        try:
            result = await engine.train(
                method=method,
                dataset=dataset,
                config=train_config,
                validate_after=validate,
                validate_scenario=validate_scenario,
                validate_model=validate_model,
            )
            typer.echo(f"\n=== Training complete ===")
            typer.echo(f"Method:      {result.method}")
            typer.echo(f"Base model:  {result.base_model}")
            typer.echo(f"Checkpoint:  {result.checkpoint_uri}")
            for k, v in result.final_metrics.items():
                typer.echo(f"{k:20s}: {v:.4f}")
        except TrainingEnvironmentError as exc:
            typer.echo(f"\n[error] {exc}", err=True)
            raise typer.Exit(1)
        except Exception as exc:
            typer.echo(f"\n[error] Training failed: {exc}", err=True)
            raise typer.Exit(1)

    asyncio.run(_run())


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", "-p", help="Port")] = 8000,
    configs: Annotated[Optional[Path], typer.Option(help="Path to configs directory")] = None,
    results: Annotated[Optional[Path], typer.Option(help="Path to results directory")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes")] = False,
) -> None:
    """Start the MMAlign REST API server."""
    import uvicorn
    from mmalign_api import deps as api_deps

    configs_dir = configs or _find_configs_dir()
    results_dir = results or _find_results_dir(configs_dir)

    # Set env vars so paths survive uvicorn reload workers
    import os
    os.environ["MMALIGN_CONFIGS_DIR"] = str(configs_dir.resolve())
    os.environ["MMALIGN_RESULTS_DIR"] = str(results_dir.resolve())
    api_deps.configure(configs_dir.resolve(), results_dir.resolve())

    typer.echo(f"Starting MMAlign API server on http://{host}:{port}")
    typer.echo(f"Docs: http://{host}:{port}/api/docs")
    typer.echo(f"Configs: {configs_dir}  Results: {results_dir}")

    uvicorn.run(
        "mmalign_api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
