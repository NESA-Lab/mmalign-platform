"""CLI terminal summary for benchmark runs.

Uses ``rich`` for formatted terminal output.  Falls back gracefully when
``rich`` is not available or when stdout is not a TTY.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmalign.core.result import BenchmarkRun, SampleResult


def print_run_summary(
    run: "BenchmarkRun",
    sample_results: list["SampleResult"] | None = None,
) -> None:
    """Print a formatted benchmark run summary to stdout."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        _rich_summary(console, run, sample_results)
    except ImportError:
        _plain_summary(run, sample_results)


def _rich_summary(console: object, run: "BenchmarkRun", samples: list["SampleResult"] | None) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    from rich.text import Text

    console = Console()
    s = run.summary

    status_color = {
        "completed": "green",
        "failed": "red",
        "running": "yellow",
        "pending": "blue",
        "cancelled": "dim",
    }.get(run.status, "white")

    # Header panel
    header = (
        f"[bold]Run ID:[/bold] {run.run_id}\n"
        f"[bold]Status:[/bold] [{status_color}]{run.status}[/{status_color}]\n"
        f"[bold]Model:[/bold]  {run.model}  [bold]Scenario:[/bold] {run.scenario}\n"
        f"[bold]Method:[/bold] {run.method}  [bold]Evaluator:[/bold] {run.evaluator}"
    )
    if run.completed_at and run.started_at:
        elapsed = (run.completed_at - run.started_at).total_seconds()
        header += f"\n[bold]Elapsed:[/bold] {elapsed:.1f}s"

    console.print(Panel(header, title="[bold blue]MMAlign Benchmark[/bold blue]", expand=False))

    if s is None:
        console.print("[yellow]No summary available.[/yellow]")
        return

    # Scores table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    def _pct(val: float) -> str:
        return f"{val * 100:.1f}%"

    safe_color = "green" if s.safety_score >= 0.9 else ("yellow" if s.safety_score >= 0.5 else "red")
    table.add_row("Safety Score (avg)",    f"[{safe_color}]{_pct(s.safety_score)}[/{safe_color}]")
    table.add_row("Behavioral Score",      _pct(s.behavioral_score))
    table.add_row("Total SAR Score",       f"[bold]{_pct(s.total_score)}[/bold]")
    table.add_row("Unsafe Rate",           f"[red]{_pct(s.unsafe_rate)}[/red]" if s.unsafe_rate > 0 else _pct(s.unsafe_rate))
    table.add_row("Samples",               str(s.sample_count))
    table.add_row("Errors",                f"[red]{s.error_count}[/red]" if s.error_count > 0 else str(s.error_count))
    if s.capability_rate is not None:
        table.add_row("Capability Rate",   _pct(s.capability_rate))

    console.print(table)

    # Per-sample breakdown (top-5 failures)
    if samples:
        failures = [sr for sr in samples if sr.is_unsafe or sr.status == "failed"]
        if failures:
            console.print(f"\n[bold yellow]⚠ {len(failures)} unsafe / failed samples:[/bold yellow]")
            ftable = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            ftable.add_column("Sample ID")
            ftable.add_column("Status")
            ftable.add_column("SAR")
            ftable.add_column("Error")
            for sr in failures[:5]:
                sar = f"{sr.eval_result.total_score * 100:.0f}%" if sr.eval_result else "-"
                err = (sr.telemetry.error_code or "-") if sr.telemetry else "-"
                ftable.add_row(sr.sample_id, sr.status, sar, err)
            console.print(ftable)
            if len(failures) > 5:
                console.print(f"  ... and {len(failures) - 5} more.")


def _plain_summary(run: "BenchmarkRun", samples: list["SampleResult"] | None) -> None:
    s = run.summary
    print(f"\n=== MMAlign Benchmark Run ===")
    print(f"Run ID:   {run.run_id}")
    print(f"Status:   {run.status}")
    print(f"Model:    {run.model}  Scenario: {run.scenario}")
    print(f"Method:   {run.method}  Evaluator: {run.evaluator}")
    if s:
        print(f"Safety:   {s.safety_score * 100:.1f}%")
        print(f"Behavioral: {s.behavioral_score * 100:.1f}%")
        print(f"SAR Total:  {s.total_score * 100:.1f}%")
        print(f"Unsafe rate: {s.unsafe_rate * 100:.1f}%  Samples: {s.sample_count}  Errors: {s.error_count}")
    print("=============================\n")
