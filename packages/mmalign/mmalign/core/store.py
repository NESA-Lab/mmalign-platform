"""ResultStore Protocol and FileResultStore (MVP file-system implementation).

File layout
-----------
``<workspace>/results/`` (default ``data/results/`` in dev, XDG workspace when
``MMALIGN_USE_XDG=1``) or a custom ``root`` path:

    results/
    ├── runs/
    │   └── {run_id}.json          # BenchmarkRun (metadata + summary)
    ├── samples/
    │   └── {run_id}.jsonl         # one SampleResult per line (stream-append)
    ├── reviews/
    │   └── {run_id}.jsonl         # one HumanReviewLabel per line
    └── index.json                 # BenchmarkRunSummary list; rebuilt by rescan()

Thread-safety
-------------
``FileResultStore`` is safe for concurrent reads.  Concurrent writes from
multiple processes (e.g. CLI + API both appending to the same run) may cause
interleaved JSONL lines; callers are expected to use a single writer process
per run in the MVP.  The ``index.json`` is rewritten atomically via a temp-
file rename.

Rebuilding
----------
``FileResultStore.rescan()`` regenerates ``index.json`` by reading all
``runs/*.json`` files.  Call it after external file copies or repairs.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol, cast, runtime_checkable

from .result import (
    BenchmarkRun,
    BenchmarkRunSummary,
    HumanReviewLabel,
    RunStatus,
    SampleResult,
)


def _default_root() -> Path:
    from mmalign.paths import default_results_dir

    return default_results_dir()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ResultStore(Protocol):
    """Unified result storage interface.

    CLI, REST API, and WebUI all share the same underlying store instance.
    The protocol is intentionally minimal; the ``FileResultStore`` provides
    additional helpers (``rescan``, ``delete_run``) not in the Protocol.
    """

    def save_run(self, run: BenchmarkRun) -> None:
        """Persist or update a BenchmarkRun record (upsert by run_id)."""
        ...

    def append_sample_result(self, result: SampleResult) -> None:
        """Append one SampleResult to the run's JSONL file."""
        ...

    def append_review_label(self, label: HumanReviewLabel) -> None:
        """Append one HumanReviewLabel to the run's review JSONL file."""
        ...

    def get_run(self, run_id: str) -> BenchmarkRun:
        """Return the BenchmarkRun for ``run_id``.

        Raises
        ------
        KeyError
            If the run does not exist.
        """
        ...

    def list_runs(
        self,
        *,
        status: RunStatus | None = None,
        scenario: str | None = None,
        model: str | None = None,
        method: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[BenchmarkRunSummary], int]:
        """Return a page of run summaries and the total count.

        All filters are ANDed.  Returns ``(page, total)`` where ``total``
        is the unfiltered-but-filtered total (not the full store size).
        """
        ...

    def iter_sample_results(self, run_id: str) -> Iterator[SampleResult]:
        """Yield SampleResult objects from the run's JSONL file."""
        ...

    def iter_review_labels(self, run_id: str) -> Iterator[HumanReviewLabel]:
        """Yield HumanReviewLabel objects from the run's review JSONL file."""
        ...


# ---------------------------------------------------------------------------
# File-system implementation
# ---------------------------------------------------------------------------

class RunNotFoundError(KeyError):
    """Raised when a requested run_id is not found in the store."""


class FileResultStore:
    """MVP ResultStore backed by the local file system.

    Parameters
    ----------
    root:
        Path to the results root directory.  Created on first write if
        missing.  Defaults to :func:`mmalign.paths.default_results_dir`.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root) if root else _default_root()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        for d in (self._runs_dir, self._samples_dir, self._reviews_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def _runs_dir(self) -> Path:
        return self._root / "runs"

    @property
    def _samples_dir(self) -> Path:
        return self._root / "samples"

    @property
    def _reviews_dir(self) -> Path:
        return self._root / "reviews"

    @property
    def _index_path(self) -> Path:
        return self._root / "index.json"

    @property
    def _platform_stats_path(self) -> Path:
        return self._root / "platform_stats.json"

    def _run_path(self, run_id: str) -> Path:
        return self._runs_dir / f"{run_id}.json"

    def _samples_path(self, run_id: str) -> Path:
        return self._samples_dir / f"{run_id}.jsonl"

    def _reviews_path(self, run_id: str) -> Path:
        return self._reviews_dir / f"{run_id}.jsonl"

    def _write_atomic(self, path: Path, content: str) -> None:
        """Write content to path atomically via a temp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _update_index(self, summary: BenchmarkRunSummary) -> None:
        """Upsert one summary into index.json atomically."""
        summaries = self._load_index()
        existing = {s.run_id: i for i, s in enumerate(summaries)}
        if summary.run_id in existing:
            summaries[existing[summary.run_id]] = summary
        else:
            summaries.append(summary)
        self._write_atomic(
            self._index_path,
            json.dumps(
                [s.model_dump(mode="json") for s in summaries],
                ensure_ascii=False,
                indent=None,
            ),
        )

    def _load_index(self) -> list[BenchmarkRunSummary]:
        if not self._index_path.exists():
            return []
        try:
            data: list[dict[str, Any]] = json.loads(
                self._index_path.read_text(encoding="utf-8")
            )
            return [BenchmarkRunSummary.model_validate(d) for d in data]
        except (json.JSONDecodeError, Exception):
            return []

    # ------------------------------------------------------------------
    # Write interface
    # ------------------------------------------------------------------

    def save_run(self, run: BenchmarkRun) -> None:
        self._ensure_dirs()
        self._write_atomic(
            self._run_path(run.run_id),
            run.model_dump_json(indent=2),
        )
        self._update_index(run.to_summary())

    def append_sample_result(self, result: SampleResult) -> None:
        self._ensure_dirs()
        path = self._samples_path(result.run_id)
        line = result.model_dump_json() + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    def append_review_label(self, label: HumanReviewLabel) -> None:
        self._ensure_dirs()
        path = self._reviews_path(label.run_id)
        line = label.model_dump_json() + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    # ------------------------------------------------------------------
    # Read interface
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> BenchmarkRun:
        path = self._run_path(run_id)
        if not path.exists():
            raise RunNotFoundError(f"Run '{run_id}' not found at {path}")
        return BenchmarkRun.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(
        self,
        *,
        status: RunStatus | None = None,
        scenario: str | None = None,
        model: str | None = None,
        method: str | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[BenchmarkRunSummary], int]:
        summaries = self._load_index()

        # Apply filters
        filtered: list[BenchmarkRunSummary] = []
        for s in summaries:
            if status is not None and s.status != status:
                continue
            if scenario is not None and s.scenario != scenario:
                continue
            if model is not None and s.model != model:
                continue
            if method is not None and s.method != method:
                continue
            if started_after is not None and s.started_at <= started_after:
                continue
            if started_before is not None and s.started_at >= started_before:
                continue
            filtered.append(s)

        # Sort newest first
        filtered.sort(key=lambda s: s.started_at, reverse=True)
        total = len(filtered)
        page = filtered[offset : offset + limit]
        return page, total

    def iter_sample_results(self, run_id: str) -> Iterator[SampleResult]:
        path = self._samples_path(run_id)
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield SampleResult.model_validate_json(line)

    def iter_review_labels(self, run_id: str) -> Iterator[HumanReviewLabel]:
        path = self._reviews_path(run_id)
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield HumanReviewLabel.model_validate_json(line)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def rescan(self) -> int:
        """Rebuild index.json by scanning all runs/*.json files.

        Returns the number of runs indexed.
        """
        self._ensure_dirs()
        summaries: list[BenchmarkRunSummary] = []
        for path in sorted(self._runs_dir.glob("*.json")):
            try:
                run = BenchmarkRun.model_validate_json(path.read_text(encoding="utf-8"))
                summaries.append(run.to_summary())
            except Exception:
                continue
        self._write_atomic(
            self._index_path,
            json.dumps(
                [s.model_dump(mode="json") for s in summaries],
                ensure_ascii=False,
            ),
        )
        return len(summaries)

    # ------------------------------------------------------------------
    # Platform stats (values with no per-run home: GitHub stars,
    # fine-tuned architectures, etc.). Written by the data generator,
    # read by the Overview metrics endpoint.
    # ------------------------------------------------------------------

    def read_platform_stats(self) -> dict[str, Any]:
        if not self._platform_stats_path.exists():
            return {}
        try:
            return cast(dict[str, Any], json.loads(self._platform_stats_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return {}

    def write_platform_stats(self, data: dict[str, Any]) -> None:
        self._ensure_dirs()
        self._write_atomic(
            self._platform_stats_path,
            json.dumps(data, ensure_ascii=False, indent=2),
        )

    def reset(self) -> None:
        """Remove all runs, samples, reviews, the index, and platform stats.

        Used by the mock data generator's ``--reset`` flag to start from a
        clean store.  Irreversible.
        """
        import shutil

        for d in (self._runs_dir, self._samples_dir, self._reviews_dir):
            if d.exists():
                shutil.rmtree(d)
        for f in (self._index_path, self._platform_stats_path):
            if f.exists():
                f.unlink()

    def delete_run(self, run_id: str) -> None:
        """Remove all files associated with a run (irreversible)."""
        for path in (
            self._run_path(run_id),
            self._samples_path(run_id),
            self._reviews_path(run_id),
        ):
            if path.exists():
                path.unlink()
        # Rebuild index without this run
        self.rescan()

    def root(self) -> Path:
        return self._root


def new_run_id() -> str:
    """Generate a unique run ID with a timestamp prefix for natural sort order."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    short = str(uuid.uuid4()).replace("-", "")[:8]
    return f"run-{ts}-{short}"
