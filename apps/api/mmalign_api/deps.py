"""FastAPI dependency injection — shared state for store, configs, and engine.

Paths are resolved via ``mmalign.paths`` (workspace ``data/`` with
``benchmarks/`` + ``results/``; deploy: ``MMALIGN_USE_XDG=1``).
"""

from __future__ import annotations

from pathlib import Path

from mmalign.core.store import FileResultStore
from mmalign.paths import default_configs_dir, default_results_dir, workspace_dir

_configs_dir: Path | None = None
_results_dir: Path | None = None


def configure(configs_dir: str | Path, results_dir: str | Path) -> None:
    """Called by ``mmalign serve`` to wire up paths before app startup."""
    global _configs_dir, _results_dir
    _configs_dir = Path(configs_dir)
    _results_dir = Path(results_dir)


def _effective_configs_dir() -> Path:
    if _configs_dir is not None:
        return _configs_dir
    return default_configs_dir()


def _effective_results_dir() -> Path:
    if _results_dir is not None:
        return _results_dir
    return default_results_dir()


def get_store() -> FileResultStore:
    return FileResultStore(_effective_results_dir())


def get_configs_dir() -> Path:
    return _effective_configs_dir()


def get_results_dir() -> Path:
    return _effective_results_dir()


def get_workspace_dir() -> Path:
    if _configs_dir is not None:
        return workspace_dir(_configs_dir.parent)
    return workspace_dir()
