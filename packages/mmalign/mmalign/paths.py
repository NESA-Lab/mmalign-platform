"""Workspace path resolution — ``data/`` is the platform working directory.

Layout (dev and deploy share the same shape)::

    <workspace>/
    ├── benchmarks/
    │   ├── builtin/    # self-contained packages + metadata.yaml
    │   └── user/       # user-uploaded packages
    └── results/        # ResultStore (runs/, samples/, reviews/, index.json)

Environment variables
---------------------
MMALIGN_CONFIGS_DIR   Override configs directory (default: ``<project>/configs``).
MMALIGN_DATA_DIR      Override workspace root (default: ``<project>/data`` or XDG).
MMALIGN_RESULTS_DIR   Override ResultStore root only (default: ``<workspace>/results``).
MMALIGN_USE_XDG       When ``1``/``true``, workspace defaults to
                      ``$XDG_DATA_HOME/mmalign/data/`` when ``MMALIGN_DATA_DIR`` unset.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until ``configs/models.yaml`` exists."""
    cwd = (start or Path.cwd()).resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "configs" / "models.yaml").is_file():
            return candidate
    return cwd


def xdg_workspace_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "mmalign" / "data"


def workspace_dir(start: Path | None = None) -> Path:
    """Platform working directory containing ``benchmarks/`` and ``results/``."""
    if start is not None:
        return _resolve_workspace(start)
    env = os.environ.get("MMALIGN_DATA_DIR")
    if env:
        return Path(env).resolve()
    if os.environ.get("MMALIGN_USE_XDG", "").lower() in ("1", "true", "yes"):
        return xdg_workspace_dir()
    return find_project_root() / "data"


def default_configs_dir(start: Path | None = None) -> Path:
    env = os.environ.get("MMALIGN_CONFIGS_DIR")
    if env:
        return Path(env).resolve()
    return find_project_root(start) / "configs"


def default_data_dir(start: Path | None = None) -> Path:
    """Alias for :func:`workspace_dir` (historical name)."""
    return workspace_dir(start)


def default_results_dir(start: Path | None = None) -> Path:
    """ResultStore root — ``<workspace>/results`` unless overridden."""
    env = os.environ.get("MMALIGN_RESULTS_DIR")
    if env:
        return Path(env).resolve()
    root = start.resolve() if start is not None else None
    if root is not None and root.name != "data" and (root / "data").is_dir():
        return root / "data" / "results"
    return workspace_dir(start) / "results"


def default_benchmarks_dir(start: Path | None = None) -> Path:
    return _resolve_workspace(start) / "benchmarks"


def _resolve_workspace(start: Path | None) -> Path:
    """*start* may be project root or ``data/`` workspace root."""
    if start is None:
        return workspace_dir()
    env = os.environ.get("MMALIGN_DATA_DIR")
    if env:
        return Path(env).resolve()
    if os.environ.get("MMALIGN_USE_XDG", "").lower() in ("1", "true", "yes"):
        return xdg_workspace_dir()
    p = start.resolve()
    if p.name == "data":
        return p
    if (p / "data" / "benchmarks").is_dir():
        return p / "data"
    if (p / "benchmarks").is_dir() and p.name == "data":
        return p
    return find_project_root(p) / "data"


def builtin_benchmarks_dir(start: Path | None = None) -> Path:
    return _resolve_workspace(start) / "benchmarks" / "builtin"


def user_benchmarks_dir(start: Path | None = None) -> Path:
    return _resolve_workspace(start) / "benchmarks" / "user"
