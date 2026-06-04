#!/usr/bin/env python3
"""Seed sample data into the FileResultStore.

Thin wrapper around :func:`mmalign.mock.generate` (the canonical generator,
also exposed as ``mmalign seed``).  Kept for backward compatibility.

Usage
-----
    uv run python scripts/seed_store.py
    uv run python scripts/seed_store.py --root /tmp/mmalign-test --scale small --reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from mmalign.mock import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed mock benchmark data into FileResultStore")
    parser.add_argument("--root", default=None, help="ResultStore root (default: workspace results dir)")
    parser.add_argument("--scale", default="large", choices=["large", "small"], help="Dataset size")
    parser.add_argument("--reset", action="store_true", help="Wipe the store before generating")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    report = generate(scale=args.scale, reset=args.reset, seed=args.seed, results_dir=args.root)
    d = report.as_dict()
    print(
        f"Done — {d['runs']} runs, {d['samples']} samples, {d['reviews']} reviews, "
        f"{d['benchmark_prompts']} prompts at {d['root']}"
    )


if __name__ == "__main__":
    main()
