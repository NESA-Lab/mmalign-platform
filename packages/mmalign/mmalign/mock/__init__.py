"""Mock data generation — a single command that populates the platform with
realistic, file-backed evaluation data.

Everything the dashboard renders (overview metrics, heatmap, controller
time-series, drift, consistency, review agreement) derives from the corpus
written here: benchmark packages, ``BenchmarkRun`` / ``SampleResult`` /
``HumanReviewLabel`` records in the ``FileResultStore``, and ``platform_stats``.
No endpoint invents its own numbers.

Entry point
-----------
>>> from mmalign.mock import generate
>>> generate(scale="large", reset=True)
"""

from __future__ import annotations

from .generator import GenerationReport, generate

__all__ = ["generate", "GenerationReport"]
