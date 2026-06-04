"""Model backend implementations.

The ``ModelBackend`` Protocol lives in ``mmalign.core.backend``.
This package provides concrete adapters.

Available backends:
  ``mock``             Deterministic stub, no network calls, for unit tests.
  ``openai``           OpenAI / Azure OpenAI / OpenAI-compatible HTTP adapter.
"""

from .mock import MockBackend
from .openai import OpenAIBackend

__all__ = ["MockBackend", "OpenAIBackend"]
