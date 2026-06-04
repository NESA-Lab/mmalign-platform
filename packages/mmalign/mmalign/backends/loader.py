"""Backend loader — instantiates ModelBackend from configs/models.yaml entries.

Usage
-----
>>> backends = load_backends("configs/models.yaml")
>>> backend = backends["gpt-4.1"]
>>> result = await backend.generate(request)

The loader reads ``api_key_env`` and resolves the key from the environment.
Missing environment variables result in a ``placeholder`` string being used;
the backend will fail on the first real API call, not at load time, so that
test runs using the ``mock`` backend don't require live credentials.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from mmalign.core.backend import ModelBackend
from mmalign.core.types import BackendCapabilities
from .mock import MockBackend
from .openai import OpenAIBackend


class UnknownProviderError(ValueError):
    """Raised when a YAML entry specifies an unsupported provider."""


def _parse_capabilities(raw: dict[str, Any] | None) -> BackendCapabilities | None:
    if not raw:
        return None
    return BackendCapabilities(
        input_modalities=set(raw.get("input_modalities", ["text"])),
        output_modalities=set(raw.get("output_modalities", ["text"])),
        embed_modalities=set(raw.get("embed_modalities", ["text"])),
        score_modalities=set(raw.get("score_modalities", [])),
        supports_system_message=raw.get("supports_system_message", True),
        supports_developer_message=raw.get("supports_developer_message", False),
        openai_compatible=raw.get("openai_compatible", False),
    )


def load_backends(
    config_path: str | Path,
    *,
    only: list[str] | None = None,
) -> dict[str, ModelBackend]:
    """Load all (or a subset of) backends from a models.yaml file.

    Parameters
    ----------
    config_path:
        Path to the ``models.yaml`` configuration file.
    only:
        If given, only load backends with these IDs.

    Returns
    -------
    dict[str, ModelBackend]
        Mapping from model ID to instantiated backend.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    models: dict[str, Any] = raw.get("models", {})

    result: dict[str, ModelBackend] = {}
    for model_id, cfg in models.items():
        if only is not None and model_id not in only:
            continue
        result[model_id] = _build_backend(model_id, cfg)
    return result


def _build_backend(model_id: str, cfg: dict[str, Any]) -> ModelBackend:
    provider: str = cfg.get("provider", "mock")
    capabilities = _parse_capabilities(cfg.get("capabilities"))

    if provider == "mock":
        return MockBackend(model_id=model_id)

    if provider in ("openai", "azure_openai", "openai_compatible"):
        api_key_env = cfg.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "placeholder") if api_key_env else "placeholder"
        return OpenAIBackend(
            model_id=model_id,
            model_name=cfg.get("model_name", model_id),
            api_style=cfg.get("api_style", "chat_completions"),
            base_url=cfg.get("base_url", "https://api.openai.com/v1"),
            api_key=api_key,
            timeout_s=float(cfg.get("timeout_s", 60)),
            max_retries=int(cfg.get("max_retries", 2)),
            capabilities=capabilities,
        )

    raise UnknownProviderError(
        f"Unknown provider '{provider}' for model '{model_id}'. "
        f"Supported: openai, azure_openai, openai_compatible, mock."
    )
