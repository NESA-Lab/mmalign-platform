"""RawMethod — baseline: call backend.generate() with no alignment."""

from __future__ import annotations

import uuid
from typing import Any

from mmalign.core.backend import ModelBackend, IncompatibleBackendError
from mmalign.core.method import AlignContext, AlignedResponse, IncompatibleMethodError
from mmalign.core.scenario import Scenario
from mmalign.core.types import GenerationRequest, Modality


class RawMethod:
    """Baseline method: forward the request unchanged to the backend."""

    name = "raw"
    supported_modalities: set[Modality] = {"text", "image", "video", "audio"}

    def validate(self, scenario: Scenario, backend: ModelBackend) -> None:
        modality = scenario.modality
        if modality == "multimodal":
            return
        if modality not in backend.capabilities.input_modalities:
            raise IncompatibleMethodError(
                f"raw: backend '{backend.model_id}' does not support "
                f"input modality '{modality}'."
            )

    async def align(
        self,
        request: GenerationRequest,
        model: ModelBackend,
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse:
        result = await model.generate(request, **kwargs)
        return AlignedResponse(
            output=result.output[0].content if result.output else [],
            method=self.name,
            usage=result.usage,
        )
