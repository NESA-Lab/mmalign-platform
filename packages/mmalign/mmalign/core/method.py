"""InferenceAlignMethod and TrainingAlignMethod Protocols.

Design decisions (recorded in TDD §关键设计决策):
- The two protocols are deliberately separate: inference-time methods receive
  a single request and return a response; training-time methods receive a
  dataset and return a checkpoint.  Their signatures are semantically
  incompatible and cannot share a single Protocol.
- ``validate()`` is called once by Runner / Controller before bulk dispatch
  so that modality and dependency mismatches are caught early.
- ``AlignContext`` carries per-run dependency injections (vanilla result path
  for align3, reward model backend for best_of_n/tpo) as typed fields,
  replacing the fragile ``**kwargs`` string convention used in SpecBench.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .types import GenerationRequest, GenerationResult, OutputContentPart, Modality
from .scenario import Scenario

if TYPE_CHECKING:
    from .backend import ModelBackend  # noqa: F401 — for type checkers only


# ---------------------------------------------------------------------------
# Shared response types
# ---------------------------------------------------------------------------

class AlignedResponse(BaseModel):
    """Return value of a single InferenceAlignMethod.align() call.

    ``output`` follows the OpenAI OutputContentPart schema; MVP only produces
    ``output_text`` and ``refusal`` parts.
    ``thought`` captures intermediate reasoning chains (e.g. align3 four-stage
    deliberation, self_refine iteration history) for debugging and tracing.
    ``attempts`` records iteration count for methods that run multiple rounds
    (self_refine, tpo).
    """
    output: list[OutputContentPart]
    method: str
    thought: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    attempts: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    def text(self) -> str:
        """Return concatenated output_text across all output parts."""
        return "".join(
            p.text for p in self.output if p.type == "output_text" and p.text
        )

    def is_refusal(self) -> bool:
        return any(p.type == "refusal" for p in self.output)


class AlignContext(BaseModel):
    """Explicit dependency injection container for alignment methods.

    Populated by Runner / Controller before bulk dispatch.  Using a typed
    model instead of ``**kwargs`` ensures that:
    - align3 receives ``vanilla_results_path`` as a typed ``str | None``
    - best_of_n / tpo receive ``reward_model`` as a typed ``ModelBackend | None``
    - Future dependencies can be added without changing Protocol signatures
    """
    vanilla_results_path: str | None = None
    """Path to vanilla method evaluation results.  Required by align3 to focus
    on previously violated specifications only."""

    reward_model: Any = None
    """Reward model backend (ModelBackend | None).  Required by best_of_n and tpo.
    Typed as Any to avoid forward-reference issues at Pydantic model construction time;
    callers should pass a ModelBackend instance or None."""

    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class TrainResult(BaseModel):
    """Return value of a single TrainingAlignMethod.train() call."""
    method: str
    base_model: str
    checkpoint_uri: str
    final_metrics: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainingProgress(BaseModel):
    """Intermediate progress event emitted during training."""
    step: int
    total_steps: int | None = None
    loss: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    elapsed_s: float = 0.0


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

class IncompatibleMethodError(ValueError):
    """Raised by validate() when a method cannot be applied to the given
    scenario or backend."""


@runtime_checkable
class InferenceAlignMethod(Protocol):
    """Inference-time alignment algorithm (TTD / Align3 / SelfRefine etc.).

    Does NOT carry online compliance-control semantics; that is the
    responsibility of ``ControlStrategy``.

    Lifecycle
    ---------
    1. ``__init__(**method_config)`` — instantiated by Registry from
       ``configs/methods.yaml`` entries.
    2. ``validate(scenario, backend)`` — called once by Runner / Controller
       before bulk dispatch; raises ``IncompatibleMethodError`` on mismatch.
    3. ``align(...)`` — called once per sample.
    """

    name: str
    supported_modalities: set[Modality]

    def validate(self, scenario: Scenario, backend: "ModelBackend") -> None:
        """Pre-flight compatibility check.

        Implementations should verify:
        - ``scenario.modality`` is in ``supported_modalities``
        - Required ``AlignContext`` fields are satisfiable by the given backend
        - Backend capabilities match method requirements

        Raises ``IncompatibleMethodError`` with a descriptive message on failure.
        """
        ...

    async def align(
        self,
        request: GenerationRequest,
        model: "ModelBackend",
        scenario: Scenario,
        context: AlignContext | None = None,
        **kwargs: Any,
    ) -> AlignedResponse: ...


@runtime_checkable
class TrainingAlignMethod(Protocol):
    """Training-time alignment method (DPO / SFT / LoRA etc.).

    ``train`` is an async generator that yields ``TrainingProgress`` events
    during training and a final ``TrainResult`` when complete, allowing CLI
    progress bars and API SSE streams to consume the same async iterator.
    """

    name: str
    supported_modalities: set[Modality]

    def validate(
        self,
        dataset: Any,
        model: "ModelBackend",
        config: dict[str, Any],
    ) -> None: ...

    async def train(
        self,
        dataset: Any,
        model: "ModelBackend",
        config: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[TrainingProgress | TrainResult]: ...
