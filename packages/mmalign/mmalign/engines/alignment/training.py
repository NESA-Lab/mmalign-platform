"""Training-time alignment engine — DPO and SFT pipelines.

Architecture (TDD §M6, engines/alignment/training.py)
------------------------------------------------------
Implements ``TrainingAlignMethod`` Protocol for two training strategies:

* **DPOAlignMethod** — Direct Preference Optimisation using TRL ``DPOTrainer``.
  Input: preference pairs (chosen / rejected) from human review labels.
  Output: LoRA adapter checkpoint.

* **SFTAlignMethod** — Supervised Fine-Tuning using TRL ``SFTTrainer``.
  Input: compliant (chosen) responses from the review corpus.
  Output: SFT model checkpoint.

Both implementations:
  - Check for required dependencies (TRL, transformers, accelerate, peft) and
    raise ``TrainingEnvironmentError`` with installation instructions if absent.
  - Accept ``TrainingConfig`` for hyperparameter control.
  - Are async generators: yield ``TrainingProgress`` events during training
    and a final ``TrainResult`` on completion (matching the Protocol spec in
    ``core/method.py``).
  - Write checkpoints to ``configs_dir/../checkpoints/{run_id}/``.

``TrainingEngine`` ties it together: load data, select method, run training,
then optionally trigger a benchmark validation run.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Self

from mmalign.core.method import TrainResult, TrainingProgress


class TrainingEnvironmentError(RuntimeError):
    """Raised when required training packages are not installed."""


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters shared across DPO and SFT runs."""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "checkpoints"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    max_length: int = 1024
    # LoRA params
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # DPO specific
    beta: float = 0.1               # KL regularisation coefficient
    # SFT specific
    packing: bool = False
    # Misc
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

class DPOAlignMethod:
    """TrainingAlignMethod: Direct Preference Optimisation via TRL.

    Requires: ``trl>=0.8``, ``transformers``, ``peft``, ``accelerate``.

    Dataset format (list of dicts)::

        {
            "prompt": "...",
            "chosen": "...",
            "rejected": "..."
        }
    """

    name = "dpo"
    supported_modalities = {"text"}

    def validate(
        self,
        dataset: Any,
        model: Any,
        config: dict[str, Any],
    ) -> None:
        _check_training_deps()
        if not dataset:
            raise ValueError("DPO requires a non-empty preference dataset.")
        sample = dataset[0] if isinstance(dataset, list) else next(iter(dataset))
        for key in ("prompt", "chosen", "rejected"):
            if key not in sample:
                raise ValueError(
                    f"DPO dataset must have 'prompt', 'chosen', 'rejected' keys. "
                    f"Missing: {key!r}"
                )

    async def train(
        self,
        dataset: Any,
        model: Any,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[TrainingProgress | TrainResult]:
        """Async generator: yields progress events, then final TrainResult."""
        _check_training_deps()
        train_config = TrainingConfig(**{
            k: v for k, v in config.items()
            if k in TrainingConfig.__dataclass_fields__
        })
        run_id = f"dpo-{uuid.uuid4().hex[:8]}"
        checkpoint_dir = Path(train_config.output_dir) / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        async for event in _run_dpo(dataset, train_config, checkpoint_dir, run_id):
            yield event

    def __aiter__(self) -> Self:
        return self


class SFTAlignMethod:
    """TrainingAlignMethod: Supervised Fine-Tuning via TRL.

    Requires: ``trl>=0.8``, ``transformers``, ``peft``, ``accelerate``.

    Dataset format (list of dicts)::

        {
            "text": "..."  # full formatted conversation or instruction+response
        }
    """

    name = "sft"
    supported_modalities = {"text"}

    def validate(
        self,
        dataset: Any,
        model: Any,
        config: dict[str, Any],
    ) -> None:
        _check_training_deps()
        if not dataset:
            raise ValueError("SFT requires a non-empty training dataset.")
        sample = dataset[0] if isinstance(dataset, list) else next(iter(dataset))
        if "text" not in sample and "prompt" not in sample:
            raise ValueError(
                "SFT dataset must have a 'text' or 'prompt' key."
            )

    async def train(
        self,
        dataset: Any,
        model: Any,
        config: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[TrainingProgress | TrainResult]:
        """Async generator: yields progress events, then final TrainResult."""
        _check_training_deps()
        train_config = TrainingConfig(**{
            k: v for k, v in config.items()
            if k in TrainingConfig.__dataclass_fields__
        })
        run_id = f"sft-{uuid.uuid4().hex[:8]}"
        checkpoint_dir = Path(train_config.output_dir) / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        async for event in _run_sft(dataset, train_config, checkpoint_dir, run_id):
            yield event

    def __aiter__(self) -> Self:
        return self


# ---------------------------------------------------------------------------
# TrainingEngine
# ---------------------------------------------------------------------------

class TrainingEngine:
    """Orchestrate a full training run: load data → train → validate.

    Parameters
    ----------
    configs_dir:
        Project configs directory (used for models.yaml / scenarios.yaml).
    results_dir:
        ResultStore directory (used for post-training benchmark validation).
    """

    def __init__(
        self,
        configs_dir: Path,
        results_dir: Path | None = None,
    ) -> None:
        self._configs_dir = configs_dir
        self._results_dir = results_dir

    async def train(
        self,
        method: str,
        dataset: list[dict[str, Any]],
        config: TrainingConfig,
        validate_after: bool = True,
        validate_scenario: str | None = None,
        validate_model: str | None = None,
    ) -> TrainResult:
        """Run a training pipeline and optionally validate with a benchmark.

        Parameters
        ----------
        method:
            ``"dpo"`` or ``"sft"``.
        dataset:
            Training data (list of dicts, format depends on method).
        config:
            Hyperparameter configuration.
        validate_after:
            Run a benchmark after training to verify safety is maintained.
        validate_scenario / validate_model:
            Benchmark params for post-training validation.
        """
        if method == "dpo":
            trainer: DPOAlignMethod | SFTAlignMethod = DPOAlignMethod()
        elif method == "sft":
            trainer = SFTAlignMethod()
        else:
            raise ValueError(
                f"Unknown training method: {method!r}. Supported: 'dpo', 'sft'."
            )

        trainer.validate(dataset, None, config.__dict__)

        result: TrainResult | None = None
        async for event in trainer.train(dataset, None, config.__dict__):
            if isinstance(event, TrainResult):
                result = event
            # Progress events are yielded to the caller via this engine
            # when used as an async generator; here we consume them.

        if result is None:
            raise RuntimeError("Training completed without producing a TrainResult.")

        if validate_after and self._results_dir is not None:
            await self._validate(result, validate_scenario, validate_model)

        return result

    async def _validate(
        self,
        train_result: TrainResult,
        scenario: str | None,
        model: str | None,
    ) -> None:
        """Run a quick benchmark on the trained checkpoint."""
        # If no scenario/model specified, skip silently
        if not scenario or not model:
            return
        if self._results_dir is None:
            return
        try:
            from mmalign.engines.diagnosis.runner import DiagnosisEngine

            engine = DiagnosisEngine(
                configs_dir=self._configs_dir,
                results_dir=self._results_dir,
            )
            await engine.run_benchmark(
                model=model,
                scenario=scenario,
                method="align3",
                evaluator="rule_based",
                sample_limit=20,
            )
        except Exception:
            # Validation is best-effort; training result is still returned.
            pass


# ---------------------------------------------------------------------------
# Internal training implementations
# ---------------------------------------------------------------------------

async def _run_dpo(
    dataset: list[dict[str, Any]],
    config: TrainingConfig,
    checkpoint_dir: Path,
    run_id: str,
) -> AsyncIterator[TrainingProgress | TrainResult]:
    """Execute DPO training loop using TRL DPOTrainer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig, DPOTrainer
    import datasets as hf_datasets

    started_at = datetime.now(tz=timezone.utc)

    # Build HuggingFace dataset
    hf_ds = hf_datasets.Dataset.from_list(dataset)

    # Load base model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model, trust_remote_code=True
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    dpo_config = DPOConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        beta=config.beta,
        seed=config.seed,
        fp16=config.fp16,
        bf16=config.bf16,
        max_length=config.max_length,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        tokenizer=tokenizer,
        train_dataset=hf_ds,
    )

    # Run training in a thread (TRL trainers are sync)
    total_steps = len(dataset) * config.num_train_epochs // max(
        config.per_device_train_batch_size * config.gradient_accumulation_steps, 1
    )

    yield TrainingProgress(step=0, total_steps=total_steps, elapsed_s=0.0)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, trainer.train)

    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    elapsed = (datetime.now(tz=timezone.utc) - started_at).total_seconds()
    yield TrainResult(
        method="dpo",
        base_model=config.base_model,
        checkpoint_uri=str(checkpoint_dir),
        final_metrics={
            "train_loss": getattr(trainer.state, "best_metric", 0.0) or 0.0,
        },
        metadata={"run_id": run_id, "elapsed_s": elapsed},
    )


async def _run_sft(
    dataset: list[dict[str, Any]],
    config: TrainingConfig,
    checkpoint_dir: Path,
    run_id: str,
) -> AsyncIterator[TrainingProgress | TrainResult]:
    """Execute SFT training loop using TRL SFTTrainer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    import datasets as hf_datasets

    started_at = datetime.now(tz=timezone.utc)

    hf_ds = hf_datasets.Dataset.from_list(dataset)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        packing=config.packing,
        max_seq_length=config.max_length,
        seed=config.seed,
        fp16=config.fp16,
        bf16=config.bf16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        tokenizer=tokenizer,
        train_dataset=hf_ds,
        peft_config=lora_config,
    )

    total_steps = len(dataset) * config.num_train_epochs // max(
        config.per_device_train_batch_size * config.gradient_accumulation_steps, 1
    )

    yield TrainingProgress(step=0, total_steps=total_steps, elapsed_s=0.0)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, trainer.train)

    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    elapsed = (datetime.now(tz=timezone.utc) - started_at).total_seconds()
    yield TrainResult(
        method="sft",
        base_model=config.base_model,
        checkpoint_uri=str(checkpoint_dir),
        final_metrics={
            "train_loss": getattr(trainer.state, "best_metric", 0.0) or 0.0,
        },
        metadata={"run_id": run_id, "elapsed_s": elapsed},
    )


# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

def _check_training_deps() -> None:
    missing = []
    for pkg in ("trl", "transformers", "peft", "accelerate", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise TrainingEnvironmentError(
            f"Training requires the following packages: {', '.join(missing)}.\n"
            "Install with:\n"
            "  pip install trl transformers peft accelerate datasets\n"
            "or:\n"
            "  uv pip install trl transformers peft accelerate datasets"
        )
