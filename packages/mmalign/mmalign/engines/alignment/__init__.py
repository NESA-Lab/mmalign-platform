"""Alignment engine — inference-time and training-time alignment.

Architecture (TDD §Application Layer → engines/alignment/)
----------------------------------------------------------
    InferenceAlignmentEngine  wrap InferenceAlignMethod + backend → single aligned response
    DPOAlignMethod            TrainingAlignMethod: DPO fine-tune via TRL
    SFTAlignMethod            TrainingAlignMethod: SFT fine-tune via TRL
    TrainingEngine            orchestrate training runs, write checkpoints
"""

from .inference import InferenceAlignmentEngine
from .training import DPOAlignMethod, SFTAlignMethod, TrainingEngine, TrainingConfig

__all__ = [
    "InferenceAlignmentEngine",
    "DPOAlignMethod",
    "SFTAlignMethod",
    "TrainingEngine",
    "TrainingConfig",
]
