"""MMAlign — Multimodal Value Alignment Platform."""

__version__ = "0.1.0"

from mmalign.engines.correction.controller import ComplianceController, ControlledResponse

__all__ = ["ComplianceController", "ControlledResponse", "__version__"]
