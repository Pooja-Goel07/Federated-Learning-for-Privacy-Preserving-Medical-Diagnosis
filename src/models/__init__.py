"""Backbone + dual-head (pathology + severity) architecture.

Public API:
    Backbone                              -- backbone.py
    PathologyHead, SeverityHead           -- heads.py
    DualHeadNet, ModelConfig              -- model.py
    DualHeadLoss, LossOutput              -- losses.py
"""

from __future__ import annotations

from .backbone import Backbone
from .heads import PathologyHead, SeverityHead
from .losses import DualHeadLoss, LossOutput
from .model import DualHeadNet, ModelConfig

__all__ = [
    "Backbone",
    "PathologyHead",
    "SeverityHead",
    "DualHeadNet",
    "ModelConfig",
    "DualHeadLoss",
    "LossOutput",
]
