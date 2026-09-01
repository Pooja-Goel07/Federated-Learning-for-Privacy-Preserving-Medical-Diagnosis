"""Dual-head model: shared backbone + pathology and severity heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.data.constants import NUM_PATHOLOGIES, SEVERITY_CLASSES

from .backbone import Backbone
from .heads import PathologyHead, SeverityHead


@dataclass
class ModelConfig:
    backbone: str = "densenet121"  # 'densenet121' | 'resnet50'
    pretrained: bool = True
    dropout: float = 0.0
    pathology_head: bool = True
    severity_head: bool = True


class DualHeadNet(nn.Module):
    """Backbone + {pathology, severity} heads.

    forward(x) -> dict with keys among {'pathology', 'severity'} (only the enabled
    heads), each holding raw logits. Losses/metrics live in ``losses.py`` /
    ``src.evaluation``.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = Backbone(self.config.backbone, pretrained=self.config.pretrained)
        dim = self.backbone.feature_dim

        self.pathology_head = (
            PathologyHead(dim, NUM_PATHOLOGIES, self.config.dropout)
            if self.config.pathology_head
            else None
        )
        self.severity_head = (
            SeverityHead(dim, len(SEVERITY_CLASSES), self.config.dropout)
            if self.config.severity_head
            else None
        )
        if self.pathology_head is None and self.severity_head is None:
            raise ValueError("at least one head must be enabled")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        out: dict[str, torch.Tensor] = {}
        if self.pathology_head is not None:
            out["pathology"] = self.pathology_head(feats)
        if self.severity_head is not None:
            out["severity"] = self.severity_head(feats)
        return out
