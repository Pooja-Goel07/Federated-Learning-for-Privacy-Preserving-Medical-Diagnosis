"""Task heads for the dual-objective model.

Two heads sit on top of the shared backbone feature vector:
  * PathologyHead : 14 independent logits -> multi-label detection (BCE loss).
  * SeverityHead  : 3 logits for ordinal grading (mild/moderate/severe). We treat
                    severity as 3-way classification but evaluate it ordinally
                    (MAE / quadratic-weighted-kappa); the ordering is respected by
                    the metrics, and the head stays simple and stable to train.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.constants import NUM_PATHOLOGIES, SEVERITY_CLASSES


class PathologyHead(nn.Module):
    def __init__(self, in_features: int, num_labels: int = NUM_PATHOLOGIES, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_labels)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(feats))  # raw logits [B, 14]


class SeverityHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = len(SEVERITY_CLASSES), dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(feats))  # raw logits [B, 3]
