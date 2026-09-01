"""Feature-extractor backbones for chest-radiograph classification.

DenseNet-121 (default) or ResNet-50, both ImageNet-pretrained — the standard
choices for ChestX-ray14 in the literature. The backbone returns a pooled feature
vector; the task heads (pathology, severity) live in ``heads.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# name -> (constructor, pooled feature dimension)
_BACKBONES = {
    "densenet121": (models.densenet121, 1024),
    "resnet50": (models.resnet50, 2048),
}


class Backbone(nn.Module):
    """Wraps a torchvision CNN and exposes a pooled feature vector of ``feature_dim``."""

    def __init__(self, name: str = "densenet121", pretrained: bool = True) -> None:
        super().__init__()
        if name not in _BACKBONES:
            raise ValueError(f"unknown backbone {name!r}; choose from {list(_BACKBONES)}")
        constructor, feature_dim = _BACKBONES[name]
        weights = "DEFAULT" if pretrained else None
        base = constructor(weights=weights)

        self.name = name
        self.feature_dim = feature_dim
        if name == "densenet121":
            # densenet forward = features -> relu -> global avg pool -> flatten
            self.features = base.features
            self._forward_impl = self._densenet_forward
        else:  # resnet50: everything up to (and including) the avgpool
            self.features = nn.Sequential(*list(base.children())[:-1])
            self._forward_impl = self._resnet_forward

    def _densenet_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = F.relu(out, inplace=False)  # not in-place: Opacus DP hooks forbid view+inplace
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return torch.flatten(out, 1)

    def _resnet_forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)  # [B, 2048, 1, 1]
        return torch.flatten(out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
