"""Shared model / optimizer builders used by centralized, single-site and FL runs.

Keeping these in one place guarantees every run type instantiates the *same*
architecture and optimizer from the *same* config keys, so floor (single-site),
ceiling (centralized) and federated results are directly comparable.
"""

from __future__ import annotations

import torch

from src.models import DualHeadNet, ModelConfig

from .config import cfg_get


def build_model(config: dict) -> DualHeadNet:
    heads = cfg_get(config, "model.heads", {}) or {}
    model_cfg = ModelConfig(
        backbone=cfg_get(config, "model.backbone", "densenet121"),
        pretrained=bool(cfg_get(config, "model.pretrained", True)),
        dropout=float(cfg_get(config, "model.dropout", 0.0)),
        pathology_head=bool(heads.get("pathology", True)),
        severity_head=bool(heads.get("severity", True)),
    )
    return DualHeadNet(model_cfg)


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    name = cfg_get(config, "training.optimizer", "adam").lower()
    lr = float(cfg_get(config, "training.lr", 1e-4))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer {name!r}")
