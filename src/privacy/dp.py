"""Client-level differential privacy via Opacus (Task 8).

Wraps a client's local training step with DP-SGD: per-sample gradient clipping to a
max L2 norm ``C`` plus calibrated Gaussian noise (std = ``noise_multiplier`` * C),
and tracks the cumulative (epsilon, delta) budget with a Renyi-DP (RDP) accountant.

Two structural requirements Opacus imposes, handled here:
  * BatchNorm mixes samples within a batch, so it is not DP-compatible. We replace
    all BN layers with GroupNorm via ``ModuleValidator.fix`` (``ensure_dp_compatible``).
    Consequence: DP runs are mutually exclusive with FedBN, and every model in a DP
    run (all clients + the global model) must be built the same way so parameter
    keys line up for aggregation.
  * The loss must keep the batch dimension intact for per-sample gradients — the
    dual-head loss already uses ``ignore_index`` for severity rather than boolean
    indexing (see src/models/losses.py).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from src.models.losses import DualHeadLoss

# Benign, expected under DP-SGD here — silence to keep run output readable:
#  * "Secure RNG turned off": we use fast RNG for experimentation (set secure_mode
#    for a production run); documented in study.md.
#  * "Full backward hook is firing ...": from the zero-weighted severity term on
#    all-unlabelled batches; does not affect the DP gradients.
warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
warnings.filterwarnings("ignore", message=".*Full backward hook is firing.*")


@dataclass
class DPConfig:
    enabled: bool = False
    max_grad_norm: float = 1.0        # per-sample clip bound C
    noise_multiplier: float = 1.0     # sigma (noise std = sigma * C)
    target_delta: float = 1e-5        # delta for the (eps, delta) report
    accountant: str = "rdp"           # Renyi-DP accountant

    @staticmethod
    def from_config(config: dict) -> "DPConfig":
        from src.experiments.config import cfg_get

        return DPConfig(
            enabled=bool(cfg_get(config, "privacy.dp.enabled", False)),
            max_grad_norm=float(cfg_get(config, "privacy.dp.max_grad_norm", 1.0)),
            noise_multiplier=float(cfg_get(config, "privacy.dp.noise_multiplier", 1.0)),
            target_delta=float(cfg_get(config, "privacy.dp.target_delta", 1e-5)),
            accountant=str(cfg_get(config, "privacy.dp.accountant", "rdp")),
        )


def ensure_dp_compatible(model: torch.nn.Module) -> torch.nn.Module:
    """Make a model DP-compatible if needed.

    Two fixes: (1) replace BatchNorm with GroupNorm (``ModuleValidator.fix``), and
    (2) disable in-place activations (e.g. densenet's ``ReLU(inplace=True)``) which
    Opacus's backward hooks forbid (view + in-place breaks per-sample grads).
    Deterministic given the same architecture, so all models in a run stay aligned.
    """
    if ModuleValidator.validate(model, strict=False):
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=True)
    for module in model.modules():
        if getattr(module, "inplace", False):
            module.inplace = False
    return model


def make_private(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    dp: DPConfig,
) -> tuple[PrivacyEngine, torch.nn.Module, torch.optim.Optimizer, DataLoader]:
    """Attach a PrivacyEngine. Returns (engine, dp_model, dp_optimizer, dp_loader).

    ``dp_model`` shares parameters with ``model`` (it wraps it), so loading global
    parameters into ``model`` between rounds is seen by ``dp_model``.
    """
    engine = PrivacyEngine(accountant=dp.accountant)
    dp_model, dp_optimizer, dp_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp.noise_multiplier,
        max_grad_norm=dp.max_grad_norm,
    )
    return engine, dp_model, dp_optimizer, dp_loader


def get_epsilon(engine: PrivacyEngine, delta: float) -> float:
    """Cumulative epsilon spent so far, at the given delta."""
    try:
        return float(engine.get_epsilon(delta))
    except Exception:
        # Accountant has no steps yet (before any optimizer.step()).
        return float("inf")


def dp_train_one_epoch(
    dp_model: torch.nn.Module,
    dp_loader: DataLoader,
    loss_fn: DualHeadLoss,
    dp_optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
) -> dict:
    """One local epoch under DP-SGD (clipping + noise handled by the dp_optimizer)."""
    dp_model.train()
    total, n_batches = 0.0, 0
    for b, (images, targets) in enumerate(dp_loader):
        if max_batches is not None and b >= max_batches:
            break
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        dp_optimizer.zero_grad(set_to_none=True)
        outputs = dp_model(images)
        loss = loss_fn(outputs, targets)
        loss.total.backward()
        dp_optimizer.step()  # per-sample clip + Gaussian noise applied here
        total += float(loss.total.item())
        n_batches += 1
    return {"total": total / n_batches if n_batches else 0.0, "num_batches": n_batches}
