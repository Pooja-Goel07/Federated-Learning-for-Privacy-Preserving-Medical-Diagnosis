"""DP-enabled federated client (Task 8).

Subclasses ``FLClient`` and replaces the local training step with DP-SGD (Opacus).
The PrivacyEngine is created once per client and reused across rounds so the RDP
accountant accumulates the true cumulative (epsilon, delta) budget over the whole
federated run. The reported ``epsilon`` rides along in the client's fit metrics.

DP is mutually exclusive with FedProx/FedBN (per-sample gradients + GroupNorm), so
those are forced off here; the model passed in must already be DP-compatible
(``ensure_dp_compatible``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.federated.client import FLClient
from src.federated.params import get_parameters, set_parameters

from .dp import DPConfig, dp_train_one_epoch, get_epsilon, make_private


class DPFLClient(FLClient):
    def __init__(
        self,
        node_id: str,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer],
        dp: DPConfig,
        device: torch.device | str = "cpu",
        local_epochs: int = 1,
        severity_weight: float = 1.0,
        max_batches: int | None = None,
        eval_loader: DataLoader | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            model=model,
            train_loader=train_loader,
            optimizer_fn=optimizer_fn,
            device=device,
            local_epochs=local_epochs,
            severity_weight=severity_weight,
            max_batches=max_batches,
            eval_loader=eval_loader,
            fedprox_mu=0.0,  # DP is incompatible with FedProx per-sample grads
            fedbn=False,     # DP replaces BN with GroupNorm
        )
        self.dp = dp
        self._engine = None
        self._dp_model = None
        self._dp_optimizer = None
        self._dp_loader = None

    def _ensure_private(self) -> None:
        if self._engine is None:
            optimizer = self.optimizer_fn(self.model)
            self._engine, self._dp_model, self._dp_optimizer, self._dp_loader = make_private(
                self.model, optimizer, self.train_loader, self.dp
            )

    def fit(self, parameters: list[np.ndarray], config: dict | None = None):
        set_parameters(self.model, parameters, self.exclude_keys)
        self._ensure_private()
        last = {}
        for _ in range(self.local_epochs):
            last = dp_train_one_epoch(
                self._dp_model, self._dp_loader, self.loss_fn, self._dp_optimizer,
                self.device, max_batches=self.max_batches,
            )
        epsilon = get_epsilon(self._engine, self.dp.target_delta)
        num_examples = self._num_train_examples()
        metrics = {
            "node_id": self.node_id,
            "train_loss": float(last.get("total", float("nan"))),
            "epsilon": epsilon,
        }
        return get_parameters(self.model, self.exclude_keys), num_examples, metrics
