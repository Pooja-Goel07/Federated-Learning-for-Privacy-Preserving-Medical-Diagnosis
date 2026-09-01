"""Flower client for a simulated hospital node.

Subclasses Flower's ``NumPyClient`` so it works with both Flower's own simulation
engine and our in-process orchestrator (``simulation.py``). ``fit`` runs the exact
same local training step as the centralized/single-site baselines (via the shared
engine), so federated results are comparable. DP wrapping (Task 8) and Byzantine
behaviours (Task 9) will subclass / decorate this client.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from flwr.client import NumPyClient
from torch.utils.data import DataLoader

from src.models import DualHeadLoss
from src.models.engine import evaluate, train_one_epoch

from .params import bn_state_keys, get_parameters, set_parameters


class FLClient(NumPyClient):
    def __init__(
        self,
        node_id: str,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer],
        device: torch.device | str = "cpu",
        local_epochs: int = 1,
        severity_weight: float = 1.0,
        max_batches: int | None = None,
        eval_loader: DataLoader | None = None,
        fedprox_mu: float = 0.0,
        fedbn: bool = False,
    ) -> None:
        self.node_id = node_id
        self.model = model
        self.train_loader = train_loader
        self.optimizer_fn = optimizer_fn
        self.device = device
        self.local_epochs = local_epochs
        self.max_batches = max_batches
        self.eval_loader = eval_loader
        self.fedprox_mu = fedprox_mu
        # FedBN: BN weights/buffers stay local (excluded from send/aggregate/receive).
        self.exclude_keys = bn_state_keys(model) if fedbn else set()
        self.loss_fn = DualHeadLoss(severity_weight=severity_weight).to(device)
        self.model.to(device)

    # ---- Flower NumPyClient interface ----
    def get_parameters(self, config: dict | None = None) -> list[np.ndarray]:
        return get_parameters(self.model, self.exclude_keys)

    def fit(self, parameters: list[np.ndarray], config: dict | None = None):
        set_parameters(self.model, parameters, self.exclude_keys)
        # FedProx: snapshot the global trainable params at round start for the prox term.
        prox_ref = (
            [p.detach().clone() for p in self.model.parameters()]
            if self.fedprox_mu > 0.0
            else None
        )
        optimizer = self.optimizer_fn(self.model)
        last = {}
        for _ in range(self.local_epochs):
            last = train_one_epoch(
                self.model, self.train_loader, self.loss_fn, optimizer,
                self.device, max_batches=self.max_batches,
                prox_mu=self.fedprox_mu, prox_ref_params=prox_ref,
            )
        num_examples = self._num_train_examples()
        metrics = {
            "node_id": self.node_id,
            "train_loss": float(last.get("total", float("nan"))),
        }
        return get_parameters(self.model, self.exclude_keys), num_examples, metrics

    def evaluate(self, parameters: list[np.ndarray], config: dict | None = None):
        set_parameters(self.model, parameters, self.exclude_keys)
        loader = self.eval_loader or self.train_loader
        results = evaluate(
            self.model, loader, self.device, max_batches=self.max_batches, loss_fn=self.loss_fn
        )
        loss = float(results.get("loss", float("nan")))
        num_examples = len(loader.dataset)
        mean_auc = results.get("pathology", {}).get("mean_auc", float("nan"))
        return loss, num_examples, {"node_id": self.node_id, "mean_auc": mean_auc}

    def _num_train_examples(self) -> int:
        if self.max_batches is not None:
            bs = self.train_loader.batch_size or 1
            return min(len(self.train_loader.dataset), self.max_batches * bs)
        return len(self.train_loader.dataset)
