"""Byzantine / poisoning attack clients (Task 9).

Each subclasses the honest ``FLClient`` and corrupts either its training data or the
update it returns, so the robust aggregation rules can be stress-tested:

  * label_flip        — trains on flipped pathology labels (data poisoning)
  * sign_flip         — trains honestly, then negates its update (2*global - local)
  * scaled_update     — trains honestly, then blows up its update by a large factor
  * model_replacement — trains on flipped labels, then boosts that poisoned update
                        by the client count to try to overwrite the global model

Updates are absolute parameters (Flower convention); the "update" is (local-global),
recovered here since the client is handed the global parameters at fit time.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.federated.client import FLClient
from src.federated.params import get_parameters, set_parameters

ATTACK_TYPES = {"label_flip", "sign_flip", "scaled_update", "model_replacement"}


class ByzantineClient(FLClient):
    def __init__(
        self,
        node_id: str,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer],
        attack_type: str,
        num_clients: int,
        scale: float = 10.0,
        device: torch.device | str = "cpu",
        local_epochs: int = 1,
        severity_weight: float = 1.0,
        max_batches: int | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        if attack_type not in ATTACK_TYPES:
            raise ValueError(f"unknown attack {attack_type!r}; choose from {sorted(ATTACK_TYPES)}")
        super().__init__(
            node_id=node_id, model=model, train_loader=train_loader,
            optimizer_fn=optimizer_fn, device=device, local_epochs=local_epochs,
            severity_weight=severity_weight, max_batches=max_batches,
            pos_weight=pos_weight,
        )
        self.attack_type = attack_type
        self.num_clients = num_clients
        self.scale = scale

    def _epoch(self, optimizer: torch.optim.Optimizer, flip: bool) -> dict:
        self.model.train()
        total, nb = 0.0, 0
        for b, (images, targets) in enumerate(self.train_loader):
            if self.max_batches is not None and b >= self.max_batches:
                break
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            if flip:  # multi-label pathology flip: 0<->1
                targets["pathology"] = 1.0 - targets["pathology"]
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(images), targets)
            loss.total.backward()
            optimizer.step()
            total += float(loss.total.item())
            nb += 1
        return {"total": total / nb if nb else 0.0}

    def _train(self, parameters: list[np.ndarray], flip: bool) -> tuple[list[np.ndarray], dict]:
        set_parameters(self.model, parameters, self.exclude_keys)
        optimizer = self.optimizer_fn(self.model)
        last: dict = {}
        for _ in range(self.local_epochs):
            last = self._epoch(optimizer, flip)
        return get_parameters(self.model, self.exclude_keys), last

    def fit(self, parameters: list[np.ndarray], config: dict | None = None):
        g = parameters  # global params handed to this client this round
        if self.attack_type == "label_flip":
            new, last = self._train(g, flip=True)
        elif self.attack_type == "sign_flip":
            local, last = self._train(g, flip=False)
            new = [2.0 * gi - li for gi, li in zip(g, local)]  # negate the update
        elif self.attack_type == "scaled_update":
            local, last = self._train(g, flip=False)
            new = [gi + self.scale * (li - gi) for gi, li in zip(g, local)]
        else:  # model_replacement
            local, last = self._train(g, flip=True)
            new = [gi + self.num_clients * (li - gi) for gi, li in zip(g, local)]

        num_examples = self._num_train_examples()
        metrics = {
            "node_id": self.node_id,
            "attack": self.attack_type,
            "train_loss": float(last.get("total", float("nan"))),
        }
        return new, num_examples, metrics
