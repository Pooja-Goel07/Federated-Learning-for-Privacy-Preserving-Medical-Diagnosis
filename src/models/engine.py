"""Reusable train/eval loops shared by the centralised baseline and FL clients.

Keeping these here (rather than in an experiment script) means the federated client
in Task 6 runs *exactly* the same local training step as the centralised baseline,
so comparisons are apples-to-apples.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import pathology_metrics, severity_metrics

from .losses import DualHeadLoss


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: DualHeadLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
    log_every: int | None = None,
    prox_mu: float = 0.0,
    prox_ref_params: list[torch.Tensor] | None = None,
) -> dict:
    """One local epoch. Returns average losses over the epoch.

    FedProx (Task 7): when ``prox_mu`` > 0 and ``prox_ref_params`` (a snapshot of the
    global trainable params at round start, in ``model.parameters()`` order) is given,
    a proximal term (mu/2)·||w - w_global||² is added to the loss to keep local updates
    close to the global model on heterogeneous data.
    """
    model.train()
    use_prox = prox_mu > 0.0 and prox_ref_params is not None
    totals = {"total": 0.0, "pathology": 0.0, "severity": 0.0}
    n_batches = 0
    n_sev = 0
    for b, (images, targets) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        total_loss = loss.total
        if use_prox:
            prox = sum(
                ((p - ref) ** 2).sum()
                for p, ref in zip(model.parameters(), prox_ref_params)
            )
            total_loss = total_loss + (prox_mu / 2.0) * prox
        total_loss.backward()
        optimizer.step()

        totals["total"] += float(loss.total.item())
        totals["pathology"] += float(loss.pathology.item())
        totals["severity"] += float(loss.severity.item())
        n_sev += loss.num_severity
        n_batches += 1
        if log_every and (b % log_every == 0):
            print(f"    batch {b}: loss={loss.total.item():.4f}")

    if n_batches == 0:
        return {"total": 0.0, "pathology": 0.0, "severity": 0.0, "num_batches": 0}
    avg = {k: v / n_batches for k, v in totals.items()}
    avg["num_batches"] = n_batches
    avg["num_severity_samples"] = n_sev
    return avg


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
    max_batches: int | None = None,
    loss_fn: DualHeadLoss | None = None,
) -> dict:
    """Evaluate on a loader; returns pathology + severity metrics.

    If ``loss_fn`` is given, also returns the average total loss under key 'loss'.
    """
    model.eval()
    path_logits, path_targets = [], []
    sev_logits, sev_targets = [], []
    has_path = has_sev = False
    loss_sum, loss_batches = 0.0, 0

    for b, (images, targets) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(images)
        if loss_fn is not None:
            loss_sum += float(loss_fn(outputs, targets).total.item())
            loss_batches += 1
        if "pathology" in outputs:
            has_path = True
            path_logits.append(outputs["pathology"].cpu().numpy())
            path_targets.append(targets["pathology"].cpu().numpy())
        if "severity" in outputs:
            has_sev = True
            sev_logits.append(outputs["severity"].cpu().numpy())
            sev_targets.append(targets["severity"].cpu().numpy())

    results: dict = {}
    if has_path and path_logits:
        results["pathology"] = pathology_metrics(
            np.concatenate(path_logits), np.concatenate(path_targets)
        )
    if has_sev and sev_logits:
        results["severity"] = severity_metrics(
            np.concatenate(sev_logits), np.concatenate(sev_targets)
        )
    if loss_fn is not None:
        results["loss"] = loss_sum / loss_batches if loss_batches else float("nan")
    return results
