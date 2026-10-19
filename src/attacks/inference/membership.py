"""Loss-based membership-inference audit (Task 11, Objective 7, gap G3).

A membership-inference attack tries to decide whether a given sample was in a model's
training set. The simplest, well-established form (Yeom et al.) exploits overfitting:
training ("member") samples tend to have lower loss than unseen ("non-member") samples.
We score each sample by its negative per-sample pathology loss and measure how well that
score separates members from non-members — the attack ROC-AUC. 0.5 means no membership
signal leaks (the ideal under strong privacy); higher means the model leaks membership.

Running this at several DP budgets turns "we added privacy" into "we measured the
residual leakage at a stated budget" — i.e. privacy audited, not asserted (gap G3).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


@torch.no_grad()
def per_sample_pathology_loss(
    model: nn.Module, loader: DataLoader, device, max_batches: int | None = None
) -> np.ndarray:
    """Mean BCE loss per sample over the 14 pathology labels."""
    model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    out = []
    for b, (images, targets) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        logits = model(images.to(device))["pathology"]
        t = targets["pathology"].to(device).float()
        out.append(bce(logits, t).mean(dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def membership_attack(
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device,
    max_batches: int | None = None,
) -> dict:
    """Attack ROC-AUC separating members (train) from non-members (held-out).

    Returns attack_auc (0.5 = no leakage), the mean losses of each group, and counts.
    """
    lm = per_sample_pathology_loss(model, member_loader, device, max_batches)
    ln = per_sample_pathology_loss(model, nonmember_loader, device, max_batches)
    if len(lm) == 0 or len(ln) == 0:
        return {"attack_auc": float("nan"), "n_member": int(len(lm)), "n_nonmember": int(len(ln))}
    scores = np.concatenate([-lm, -ln])           # higher score -> predicted member
    labels = np.concatenate([np.ones_like(lm), np.zeros_like(ln)])
    auc = float(roc_auc_score(labels, scores))
    # Attack advantage over the 0.5 no-leakage baseline (kept in [0, 1]).
    return {
        "attack_auc": auc,
        "attack_advantage": float(abs(auc - 0.5) * 2.0),
        "member_mean_loss": float(lm.mean()),
        "nonmember_mean_loss": float(ln.mean()),
        "n_member": int(len(lm)),
        "n_nonmember": int(len(ln)),
    }
