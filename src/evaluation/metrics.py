"""Metrics for the dual-head model.

Pathology (multi-label): per-class ROC-AUC + mean AUC, and mean per-label accuracy
at a 0.5 threshold. Classes with only one label present in y_true have undefined
AUC and are skipped in the mean.

Severity (ordinal 3-class): mean absolute error (in class steps) and quadratic
weighted kappa (QWK) — both respect the mild<moderate<severe ordering. Computed
only over samples that carry a severity proxy.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from src.data.constants import PATHOLOGIES, SEVERITY_IGNORE_INDEX


def pathology_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    """logits/targets: (N, 14). Returns per-class + mean AUC and mean accuracy."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    per_class_auc: dict[str, float] = {}
    aucs: list[float] = []
    for i, name in enumerate(PATHOLOGIES):
        y = targets[:, i]
        if y.min() == y.max():  # only one class present -> AUC undefined
            per_class_auc[name] = float("nan")
            continue
        auc = float(roc_auc_score(y, probs[:, i]))
        per_class_auc[name] = auc
        aucs.append(auc)
    preds = (probs >= 0.5).astype(np.int64)
    mean_acc = float((preds == targets).mean())
    return {
        "per_pathology_auc": per_class_auc,
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "mean_accuracy": mean_acc,
    }


def severity_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    """logits: (N, 3), targets: (N,) with SEVERITY_IGNORE_INDEX for unlabelled."""
    mask = targets != SEVERITY_IGNORE_INDEX
    n = int(mask.sum())
    if n == 0:
        return {"severity_mae": float("nan"), "severity_qwk": float("nan"), "n": 0}
    preds = logits[mask].argmax(axis=1)
    y = targets[mask]
    mae = float(np.abs(preds - y).mean())
    # QWK needs >=2 distinct labels present; else undefined.
    if len(np.unique(y)) < 2:
        qwk = float("nan")
    else:
        qwk = float(cohen_kappa_score(y, preds, weights="quadratic", labels=[0, 1, 2]))
    return {"severity_mae": mae, "severity_qwk": qwk, "n": n}
