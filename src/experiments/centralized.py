"""Task 4 — centralised baseline (the utility upper bound).

Trains the dual-head model on the pooled training data (union of all node
partitions) and evaluates on the shared held-out test set. This is the ceiling the
federated global model is compared against. Results are written to results/<name>/.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models import DualHeadLoss
from src.models.engine import evaluate, train_one_epoch

from .builders import build_model, build_optimizer
from .config import cfg_get
from .data_setup import build_data_context, centralized_loaders, compute_pos_weight
from .utils import get_device, save_json, set_seed


def run_centralized(config: dict, data_root: str | Path | None = None) -> dict:
    seed = int(cfg_get(config, "experiment.seed", 42))
    set_seed(seed)
    device = get_device(prefer_gpu=bool(cfg_get(config, "training.use_gpu", True)))

    ctx = build_data_context(config, data_root=data_root)
    train_loader, test_loader = centralized_loaders(ctx, config)

    model = build_model(config).to(device)
    optimizer = build_optimizer(config, model)
    pos_weight = None
    if bool(cfg_get(config, "training.class_balanced", False)):
        train_files = [f for files in ctx.manifest.nodes.values() for f in files]
        pos_weight = compute_pos_weight(ctx.data_entry, train_files)
    loss_fn = DualHeadLoss(
        severity_weight=float(cfg_get(config, "training.severity_weight", 1.0)),
        pos_weight=pos_weight,
    ).to(device)

    epochs = int(cfg_get(config, "training.local_epochs", cfg_get(config, "training.epochs", 1)))
    max_batches = cfg_get(config, "training.max_batches")  # None in real runs

    history = []
    for epoch in range(epochs):
        stats = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, max_batches=max_batches
        )
        stats["epoch"] = epoch
        history.append(stats)
        print(f"[centralized] epoch {epoch}: train_loss={stats['total']:.4f}")

    test_metrics = evaluate(model, test_loader, device, max_batches=max_batches)

    results = {
        "run_type": "centralized",
        "config_source": config.get("_source"),
        "seed": seed,
        "device": str(device),
        "backbone": cfg_get(config, "model.backbone", "densenet121"),
        "partition": {
            "scheme": ctx.manifest.scheme,
            "num_nodes": ctx.manifest.num_nodes,
            "alpha": ctx.manifest.alpha,
            "node_sizes": {k: len(v) for k, v in ctx.manifest.nodes.items()},
            "test_size": len(ctx.manifest.test),
        },
        "severity_thresholds": ctx.severity_thresholds,
        "train_history": history,
        "test_metrics": test_metrics,
    }

    out_dir = Path(cfg_get(config, "experiment.output_dir", "results/centralized"))
    save_json(results, out_dir / "centralized_results.json")
    print(f"[centralized] wrote results -> {out_dir / 'centralized_results.json'}")
    return results
