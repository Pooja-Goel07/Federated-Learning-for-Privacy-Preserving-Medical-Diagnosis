"""Task 5 — isolated single-site baselines (the utility floor).

Trains a *separate* model on each node's local partition only (no federation) and
evaluates every one on the SAME shared held-out test set used by the centralized
(Task 4) and federated (Task 6) runs. This is the floor the federated global model
must beat: it quantifies what each hospital could achieve alone with its own data.

Run for both IID and non-IID partitions (via the config's partition.scheme) to show
how data isolation hurts more under heterogeneity.
"""

from __future__ import annotations

from pathlib import Path

from src.models import DualHeadLoss
from src.models.engine import evaluate, train_one_epoch

from .builders import build_model, build_optimizer
from .config import cfg_get
from .data_setup import build_data_context, node_loader, test_loader
from .utils import get_device, save_json, set_seed


def run_single_site(config: dict, data_root: str | Path | None = None) -> dict:
    seed = int(cfg_get(config, "experiment.seed", 42))
    set_seed(seed)
    device = get_device(prefer_gpu=bool(cfg_get(config, "training.use_gpu", True)))

    ctx = build_data_context(config, data_root=data_root)
    shared_test = test_loader(ctx, config)

    epochs = int(cfg_get(config, "training.local_epochs", cfg_get(config, "training.epochs", 1)))
    max_batches = cfg_get(config, "training.max_batches")
    severity_weight = float(cfg_get(config, "training.severity_weight", 1.0))

    per_node: dict[str, dict] = {}
    for node_id in ctx.manifest.nodes:
        # Fresh model per node — no weight sharing (that's the point of the floor).
        model = build_model(config).to(device)
        optimizer = build_optimizer(config, model)
        loss_fn = DualHeadLoss(severity_weight=severity_weight).to(device)
        train_loader = node_loader(ctx, node_id, config, train=True)

        history = []
        for epoch in range(epochs):
            stats = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device, max_batches=max_batches
            )
            stats["epoch"] = epoch
            history.append(stats)

        metrics = evaluate(model, shared_test, device, max_batches=max_batches)
        per_node[node_id] = {
            "train_size": len(ctx.manifest.nodes[node_id]),
            "train_history": history,
            "test_metrics": metrics,
        }
        auc = metrics.get("pathology", {}).get("mean_auc", float("nan"))
        print(f"[single-site] node {node_id}: mean_auc={auc:.4f}")

    results = {
        "run_type": "single_site",
        "config_source": config.get("_source"),
        "seed": seed,
        "device": str(device),
        "backbone": cfg_get(config, "model.backbone", "densenet121"),
        "partition": {
            "scheme": ctx.manifest.scheme,
            "num_nodes": ctx.manifest.num_nodes,
            "alpha": ctx.manifest.alpha,
            "test_size": len(ctx.manifest.test),
        },
        "per_node": per_node,
        "mean_auc_across_nodes": _mean_over_nodes(per_node, "mean_auc"),
    }

    out_dir = Path(cfg_get(config, "experiment.output_dir", "results/single_site"))
    save_json(results, out_dir / "single_site_results.json")
    print(f"[single-site] wrote results -> {out_dir / 'single_site_results.json'}")
    return results


def _mean_over_nodes(per_node: dict[str, dict], key: str) -> float:
    vals = [
        n["test_metrics"].get("pathology", {}).get(key)
        for n in per_node.values()
    ]
    vals = [v for v in vals if v is not None and v == v]  # drop None / NaN
    return float(sum(vals) / len(vals)) if vals else float("nan")
