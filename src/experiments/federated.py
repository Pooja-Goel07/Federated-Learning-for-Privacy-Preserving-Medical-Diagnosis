"""Task 6 — core federated learning loop (Flower FedAvg).

Builds one client per hospital node from the partition manifest, runs federated
rounds via the in-process orchestrator (which delegates aggregation to a real
Flower Strategy), and evaluates the global model on the shared test set each round.
This is the core demo deliverable: 3 nodes train locally -> centre aggregates ->
global model is evaluated.
"""

from __future__ import annotations

from pathlib import Path

from src.attacks.byzantine import ByzantineClient
from src.federated.client import FLClient
from src.federated.simulation import run_federated_rounds
from src.federated.strategies import build_strategy
from src.models import DualHeadLoss
from src.models.engine import evaluate
from src.privacy.dp import DPConfig, ensure_dp_compatible
from src.privacy.client import DPFLClient

from .builders import build_model, build_optimizer
from .config import cfg_get
from .data_setup import build_data_context, compute_pos_weight, node_loader, test_loader
from .utils import get_device, save_json, set_seed


def run_federated(config: dict, data_root: str | Path | None = None) -> dict:
    seed = int(cfg_get(config, "experiment.seed", 42))
    set_seed(seed)
    device = get_device(prefer_gpu=bool(cfg_get(config, "training.use_gpu", True)))

    ctx = build_data_context(config, data_root=data_root)

    local_epochs = int(cfg_get(config, "training.local_epochs", 1))
    max_batches = cfg_get(config, "training.max_batches")
    severity_weight = float(cfg_get(config, "training.severity_weight", 1.0))
    num_rounds = int(cfg_get(config, "training.rounds", 10))

    # Heterogeneity mitigation (Task 7): FedProx proximal term + FedBN local BN.
    fedprox_mu = float(cfg_get(config, "federated.heterogeneity_mitigation.fedprox_mu", 0.0))
    fedbn = bool(cfg_get(config, "federated.heterogeneity_mitigation.fedbn", False))

    # Differential privacy (Task 8). DP is mutually exclusive with FedProx/FedBN.
    dp = DPConfig.from_config(config)
    if dp.enabled and (fedbn or fedprox_mu > 0):
        print("[federated] DP enabled -> disabling FedBN/FedProx (incompatible with DP-SGD)")
        fedbn, fedprox_mu = False, 0.0

    # Byzantine attacks (Task 9). A fraction of nodes become malicious clients.
    num_nodes = len(ctx.manifest.nodes)
    attack_enabled = bool(cfg_get(config, "attack.enabled", False))
    attack_type = cfg_get(config, "attack.byzantine.type", "none")
    attack_fraction = float(cfg_get(config, "attack.byzantine.fraction", 0.0))
    attack_scale = float(cfg_get(config, "attack.byzantine.scale", 10.0))
    num_malicious = (
        int(round(attack_fraction * num_nodes))
        if attack_enabled and attack_type not in ("none", None)
        else 0
    )
    node_ids = list(ctx.manifest.nodes)
    malicious_ids = set(node_ids[:num_malicious])  # deterministic for reproducibility

    def optimizer_fn(model):
        return build_optimizer(config, model)

    def make_model():
        model = build_model(config)
        return ensure_dp_compatible(model) if dp.enabled else model

    # Class balancing (shared across clients + eval): pos_weight from all training data.
    pos_weight = None
    if bool(cfg_get(config, "training.class_balanced", False)):
        all_train_files = [f for files in ctx.manifest.nodes.values() for f in files]
        pos_weight = compute_pos_weight(ctx.data_entry, all_train_files)

    clients = []
    for node_id in node_ids:
        train_loader = node_loader(ctx, node_id, config, train=True)
        if node_id in malicious_ids:
            client = ByzantineClient(
                node_id=node_id, model=make_model(), train_loader=train_loader,
                optimizer_fn=optimizer_fn, attack_type=attack_type, num_clients=num_nodes,
                scale=attack_scale, device=device, local_epochs=local_epochs,
                severity_weight=severity_weight, max_batches=max_batches, pos_weight=pos_weight,
            )
        elif dp.enabled:
            client = DPFLClient(
                node_id=node_id, model=make_model(), train_loader=train_loader,
                optimizer_fn=optimizer_fn, dp=dp, device=device,
                local_epochs=local_epochs, severity_weight=severity_weight,
                max_batches=max_batches, pos_weight=pos_weight,
            )
        else:
            client = FLClient(
                node_id=node_id, model=make_model(), train_loader=train_loader,
                optimizer_fn=optimizer_fn, device=device, local_epochs=local_epochs,
                severity_weight=severity_weight, max_batches=max_batches,
                fedprox_mu=fedprox_mu, fedbn=fedbn, pos_weight=pos_weight,
            )
        clients.append(client)

    strategy = build_strategy(
        cfg_get(config, "federated.strategy", "fedavg"),
        fraction_fit=float(cfg_get(config, "federated.fraction_fit", 1.0)),
        min_fit_clients=len(clients),
        min_available_clients=len(clients),
        num_malicious=max(num_malicious, 1),  # Krum needs an assumed f>=1
        multi_k=int(cfg_get(config, "federated.robust.multi_k", 2)),
        beta=float(cfg_get(config, "federated.robust.beta", 0.2)),
    )

    global_model = make_model().to(device)
    shared_test = test_loader(ctx, config)
    eval_loss_fn = DualHeadLoss(severity_weight=severity_weight, pos_weight=pos_weight).to(device)

    def evaluate_model_fn(model):
        return evaluate(model, shared_test, device, max_batches=max_batches, loss_fn=eval_loss_fn)

    history, final_params = run_federated_rounds(
        global_model, clients, strategy, num_rounds, evaluate_model_fn, fedbn=fedbn
    )

    results = {
        "run_type": "federated",
        "config_source": config.get("_source"),
        "seed": seed,
        "device": str(device),
        "strategy": cfg_get(config, "federated.strategy", "fedavg"),
        "backbone": cfg_get(config, "model.backbone", "densenet121"),
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "heterogeneity_mitigation": {"fedprox_mu": fedprox_mu, "fedbn": fedbn},
        "attack": {
            "enabled": attack_enabled and num_malicious > 0,
            "type": attack_type if num_malicious > 0 else "none",
            "fraction": attack_fraction,
            "num_malicious": num_malicious,
            "malicious_nodes": sorted(malicious_ids),
        },
        "privacy": {
            "dp_enabled": dp.enabled,
            "max_grad_norm": dp.max_grad_norm,
            "noise_multiplier": dp.noise_multiplier,
            "target_delta": dp.target_delta,
            "final_epsilon": _final_epsilon(history) if dp.enabled else None,
        },
        "partition": {
            "scheme": ctx.manifest.scheme,
            "num_nodes": ctx.manifest.num_nodes,
            "alpha": ctx.manifest.alpha,
            "node_sizes": {k: len(v) for k, v in ctx.manifest.nodes.items()},
            "test_size": len(ctx.manifest.test),
        },
        "history": history,
        "final_test_metrics": history[-1]["test_metrics"],
    }

    out_dir = Path(cfg_get(config, "experiment.output_dir", "results/federated"))
    save_json(results, out_dir / "federated_results.json")
    print(f"[federated] wrote results -> {out_dir / 'federated_results.json'}")
    return results


def _final_epsilon(history: list[dict]) -> float:
    """Worst-case (max) cumulative epsilon across clients in the final round."""
    if len(history) < 2:
        return float("inf")
    eps = [c.get("epsilon") for c in history[-1]["clients"] if c.get("epsilon") is not None]
    return max(eps) if eps else float("inf")
