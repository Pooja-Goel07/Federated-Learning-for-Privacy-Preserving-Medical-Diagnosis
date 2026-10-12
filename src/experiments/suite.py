"""Run the core comparison suite (centralised + single-site + federated) in one call.

This is the shared driver behind both the local synthetic demo (results/demo) and the
real-data Colab run (notebooks/colab_training.ipynb). It runs the three configurations
from a single base config, so floor / federated / ceiling are produced from the *same*
partition and the *same* held-out test set, then writes a compact ``summary.json`` that
``results/demo/make_report.py`` turns into charts + a report.

Defence layers (DP, Byzantine-robust aggregation, attacks) are controlled by the config
and are OFF by default here — this suite is the plain-FedAvg core.
"""

from __future__ import annotations

import copy
from pathlib import Path

from .centralized import run_centralized
from .config import cfg_get
from .federated import run_federated
from .single_site import run_single_site
from .utils import save_json


def _path_auc(metrics: dict) -> float:
    return metrics.get("pathology", {}).get("mean_auc", float("nan"))


def _path_acc(metrics: dict) -> float:
    return metrics.get("pathology", {}).get("mean_accuracy", float("nan"))


def build_summary(central: dict, single: dict, fed: dict, extra_setup: dict | None = None) -> dict:
    """Compile a compact, plot-ready summary from the three run-result dicts."""
    cm = central["test_metrics"]
    fm = fed["final_test_metrics"]
    setup = {
        "backbone": fed.get("backbone"),
        "num_nodes": fed.get("partition", {}).get("num_nodes"),
        "partition": f"{fed.get('partition', {}).get('scheme')} (alpha={fed.get('partition', {}).get('alpha')})",
        "fed_rounds": fed.get("num_rounds"),
        "defences": "none (plain FedAvg, no DP, no robust aggregation)",
    }
    if extra_setup:
        setup.update(extra_setup)
    return {
        "setup": setup,
        "centralized": {
            "mean_auc": _path_auc(cm),
            "mean_accuracy": _path_acc(cm),
            "per_pathology_auc": cm.get("pathology", {}).get("per_pathology_auc", {}),
            "severity": cm.get("severity", {}),
        },
        "single_site": {
            "per_node_auc": {nid: _path_auc(v["test_metrics"]) for nid, v in single["per_node"].items()},
            "mean_auc": single["mean_auc_across_nodes"],
        },
        "federated": {
            "final_mean_auc": _path_auc(fm),
            "final_mean_accuracy": _path_acc(fm),
            "per_pathology_auc": fm.get("pathology", {}).get("per_pathology_auc", {}),
            "severity": fm.get("severity", {}),
            "history": [
                {
                    "round": h["round"],
                    "mean_auc": _path_auc(h["test_metrics"]),
                    "loss": h["test_metrics"].get("loss", float("nan")),
                }
                for h in fed["history"]
            ],
        },
    }


def run_core_suite(
    config: dict,
    out_dir: str | Path,
    data_root: str | Path | None = None,
    fed_local_epochs: int = 1,
    extra_setup: dict | None = None,
) -> dict:
    """Run centralised, single-site and federated from one base config.

    Args:
        config: base experiment config (defences off = core suite).
        out_dir: directory all three results + summary.json are written to. The three
            runs use distinct filenames (centralized_/single_site_/federated_results.json)
            so one directory holds everything.
        data_root: optional override for data.root (e.g. a synthetic or Colab dir).
        fed_local_epochs: local epochs per federated round (baselines use the config's
            training.local_epochs; federation typically wants 1).
        extra_setup: extra key/values merged into summary['setup'] (dataset name, etc.).

    Returns:
        The summary dict (also written to ``out_dir/summary.json``).
    """
    out_dir = Path(out_dir)

    cfg_c = copy.deepcopy(config)
    cfg_c["experiment"]["output_dir"] = str(out_dir)
    print("\n--- [1/3] Centralised baseline (upper bound) ---")
    central = run_centralized(cfg_c, data_root=data_root)

    cfg_s = copy.deepcopy(config)
    cfg_s["experiment"]["output_dir"] = str(out_dir)
    print("\n--- [2/3] Single-site baselines (lower bound) ---")
    single = run_single_site(cfg_s, data_root=data_root)

    cfg_f = copy.deepcopy(config)
    cfg_f["experiment"]["output_dir"] = str(out_dir)
    cfg_f.setdefault("training", {})["local_epochs"] = fed_local_epochs
    print("\n--- [3/3] Federated FedAvg (nodes -> centre) ---")
    fed = run_federated(cfg_f, data_root=data_root)

    summary = build_summary(central, single, fed, extra_setup=extra_setup)
    save_json(summary, out_dir / "summary.json")
    print(f"\n[suite] wrote summary -> {out_dir / 'summary.json'}")
    print("=" * 66)
    print(f"  Centralised (ceiling)  mean AUC = {summary['centralized']['mean_auc']:.3f}")
    print(f"  Single-site (floor)    mean AUC = {summary['single_site']['mean_auc']:.3f}")
    print(f"  Federated  (FedAvg)    mean AUC = {summary['federated']['final_mean_auc']:.3f}")
    print("=" * 66)
    return summary
