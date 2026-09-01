"""Smoke test for the core federated loop (Task 6).

Runs 2 FedAvg rounds over 3 nodes on a tiny synthetic dataset (CPU, no downloads).
Run directly:  python tests/test_federated.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import make_synthetic_dataset
from src.experiments.federated import run_federated
from src.federated import available_strategies


def _config(root: Path, out_dir: Path, rounds: int = 2) -> dict:
    return {
        "_source": "smoke-fed",
        "experiment": {"seed": 0, "output_dir": str(out_dir)},
        "data": {"root": str(root), "image_size": 64, "severity": {"enabled": True, "binning": "tercile"}},
        "partition": {"num_nodes": 3, "scheme": "non_iid", "test_frac": 0.3, "non_iid": {"alpha": 0.5}},
        "model": {"backbone": "densenet121", "pretrained": False, "dropout": 0.0,
                  "heads": {"pathology": True, "severity": True}},
        "training": {"local_epochs": 1, "batch_size": 8, "optimizer": "adam", "lr": 1e-4,
                     "rounds": rounds, "num_workers": 0, "use_gpu": False,
                     "severity_weight": 1.0, "max_batches": 2},
        "federated": {"strategy": "fedavg", "fraction_fit": 1.0},
    }


def test_strategy_registry() -> None:
    assert "fedavg" in available_strategies()
    print(f"  strategy registry: {available_strategies()} OK")


def test_federated_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = make_synthetic_dataset(tmp / "data", num_patients=45, seed=5, write_images=True)
        rounds = 2
        results = run_federated(_config(paths["image_dir"].parent, tmp / "results", rounds))

        assert results["run_type"] == "federated"
        assert results["strategy"] == "fedavg"
        # round 0 baseline + one entry per round
        assert len(results["history"]) == rounds + 1
        for r in results["history"][1:]:
            assert len(r["clients"]) == 3, "all 3 nodes should contribute each round"
            assert "pathology" in r["test_metrics"]
        assert "mean_auc" in results["final_test_metrics"]["pathology"]
        assert (tmp / "results" / "federated_results.json").exists()
        final_auc = results["final_test_metrics"]["pathology"]["mean_auc"]
        print(f"  federated 3-node x {rounds} rounds OK | final global mean_auc={final_auc:.3f}")


def main() -> None:
    test_strategy_registry()
    test_federated_run()
    print("ALL FEDERATED SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
