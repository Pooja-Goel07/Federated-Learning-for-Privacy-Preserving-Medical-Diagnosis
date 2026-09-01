"""Tests for Byzantine-robust aggregation + attack clients (Task 9).

Two levels:
  1. Unit: feed 4 honest updates + 1 gross outlier to each aggregation rule and
     check the robust rules reject the outlier while plain FedAvg is dragged toward it.
  2. End-to-end: run a federated job with a malicious client and confirm the attack
     is applied/recorded and a robust strategy still completes.

Run directly:  python tests/test_byzantine.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from src.data.synthetic import make_synthetic_dataset
from src.experiments.federated import run_federated
from src.federated.strategies import available_strategies, build_strategy


def _results(updates, ns):
    out = []
    for u, n in zip(updates, ns):
        fr = FitRes(status=Status(Code.OK, "ok"),
                    parameters=ndarrays_to_parameters(u), num_examples=n, metrics={})
        out.append((None, fr))
    return out


def test_registry() -> None:
    expected = {"fedavg", "krum", "multikrum", "trimmed_mean", "median", "trust"}
    assert expected.issubset(set(available_strategies())), available_strategies()
    print(f"  strategies registered: {available_strategies()} OK")


def test_robust_rejects_outlier() -> None:
    rng = np.random.default_rng(0)
    honest = [[np.array([1.0, 1.0, 1.0]) + 0.01 * rng.standard_normal(3)] for _ in range(4)]
    malicious = [np.array([100.0, -100.0, 100.0])]  # gross outlier
    updates = honest + [malicious]
    ns = [10, 10, 10, 10, 10]

    def agg(name, **kw):
        strat = build_strategy(name, min_fit_clients=1, min_available_clients=1, **kw)
        params, _ = strat.aggregate_fit(1, _results(updates, ns), [])
        return parameters_to_ndarrays(params)[0]

    honest_mean = np.mean([h[0] for h in honest], axis=0)
    fedavg = agg("fedavg")
    for name, kw in [("krum", {"num_malicious": 1}), ("median", {}),
                     ("trimmed_mean", {"beta": 0.2}), ("trust", {})]:
        robust = agg(name, **kw)
        d_robust = float(np.linalg.norm(robust - honest_mean))
        d_fedavg = float(np.linalg.norm(fedavg - honest_mean))
        assert d_robust < d_fedavg, f"{name}: not more robust than fedavg ({d_robust} !< {d_fedavg})"
        print(f"  {name:12s}: dist-to-honest={d_robust:.3f}  (fedavg={d_fedavg:.3f}) OK")


def _config(root: Path, out: Path, strategy: str, attack: str) -> dict:
    return {
        "_source": "smoke-byz",
        "experiment": {"seed": 0, "output_dir": str(out)},
        "data": {"root": str(root), "image_size": 64, "severity": {"enabled": True, "binning": "tercile"}},
        "partition": {"num_nodes": 3, "scheme": "iid", "test_frac": 0.3},
        "model": {"backbone": "densenet121", "pretrained": False,
                  "heads": {"pathology": True, "severity": True}},
        "training": {"local_epochs": 1, "batch_size": 8, "optimizer": "adam", "lr": 1e-4,
                     "rounds": 2, "num_workers": 0, "use_gpu": False,
                     "severity_weight": 1.0, "max_batches": 2},
        "federated": {"strategy": strategy, "fraction_fit": 1.0, "robust": {"beta": 0.34, "multi_k": 2}},
        "attack": {"enabled": True, "byzantine": {"type": attack, "fraction": 0.34, "scale": 10.0}},
    }


def test_attacked_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = make_synthetic_dataset(tmp / "data", num_patients=45, seed=9, write_images=True)
        root = paths["image_dir"].parent
        for strategy, attack in [("median", "sign_flip"), ("krum", "model_replacement")]:
            res = run_federated(_config(root, tmp / f"{strategy}_{attack}", strategy, attack))
            assert res["attack"]["enabled"] is True
            assert res["attack"]["num_malicious"] == 1
            assert res["strategy"] == strategy
            # a malicious client reported its attack in the round metrics
            attacks_seen = {c.get("attack") for r in res["history"][1:] for c in r["clients"]}
            assert attack in attacks_seen, attacks_seen
            print(f"  end-to-end {strategy} vs {attack}: 1/3 malicious, completed OK")


def main() -> None:
    test_registry()
    test_robust_rejects_outlier()
    test_attacked_run()
    print("ALL BYZANTINE (TASK 9) TESTS PASSED")


if __name__ == "__main__":
    main()
