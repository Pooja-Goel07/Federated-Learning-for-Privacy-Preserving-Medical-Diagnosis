"""Smoke tests for heterogeneity mitigation (Task 7): FedProx + FedBN.

Run directly:  python tests/test_heterogeneity.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.synthetic import make_synthetic_dataset
from src.experiments.federated import run_federated
from src.federated.params import bn_state_keys, get_parameters, set_parameters
from src.models import DualHeadNet, ModelConfig


def test_bn_exclusion_plumbing() -> None:
    model = DualHeadNet(ModelConfig(backbone="densenet121", pretrained=False))
    bn_keys = bn_state_keys(model)
    assert len(bn_keys) > 0, "densenet121 should have BatchNorm layers"
    full = get_parameters(model)
    non_bn = get_parameters(model, bn_keys)
    assert len(non_bn) < len(full), "excluding BN must drop arrays"

    # Perturb local BN, then set only non-BN params -> BN must stay perturbed (local).
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.add_(5.0)
                marker = float(m.weight.detach().flatten()[0])
                break
    set_parameters(model, non_bn, bn_keys)  # should NOT touch BN
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            assert abs(float(m.weight.detach().flatten()[0]) - marker) < 1e-6, "BN was overwritten"
            break
    print(f"  BN exclusion: {len(bn_keys)} BN keys kept local, {len(non_bn)}/{len(full)} sent OK")


def _config(root: Path, out_dir: Path, mu: float = 0.0, fedbn: bool = False) -> dict:
    return {
        "_source": "smoke-het",
        "experiment": {"seed": 0, "output_dir": str(out_dir)},
        "data": {"root": str(root), "image_size": 64, "severity": {"enabled": True, "binning": "tercile"}},
        "partition": {"num_nodes": 3, "scheme": "non_iid", "test_frac": 0.3, "non_iid": {"alpha": 0.3}},
        "model": {"backbone": "densenet121", "pretrained": False, "dropout": 0.0,
                  "heads": {"pathology": True, "severity": True}},
        "training": {"local_epochs": 1, "batch_size": 8, "optimizer": "adam", "lr": 1e-4,
                     "rounds": 2, "num_workers": 0, "use_gpu": False,
                     "severity_weight": 1.0, "max_batches": 2},
        "federated": {"strategy": "fedavg", "fraction_fit": 1.0,
                      "heterogeneity_mitigation": {"fedprox_mu": mu, "fedbn": fedbn}},
    }


def _run(root: Path, out: Path, **kw) -> dict:
    results = run_federated(_config(root, out, **kw))
    assert len(results["history"]) == 3  # round 0 + 2
    for r in results["history"][1:]:
        assert len(r["clients"]) == 3
        assert "pathology" in r["test_metrics"]
    return results


def test_fedprox_and_fedbn() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = make_synthetic_dataset(tmp / "data", num_patients=45, seed=6, write_images=True)
        root = paths["image_dir"].parent

        r_prox = _run(root, tmp / "prox", mu=0.1)
        assert r_prox["heterogeneity_mitigation"]["fedprox_mu"] == 0.1
        print(f"  FedProx (mu=0.1): final mean_auc="
              f"{r_prox['final_test_metrics']['pathology']['mean_auc']:.3f} OK")

        r_bn = _run(root, tmp / "bn", fedbn=True)
        assert r_bn["heterogeneity_mitigation"]["fedbn"] is True
        print(f"  FedBN: final (per-client avg) mean_auc="
              f"{r_bn['final_test_metrics']['pathology']['mean_auc']:.3f} OK")


def main() -> None:
    test_bn_exclusion_plumbing()
    test_fedprox_and_fedbn()
    print("ALL HETEROGENEITY (TASK 7) SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
