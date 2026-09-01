"""Smoke tests for the differential-privacy layer (Task 8, Opacus DP-SGD).

Kept deliberately tiny (small model input, few batches) because per-sample
gradients are compute-heavy on CPU. Run directly:  python tests/test_dp.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from src.data.synthetic import make_synthetic_dataset
from src.experiments.federated import run_federated
from src.models import DualHeadNet, ModelConfig
from src.privacy.dp import DPConfig, ensure_dp_compatible


def test_ensure_dp_compatible() -> None:
    model = DualHeadNet(ModelConfig(backbone="densenet121", pretrained=False))
    assert any(isinstance(m, _BatchNorm) for m in model.modules()), "densenet should start with BN"
    fixed = ensure_dp_compatible(model)
    assert not any(isinstance(m, _BatchNorm) for m in fixed.modules()), "BN must be gone after fix"
    print("  ensure_dp_compatible: BatchNorm -> GroupNorm OK")


def _config(root: Path, out_dir: Path, sigma: float = 1.0) -> dict:
    return {
        "_source": "smoke-dp",
        "experiment": {"seed": 0, "output_dir": str(out_dir)},
        "data": {"root": str(root), "image_size": 32, "severity": {"enabled": True, "binning": "tercile"}},
        "partition": {"num_nodes": 3, "scheme": "iid", "test_frac": 0.3},
        "model": {"backbone": "densenet121", "pretrained": False,
                  "heads": {"pathology": True, "severity": True}},
        "training": {"local_epochs": 1, "batch_size": 4, "optimizer": "sgd", "lr": 0.05,
                     "rounds": 2, "num_workers": 0, "use_gpu": False,
                     "severity_weight": 1.0, "max_batches": 1},
        "federated": {"strategy": "fedavg", "fraction_fit": 1.0},
        "privacy": {"dp": {"enabled": True, "max_grad_norm": 1.0,
                           "noise_multiplier": sigma, "target_delta": 1e-5, "accountant": "rdp"}},
    }


def test_federated_dp_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = make_synthetic_dataset(tmp / "data", num_patients=30, seed=7, write_images=True, image_size=32)
        results = run_federated(_config(paths["image_dir"].parent, tmp / "results"))

        assert results["privacy"]["dp_enabled"] is True
        eps = results["privacy"]["final_epsilon"]
        assert eps is not None and math.isfinite(eps) and eps > 0, f"bad epsilon: {eps}"
        # every client reported an epsilon each round
        for r in results["history"][1:]:
            for c in r["clients"]:
                assert "epsilon" in c and c["epsilon"] > 0
        print(f"  federated DP (sigma=1.0) OK | final (worst-case) epsilon={eps:.3f}")


def test_epsilon_grows_with_less_noise() -> None:
    """Lower noise multiplier -> larger epsilon (weaker privacy)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = make_synthetic_dataset(tmp / "data", num_patients=30, seed=8, write_images=True, image_size=32)
        root = paths["image_dir"].parent
        eps_high_noise = run_federated(_config(root, tmp / "hi", sigma=2.0))["privacy"]["final_epsilon"]
        eps_low_noise = run_federated(_config(root, tmp / "lo", sigma=0.7))["privacy"]["final_epsilon"]
        assert eps_low_noise > eps_high_noise, f"{eps_low_noise=} !> {eps_high_noise=}"
        print(f"  epsilon monotonicity: sigma2.0->{eps_high_noise:.2f} < sigma0.7->{eps_low_noise:.2f} OK")


def main() -> None:
    test_ensure_dp_compatible()
    test_federated_dp_run()
    test_epsilon_grows_with_less_noise()
    print("ALL DP (TASK 8) SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
