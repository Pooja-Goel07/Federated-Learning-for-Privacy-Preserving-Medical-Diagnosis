"""Smoke tests for the model + centralised training loop (Tasks 3 & 4).

Runs on CPU with a tiny synthetic dataset and pretrained=False, so it needs no
downloads. Run directly:  python tests/test_model_training.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.constants import NUM_PATHOLOGIES, SEVERITY_CLASSES, SEVERITY_IGNORE_INDEX
from src.data.synthetic import make_synthetic_dataset
from src.experiments.centralized import run_centralized
from src.models import DualHeadLoss, DualHeadNet, ModelConfig


def test_model_forward_and_loss() -> None:
    model = DualHeadNet(ModelConfig(backbone="densenet121", pretrained=False))
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out["pathology"].shape == (2, NUM_PATHOLOGIES)
    assert out["severity"].shape == (2, len(SEVERITY_CLASSES))

    loss_fn = DualHeadLoss(severity_weight=1.0)
    targets = {
        "pathology": torch.randint(0, 2, (2, NUM_PATHOLOGIES)).float(),
        # one labelled severity sample, one ignored -> masking must handle it
        "severity": torch.tensor([1, SEVERITY_IGNORE_INDEX]),
    }
    loss = loss_fn(out, targets)
    assert loss.total.requires_grad and torch.isfinite(loss.total)
    assert loss.num_severity == 1
    loss.total.backward()  # gradients flow
    print("  model forward + masked dual-head loss + backward OK")


def _smoke_config(synthetic_root: Path, out_dir: Path) -> dict:
    return {
        "_source": "smoke",
        "experiment": {"seed": 0, "output_dir": str(out_dir)},
        "data": {
            "root": str(synthetic_root),
            "image_size": 64,
            "severity": {"enabled": True, "binning": "tercile"},
        },
        "partition": {
            "num_nodes": 3,
            "scheme": "non_iid",
            "non_iid": {"alpha": 0.5},
            "test_frac": 0.3,
        },
        "model": {
            "backbone": "densenet121",
            "pretrained": False,
            "dropout": 0.0,
            "heads": {"pathology": True, "severity": True},
        },
        "training": {
            "local_epochs": 1,
            "batch_size": 8,
            "optimizer": "adam",
            "lr": 1e-4,
            "num_workers": 0,
            "use_gpu": False,
            "max_batches": 2,
            "severity_weight": 1.0,
        },
    }


def test_centralized_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = make_synthetic_dataset(tmp / "data", num_patients=40, seed=2, write_images=True)
        config = _smoke_config(paths["image_dir"].parent, tmp / "results")
        results = run_centralized(config)

        assert results["run_type"] == "centralized"
        assert len(results["train_history"]) == 1
        tm = results["test_metrics"]
        assert "pathology" in tm and "mean_auc" in tm["pathology"]
        assert "severity" in tm and "severity_mae" in tm["severity"]
        assert (tmp / "results" / "centralized_results.json").exists()
        print(
            "  centralized run OK | "
            f"mean_auc={tm['pathology']['mean_auc']:.3f} "
            f"sev_n={tm['severity']['n']}"
        )


def main() -> None:
    test_model_forward_and_loss()
    test_centralized_run()
    print("ALL MODEL/TRAINING SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
