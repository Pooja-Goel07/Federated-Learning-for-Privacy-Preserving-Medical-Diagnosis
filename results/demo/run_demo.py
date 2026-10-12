"""End-to-end demo of the federated architecture on a small synthetic dataset.

Runs the WHOLE pipeline with the defence layers OFF (plain FedAvg aggregation, no
differential privacy, no Byzantine-robust aggregation, no attacks). Its purpose is
to show the architecture actually works: three simulated hospitals train locally,
a centre node aggregates their weights with FedAvg, and the global model is
evaluated each round — recovering most of the centralised (pooled-data) ceiling and
beating the isolated single-hospital floor.

The synthetic images carry a *learnable signal*: each of six active pathologies is a
bright patch at a fixed location, so a model can genuinely learn image -> label. This
is test/demo scaffolding, NOT real data — it exists only to exercise the real code
paths (src/data, src/models, src/federated) and produce presentable curves.

Run from the repo root:
    python results/demo/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Make `import src...` work when run as a plain script from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.constants import BBOX_PATHOLOGIES, NO_FINDING, PATHOLOGIES  # noqa: E402
from src.experiments.suite import run_core_suite  # noqa: E402

# --- demo knobs (kept small so the whole thing runs on a CPU in minutes) ---
DEMO_DIR = REPO_ROOT / "results" / "demo"
DATA_DIR = DEMO_DIR / "data"
IMAGE_SIZE = 48            # generated == model input size (no resize blur)
NUM_PATIENTS = 260
SEED = 42
BASELINE_EPOCHS = 8       # centralised + single-site local epochs
FED_ROUNDS = 12           # federated communication rounds
FED_LOCAL_EPOCHS = 1      # local epochs per round

# The 6 pathologies we inject a real signal for; all are also bbox pathologies, so
# the same names can label the severity boxes. The other 8 stay all-negative (their
# AUC is undefined and excluded from the mean, per src/evaluation/metrics.py).
ACTIVE = [p for p in PATHOLOGIES[:6]]
assert all(p in BBOX_PATHOLOGIES for p in ACTIVE)


def _patch_boxes(size: int) -> list[tuple[int, int, int, int]]:
    """A 2-row x 3-col grid of patch rectangles (y0, x0, y1, x1) for the 6 signals."""
    boxes = []
    cell_h, cell_w = size // 4, size // 4
    for r in range(2):
        for c in range(3):
            y0 = 3 + r * (cell_h + 3)
            x0 = 3 + c * (cell_w + 2)
            boxes.append((y0, x0, y0 + cell_h, x0 + cell_w))
    return boxes


def make_signal_dataset(out_dir: Path, seed: int = SEED) -> None:
    """Write Data_Entry / BBox CSVs + signal-bearing PNGs under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    patch = _patch_boxes(IMAGE_SIZE)
    sev_sides = {0: 6, 1: 11, 2: 16}        # image cue for mild/moderate/severe
    sev_box = {0: 80, 1: 200, 2: 400}       # bbox side (orig 1024-space) -> terciles

    rows, bbox_rows = [], []
    for pid in range(1, NUM_PATIENTS + 1):
        for follow in range(int(rng.integers(1, 3))):
            fname = f"{pid:08d}_{follow:03d}.png"
            # independent presence for each active pathology
            present = [bool(rng.random() < 0.35) for _ in ACTIVE]
            is_bbox = rng.random() < 0.35
            # build the grayscale image: dark background + noise
            img = rng.normal(28, 7, size=(IMAGE_SIZE, IMAGE_SIZE))
            for k, on in enumerate(present):
                if on:
                    y0, x0, y1, x1 = patch[k]
                    img[y0:y1, x0:x1] += 205 + rng.normal(0, 6, (y1 - y0, x1 - x0))
            sev_regime = None
            if is_bbox:
                # force one active pathology on so the box has a valid label
                if not any(present):
                    present[int(rng.integers(0, len(ACTIVE)))] = True
                sev_regime = int(rng.integers(0, 3))
                s = sev_sides[sev_regime]
                cy = IMAGE_SIZE - s - 3
                cx = (IMAGE_SIZE - s) // 2
                img[cy:cy + s, cx:cx + s] += 215 + rng.normal(0, 5, (s, s))
            img = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(img, mode="L").save(img_dir / fname)

            findings = [ACTIVE[k] for k, on in enumerate(present) if on]
            label_str = "|".join(findings) if findings else NO_FINDING
            rows.append({
                "Image Index": fname, "Finding Labels": label_str, "Patient ID": pid,
                "OriginalImage Width": 1024, "Height": 1024,
            })
            if is_bbox:
                box_label = next(f for f in findings if f in BBOX_PATHOLOGIES)
                side = float(sev_box[sev_regime])
                bbox_rows.append({
                    "Image Index": fname, "Finding Label": box_label,
                    "Bbox [x": 100.0, "y": 100.0, "w": side, "h]": side,
                })

    pd.DataFrame(rows).to_csv(out_dir / "Data_Entry_2017.csv", index=False)
    bdf = pd.DataFrame(bbox_rows)
    bdf["Unnamed: 6"] = ""
    bdf.to_csv(out_dir / "BBox_List_2017.csv", index=False)
    print(f"[demo] wrote {len(rows)} images ({len(bbox_rows)} with severity boxes) -> {img_dir}")


def base_config() -> dict:
    """Config with every defence layer OFF (plain FedAvg, no DP, no robustness)."""
    return {
        "_source": "results/demo (in-memory)",
        "experiment": {"seed": SEED},
        "data": {
            "root": str(DATA_DIR), "image_size": IMAGE_SIZE,
            "severity": {"enabled": True, "binning": "tercile"},
        },
        "partition": {
            "num_nodes": 3, "scheme": "non_iid", "test_frac": 0.25,
            # shared manifest -> all three runs use the SAME split & test set
            "manifest_path": str(DATA_DIR / "manifest.json"),
            "non_iid": {"alpha": 0.5},
        },
        "model": {
            "backbone": "densenet121", "pretrained": False, "dropout": 0.0,
            "heads": {"pathology": True, "severity": True},
        },
        "training": {
            "local_epochs": BASELINE_EPOCHS, "batch_size": 16, "optimizer": "adam",
            "lr": 1e-3, "rounds": FED_ROUNDS, "num_workers": 0, "use_gpu": False,
            "severity_weight": 1.0, "max_batches": None,
        },
        "federated": {
            "strategy": "fedavg", "fraction_fit": 1.0,
            "heterogeneity_mitigation": {"fedprox_mu": 0.0, "fedbn": False},
            "robust": {"beta": 0.2, "multi_k": 2},
        },
        "privacy": {"dp": {"enabled": False}},
        "attack": {"enabled": False, "byzantine": {"type": "none", "fraction": 0.0}},
        "evaluation": {"metrics": ["auc", "accuracy", "severity_mae", "severity_qwk"]},
    }


def main() -> None:
    print("=" * 70)
    print("DEMO: full federated architecture on synthetic data (defences OFF)")
    print("=" * 70)
    make_signal_dataset(DATA_DIR)
    run_core_suite(
        base_config(),
        out_dir=DEMO_DIR,
        data_root=str(DATA_DIR),
        fed_local_epochs=FED_LOCAL_EPOCHS,
        extra_setup={
            "dataset": "synthetic (signal-bearing)",
            "image_size": IMAGE_SIZE,
            "backbone": "densenet121 (from scratch)",
            "baseline_epochs": BASELINE_EPOCHS,
            "active_pathologies": ACTIVE,
        },
    )


if __name__ == "__main__":
    main()
