"""Generate a tiny synthetic ChestX-ray14-shaped dataset for tests / smoke runs.

Creates Data_Entry_2017.csv, BBox_List_2017.csv and small dummy PNGs with the same
schema as the NIH release, so the full pipeline (labels -> severity -> partition ->
Dataset -> model -> training) can be exercised end-to-end without the 40 GB download.
This is test scaffolding, NOT real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import BBOX_PATHOLOGIES, NO_FINDING, PATHOLOGIES


def make_synthetic_dataset(
    out_dir: str | Path,
    num_patients: int = 60,
    max_images_per_patient: int = 3,
    bbox_fraction: float = 0.15,
    image_size: int = 32,
    seed: int = 0,
    write_images: bool = True,
) -> dict[str, Path]:
    """Write a synthetic dataset under ``out_dir``.

    Returns a dict of key paths: data_entry, bbox_csv, image_dir.
    """
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows = []
    filenames: list[str] = []
    for pid in range(1, num_patients + 1):
        n_images = int(rng.integers(1, max_images_per_patient + 1))
        for follow_up in range(n_images):
            fname = f"{pid:08d}_{follow_up:03d}.png"
            filenames.append(fname)
            # 40% no-finding, else 1-3 random pathologies.
            if rng.random() < 0.4:
                findings = NO_FINDING
            else:
                k = int(rng.integers(1, 4))
                picks = rng.choice(PATHOLOGIES, size=k, replace=False)
                findings = "|".join(picks)
            rows.append(
                {
                    "Image Index": fname,
                    "Finding Labels": findings,
                    "Patient ID": pid,
                    "OriginalImage Width": 1024,
                    "Height": 1024,
                }
            )

    data_entry = pd.DataFrame(rows)
    data_entry_path = out_dir / "Data_Entry_2017.csv"
    data_entry.to_csv(data_entry_path, index=False)

    # Bounding boxes for a random subset, labelled with bbox pathologies.
    n_bbox = max(4, int(len(filenames) * bbox_fraction))
    bbox_files = rng.choice(filenames, size=n_bbox, replace=False)
    bbox_rows = []
    for fname in bbox_files:
        label = rng.choice(BBOX_PATHOLOGIES)
        w = float(rng.integers(60, 500))
        h = float(rng.integers(60, 500))
        x = float(rng.integers(0, 1024 - int(w)))
        y = float(rng.integers(0, 1024 - int(h)))
        bbox_rows.append(
            {"Image Index": fname, "Finding Label": label, "x": x, "y": y, "w": w, "h": h}
        )
    # Mirror the NIH ragged header: a "Bbox [x" style plus trailing empty columns.
    bbox_df = pd.DataFrame(bbox_rows)
    bbox_df.columns = ["Image Index", "Finding Label", "Bbox [x", "y", "w", "h]"]
    bbox_df["Unnamed: 6"] = ""  # trailing empty column like the real file
    bbox_csv_path = out_dir / "BBox_List_2017.csv"
    bbox_df.to_csv(bbox_csv_path, index=False)

    if write_images:
        from PIL import Image

        for fname in filenames:
            arr = rng.integers(0, 256, size=(image_size, image_size), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(image_dir / fname)

    return {
        "data_entry": data_entry_path,
        "bbox_csv": bbox_csv_path,
        "image_dir": image_dir,
    }
