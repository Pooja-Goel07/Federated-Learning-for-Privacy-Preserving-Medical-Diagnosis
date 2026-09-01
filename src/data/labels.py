"""Parse NIH ChestX-ray14 metadata into a multi-label table.

Reads `Data_Entry_2017.csv` and expands the pipe-separated `Finding Labels` field
into a 14-column binary multi-label matrix, plus a few helper columns used by the
partitioners (patient id, a single "dominant" label for label-skew partitioning).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    COL_FINDING_LABELS,
    COL_IMAGE_INDEX,
    COL_PATIENT_ID,
    NO_FINDING,
    PATHOLOGIES,
    PATHOLOGY_TO_INDEX,
    TEST_LIST_FILE,
    TRAINVAL_LIST_FILE,
)


def load_data_entry(csv_path: str | Path) -> pd.DataFrame:
    """Load Data_Entry_2017.csv and attach a binary column per pathology.

    Returns a DataFrame indexed by row order with, at minimum:
      - `Image Index`   : filename (str)
      - `Patient ID`    : int
      - one 0/1 column per name in ``PATHOLOGIES``
      - `dominant_label`: the partitioning label (see ``dominant_label``)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = {COL_IMAGE_INDEX, COL_FINDING_LABELS} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing expected columns: {sorted(missing)}")

    # Expand the "A|B|C" finding string into one binary column per pathology.
    findings = df[COL_FINDING_LABELS].fillna("").str.split("|")
    label_matrix = np.zeros((len(df), len(PATHOLOGIES)), dtype=np.int8)
    for row_i, labels in enumerate(findings):
        for label in labels:
            idx = PATHOLOGY_TO_INDEX.get(label.strip())
            if idx is not None:
                label_matrix[row_i, idx] = 1
    for name in PATHOLOGIES:
        df[name] = label_matrix[:, PATHOLOGY_TO_INDEX[name]]

    df["num_findings"] = label_matrix.sum(axis=1).astype(int)
    df["dominant_label"] = _dominant_labels(label_matrix)

    if COL_PATIENT_ID not in df.columns:
        # Fall back to deriving patient id from the filename (NNNNNNNN_XXX.png).
        df[COL_PATIENT_ID] = (
            df[COL_IMAGE_INDEX].str.split("_").str[0].astype(int)
        )
    return df


def _dominant_labels(label_matrix: np.ndarray) -> list[str]:
    """Pick one label per image for label-skew partitioning.

    Multi-label images can't be Dirichlet-partitioned directly, so we collapse each
    image to a single categorical class: the first positive pathology in canonical
    order, or ``NO_FINDING`` when the image has no positive label. This is a
    partitioning device only — training still uses the full multi-label target.
    """
    dominant: list[str] = []
    for row in label_matrix:
        positive = np.flatnonzero(row)
        dominant.append(PATHOLOGIES[positive[0]] if positive.size else NO_FINDING)
    return dominant


def label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the (N, 14) int8 multi-label matrix from a parsed DataFrame."""
    return df[PATHOLOGIES].to_numpy(dtype=np.int8)


def load_official_split(
    data_root: str | Path,
) -> tuple[set[str] | None, set[str] | None]:
    """Load the official NIH train/val and test image lists if present.

    Returns (trainval_images, test_images) as sets of filenames, or (None, None)
    when the manifest files are absent (caller then falls back to a patient-wise
    random split).
    """
    data_root = Path(data_root)
    trainval_path = data_root / TRAINVAL_LIST_FILE
    test_path = data_root / TEST_LIST_FILE
    if not trainval_path.exists() or not test_path.exists():
        return None, None
    trainval = {ln.strip() for ln in trainval_path.read_text().splitlines() if ln.strip()}
    test = {ln.strip() for ln in test_path.read_text().splitlines() if ln.strip()}
    return trainval, test
