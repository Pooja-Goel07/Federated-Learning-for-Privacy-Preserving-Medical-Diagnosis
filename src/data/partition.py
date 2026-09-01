"""Split ChestX-ray14 across simulated hospital nodes.

Produces, from a parsed Data_Entry DataFrame, a reproducible manifest mapping each
node -> list of image filenames, plus a held-out global test set. Two schemes:

  * IID          : patients dealt roughly evenly across nodes (each node sees the
                   same label distribution in expectation).
  * non-IID      : Dirichlet label-skew (Hsu et al. 2019). For each dominant-label
                   class, the patients of that class are split across nodes with
                   proportions drawn from Dir(alpha); small alpha -> high skew.

Partitioning is done at the PATIENT level so no patient's images leak across nodes
or into the test set (standard practice for ChestX-ray14). The manifest is a plain
JSON file (data/partitions/) so runs are reproducible and inspectable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import COL_IMAGE_INDEX, COL_PATIENT_ID
from .labels import load_official_split


@dataclass
class PartitionManifest:
    scheme: str  # 'iid' | 'dirichlet'
    num_nodes: int
    seed: int
    alpha: float | None
    test_frac: float
    nodes: dict[str, list[str]]  # node_id (str) -> image filenames
    test: list[str]
    node_label_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @staticmethod
    def load(path: str | Path) -> "PartitionManifest":
        return PartitionManifest(**json.loads(Path(path).read_text()))


def _patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per patient with a single dominant label (its mode)."""
    return (
        df.groupby(COL_PATIENT_ID)
        .agg(dominant_label=("dominant_label", lambda s: s.mode().iat[0]))
        .reset_index()
    )


def _split_patients_train_test(
    df: pd.DataFrame, data_root: str | Path | None, test_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, set[str]]:
    """Return (trainval_patient_ids, test_image_filenames).

    Uses the official NIH split lists when available under ``data_root``; otherwise
    holds out a random ``test_frac`` of patients.
    """
    if data_root is not None:
        trainval_imgs, test_imgs = load_official_split(data_root)
        if trainval_imgs is not None and test_imgs is not None:
            test_mask = df[COL_IMAGE_INDEX].isin(test_imgs)
            trainval_pids = df.loc[~test_mask, COL_PATIENT_ID].unique()
            test_files = df.loc[test_mask, COL_IMAGE_INDEX].tolist()
            return trainval_pids, set(test_files)

    patients = df[COL_PATIENT_ID].unique()
    rng.shuffle(patients)
    n_test = int(round(len(patients) * test_frac))
    test_pids = set(patients[:n_test].tolist())
    test_files = set(df.loc[df[COL_PATIENT_ID].isin(test_pids), COL_IMAGE_INDEX])
    trainval_pids = patients[n_test:]
    return trainval_pids, test_files


def _assign_iid(patient_ids: np.ndarray, num_nodes: int, rng: np.random.Generator) -> list[list]:
    order = patient_ids.copy()
    rng.shuffle(order)
    return [list(chunk) for chunk in np.array_split(order, num_nodes)]


def _assign_dirichlet(
    patients: pd.DataFrame, num_nodes: int, alpha: float, rng: np.random.Generator
) -> list[list]:
    """Distribute patients across nodes with Dirichlet label-skew.

    For each dominant-label class, draw a length-`num_nodes` proportion vector from
    Dir(alpha) and split that class's (shuffled) patients accordingly.
    """
    buckets: list[list] = [[] for _ in range(num_nodes)]
    for _, group in patients.groupby("dominant_label"):
        pids = group[COL_PATIENT_ID].to_numpy().copy()  # writable (numpy 2.x views are read-only)
        rng.shuffle(pids)
        proportions = rng.dirichlet([alpha] * num_nodes)
        # cut points along the shuffled patient list
        cuts = (np.cumsum(proportions) * len(pids)).astype(int)[:-1]
        for node_i, chunk in enumerate(np.split(pids, cuts)):
            buckets[node_i].extend(chunk.tolist())
    return buckets


def build_partition(
    df: pd.DataFrame,
    num_nodes: int = 3,
    scheme: str = "iid",
    alpha: float | None = 0.5,
    test_frac: float = 0.2,
    seed: int = 42,
    data_root: str | Path | None = None,
) -> PartitionManifest:
    """Build a node partition manifest from a parsed Data_Entry DataFrame.

    Args:
        df: output of ``labels.load_data_entry`` (must have dominant_label, Patient ID).
        num_nodes: number of simulated hospital nodes (code is node-count-agnostic).
        scheme: 'iid' or 'dirichlet'.
        alpha: Dirichlet concentration (only for scheme='dirichlet'; lower=more skew).
        test_frac: patient fraction held out as the global test set (ignored if the
            official NIH split lists are found under data_root).
        seed: RNG seed for reproducibility.
        data_root: dir possibly containing train_val_list.txt / test_list.txt.
    """
    if scheme not in ("iid", "dirichlet"):
        raise ValueError(f"unknown scheme {scheme!r}")
    rng = np.random.default_rng(seed)

    trainval_pids, test_files = _split_patients_train_test(df, data_root, test_frac, rng)
    patients = _patient_table(df[df[COL_PATIENT_ID].isin(trainval_pids)])

    if scheme == "iid":
        node_patient_lists = _assign_iid(patients[COL_PATIENT_ID].to_numpy(), num_nodes, rng)
    else:
        if alpha is None:
            raise ValueError("scheme='dirichlet' requires alpha")
        node_patient_lists = _assign_dirichlet(patients, num_nodes, alpha, rng)

    # Expand patient ids -> image filenames per node, and tally label counts.
    files_by_patient = df.groupby(COL_PATIENT_ID)[COL_IMAGE_INDEX].apply(list)
    nodes: dict[str, list[str]] = {}
    node_label_counts: dict[str, dict[str, int]] = {}
    for node_i, pids in enumerate(node_patient_lists):
        files: list[str] = []
        for pid in pids:
            files.extend(files_by_patient.get(pid, []))
        nodes[str(node_i)] = files
        sub = df[df[COL_IMAGE_INDEX].isin(files)]
        node_label_counts[str(node_i)] = (
            sub["dominant_label"].value_counts().to_dict()
        )

    return PartitionManifest(
        scheme=scheme,
        num_nodes=num_nodes,
        seed=seed,
        alpha=alpha if scheme == "dirichlet" else None,
        test_frac=test_frac,
        nodes=nodes,
        test=sorted(test_files),
        node_label_counts=node_label_counts,
    )
