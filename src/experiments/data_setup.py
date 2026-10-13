"""Turn a config into datasets / dataloaders and a partition manifest.

Central place that ties labels + severity proxy + image index + partition together,
so the centralised baseline (Task 4), single-site baselines (Task 5) and the
federated clients (Task 6) all draw from the *same* partitioned data and the *same*
held-out test set — a prerequisite for fair floor/ceiling/federated comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.data.dataset import ChestXray14Dataset, build_image_index
from src.data.labels import load_data_entry
from src.data.partition import PartitionManifest, build_partition
from src.data.severity import derive_severity

from .config import cfg_get


@dataclass
class DataContext:
    data_entry: pd.DataFrame
    image_index: dict[str, Path]
    severity_map: dict[str, int]
    manifest: PartitionManifest
    severity_thresholds: tuple[float, float]


def _resolve_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "data_entry": root / "Data_Entry_2017.csv",
        "bbox": root / "BBox_List_2017.csv",
        "image_dir": root,
    }


def build_data_context(config: dict, data_root: str | Path | None = None) -> DataContext:
    """Build (or load) everything the loaders need from a config.

    Args:
        config: parsed experiment config.
        data_root: override for config['data']['root'] (used by tests to point at a
            synthetic dataset dir).
    """
    root = Path(data_root) if data_root is not None else Path(cfg_get(config, "data.root", "data/raw"))
    paths = _resolve_paths(root)

    df = load_data_entry(paths["data_entry"])

    severity_map: dict[str, int] = {}
    thresholds = (float("nan"), float("nan"))
    if cfg_get(config, "data.severity.enabled", True) and paths["bbox"].exists():
        method = cfg_get(config, "data.severity.binning", "tercile")
        proxy = derive_severity(paths["bbox"], data_entry=df, method=method)
        severity_map = dict(zip(proxy.table["Image Index"], proxy.table["severity_index"]))
        thresholds = proxy.thresholds

    # Images may live on a different root than the CSVs (e.g. on Kaggle the labels and
    # the resized-image dataset mount at separate /kaggle/input paths). data.image_root
    # overrides where images are scanned; default is the CSV root.
    image_root = cfg_get(config, "data.image_root")
    image_index = build_image_index(Path(image_root) if image_root else paths["image_dir"])

    # Restrict to images actually present on disk. This makes partial downloads work
    # out of the box (e.g. a few NIH image zips instead of all 45 GB): rows whose PNG
    # was not downloaded are dropped instead of crashing the Dataset later.
    present = set(image_index.keys())
    if present:
        n_total = len(df)
        df = df[df["Image Index"].isin(present)].reset_index(drop=True)
        if len(df) < n_total:
            print(f"[data] {len(df)}/{n_total} labelled images found on disk; using those")

    # Optional patient-aware subsampling (keeps GPU runs tractable on Colab). Whole
    # patients are kept/dropped (no patient leakage); bbox/severity patients are kept
    # first so the severity head still has data.
    sub = cfg_get(config, "data.subsample", {}) or {}
    max_images = sub.get("max_images") if isinstance(sub, dict) else None
    if max_images and len(df) > int(max_images):
        df = _subsample_patients(
            df,
            severity_images=set(severity_map),
            max_images=int(max_images),
            keep_bbox=bool(sub.get("keep_bbox", True)),
            seed=int(cfg_get(config, "experiment.seed", 42)),
        )
        print(f"[data] subsampled to {len(df)} images across {df['Patient ID'].nunique()} patients")

    manifest = _build_or_load_manifest(config, df, root)
    return DataContext(
        data_entry=df,
        image_index=image_index,
        severity_map=severity_map,
        manifest=manifest,
        severity_thresholds=thresholds,
    )


def _build_or_load_manifest(config: dict, df: pd.DataFrame, data_root: Path) -> PartitionManifest:
    """Load a cached manifest if configured, else build one and save it."""
    manifest_path = cfg_get(config, "partition.manifest_path")
    if manifest_path and Path(manifest_path).exists():
        return PartitionManifest.load(manifest_path)

    # config uses 'iid' | 'non_iid'; the partitioner speaks 'iid' | 'dirichlet'.
    config_scheme = cfg_get(config, "partition.scheme", "iid")
    scheme = "dirichlet" if config_scheme == "non_iid" else "iid"
    manifest = build_partition(
        df,
        num_nodes=int(cfg_get(config, "partition.num_nodes", 3)),
        scheme=scheme,
        alpha=cfg_get(config, "partition.non_iid.alpha", 0.5),
        test_frac=float(cfg_get(config, "partition.test_frac", 0.2)),
        seed=int(cfg_get(config, "experiment.seed", 42)),
        data_root=data_root,
    )
    if manifest_path:
        manifest.save(manifest_path)
    return manifest


def compute_pos_weight(
    df: pd.DataFrame, files: list[str] | set[str], clamp: float = 50.0
):
    """Per-class BCE ``pos_weight`` = (#neg / #pos) over the given training files.

    Counters ChestX-ray14's heavy class imbalance (most labels are negative), which
    otherwise pins every sigmoid near 0 and flattens AUC. Classes with no positives in
    this split get weight 1.0; weights are clamped to ``clamp`` to avoid blow-ups on
    ultra-rare labels. Returns a FloatTensor[14], or None if there are no rows.
    """
    import torch

    from src.data.constants import PATHOLOGIES

    sub = df[df["Image Index"].isin(set(files))]
    if len(sub) == 0:
        return None
    pos = sub[PATHOLOGIES].to_numpy(dtype="float64").sum(axis=0)
    neg = len(sub) - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    w = np.clip(w, 1.0, clamp)
    return torch.tensor(w, dtype=torch.float32)


def _subsample_patients(
    df: pd.DataFrame,
    severity_images: set[str],
    max_images: int,
    keep_bbox: bool,
    seed: int,
) -> pd.DataFrame:
    """Keep whole patients up to ~``max_images`` images, bbox patients first.

    Patient-level (not image-level) so train/test stay patient-disjoint downstream.
    All patients owning a severity/bbox image are kept first (so the severity head
    still trains), then random other patients are added until the image budget fills.
    """
    pid_col = "Patient ID"
    sizes = df.groupby(pid_col).size()

    keep_pids: set = set()
    if keep_bbox and severity_images:
        keep_pids = set(df[df["Image Index"].isin(severity_images)][pid_col].unique())

    total = int(sizes[list(keep_pids)].sum()) if keep_pids else 0
    others = np.array([p for p in sizes.index if p not in keep_pids])
    others = np.random.default_rng(seed).permutation(others)

    selected = set(keep_pids)
    for p in others:
        if total >= max_images:
            break
        selected.add(p)
        total += int(sizes[p])
    return df[df[pid_col].isin(selected)].reset_index(drop=True)


def _records_for(df: pd.DataFrame, files: list[str]) -> pd.DataFrame:
    return df[df["Image Index"].isin(set(files))]


def _loader(dataset: ChestXray14Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    # BatchNorm raises on a size-1 batch in train mode. Drop a lone trailing sample
    # only while training (shuffle=True) and only when a full batch still remains, so
    # test batches and tiny node partitions are never silently emptied.
    n = len(dataset)
    drop_last = shuffle and n > batch_size and (n % batch_size == 1)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=drop_last,
    )


def make_dataset(
    ctx: DataContext, files: list[str], config: dict, train: bool
) -> ChestXray14Dataset:
    image_size = int(cfg_get(config, "data.image_size", 224))
    return ChestXray14Dataset(
        records=_records_for(ctx.data_entry, files),
        image_index=ctx.image_index,
        severity_map=ctx.severity_map,
        image_size=image_size,
        train=train,
    )


def centralized_loaders(ctx: DataContext, config: dict) -> tuple[DataLoader, DataLoader]:
    """Full-data train loader (union of all nodes) + shared test loader."""
    batch_size = int(cfg_get(config, "training.batch_size", 32))
    num_workers = int(cfg_get(config, "training.num_workers", 0))

    train_files = [f for files in ctx.manifest.nodes.values() for f in files]
    train_ds = make_dataset(ctx, train_files, config, train=True)
    test_ds = make_dataset(ctx, ctx.manifest.test, config, train=False)
    return (
        _loader(train_ds, batch_size, shuffle=True, num_workers=num_workers),
        _loader(test_ds, batch_size, shuffle=False, num_workers=num_workers),
    )


def node_loader(ctx: DataContext, node_id: str, config: dict, train: bool = True) -> DataLoader:
    """Loader for a single node's local data (single-site baseline / FL client)."""
    batch_size = int(cfg_get(config, "training.batch_size", 32))
    num_workers = int(cfg_get(config, "training.num_workers", 0))
    ds = make_dataset(ctx, ctx.manifest.nodes[node_id], config, train=train)
    return _loader(ds, batch_size, shuffle=train, num_workers=num_workers)


def test_loader(ctx: DataContext, config: dict) -> DataLoader:
    batch_size = int(cfg_get(config, "training.batch_size", 32))
    num_workers = int(cfg_get(config, "training.num_workers", 0))
    ds = make_dataset(ctx, ctx.manifest.test, config, train=False)
    return _loader(ds, batch_size, shuffle=False, num_workers=num_workers)
