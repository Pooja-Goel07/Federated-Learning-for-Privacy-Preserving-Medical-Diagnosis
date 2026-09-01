"""Turn a config into datasets / dataloaders and a partition manifest.

Central place that ties labels + severity proxy + image index + partition together,
so the centralised baseline (Task 4), single-site baselines (Task 5) and the
federated clients (Task 6) all draw from the *same* partitioned data and the *same*
held-out test set — a prerequisite for fair floor/ceiling/federated comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

    image_index = build_image_index(paths["image_dir"])

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


def _records_for(df: pd.DataFrame, files: list[str]) -> pd.DataFrame:
    return df[df["Image Index"].isin(set(files))]


def _loader(dataset: ChestXray14Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
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
