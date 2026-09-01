"""PyTorch Dataset for ChestX-ray14 with dual (pathology + severity) targets.

Each item yields ``(image_tensor, targets)`` where::

    targets = {
        "pathology": FloatTensor[14],   # multi-label 0/1 target
        "severity" : LongTensor([]),    # ordinal class in {0,1,2}, or
                                        # SEVERITY_IGNORE_INDEX when no proxy exists
    }

The severity target is only present for the ~984 bbox-subset images; every other
image carries ``SEVERITY_IGNORE_INDEX`` so the severity loss can mask them out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import PATHOLOGIES, SEVERITY_IGNORE_INDEX

# ImageNet normalisation (backbone is ImageNet-pretrained).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    """Standard ChestX-ray14 preprocessing. Grayscale CXR -> 3-channel for the backbone."""
    if train:
        ops = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
        ]
    else:
        ops = [transforms.Resize((image_size, image_size))]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def build_image_index(image_root: str | Path) -> dict[str, Path]:
    """Map image filename -> full path, scanning subfolders once.

    NIH ships images across images_001/ ... images_012/ (or a merged folder). This
    flattens them into a lookup so the Dataset is agnostic to the on-disk layout.
    """
    image_root = Path(image_root)
    index: dict[str, Path] = {}
    for path in image_root.rglob("*.png"):
        index.setdefault(path.name, path)
    return index


class ChestXray14Dataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        image_index: dict[str, Path],
        severity_map: dict[str, int] | None = None,
        transform: transforms.Compose | None = None,
        image_size: int = 224,
        train: bool = True,
    ) -> None:
        """
        Args:
            records: rows for this split/node; must contain 'Image Index' and the
                14 pathology columns (output of labels.load_data_entry, subset).
            image_index: filename -> path map (see ``build_image_index``).
            severity_map: filename -> severity class index (from severity proxy);
                images absent here get SEVERITY_IGNORE_INDEX.
            transform: torchvision transform; defaults to ``build_transforms``.
            image_size: square resize target.
            train: selects the train-time augmentation pipeline when transform is None.
        """
        missing_cols = set(PATHOLOGIES) - set(records.columns)
        if missing_cols:
            raise ValueError(f"records missing pathology columns: {sorted(missing_cols)}")
        self.records = records.reset_index(drop=True)
        self.image_index = image_index
        self.severity_map = severity_map or {}
        self.transform = transform or build_transforms(image_size, train=train)
        self._labels = self.records[PATHOLOGIES].to_numpy(dtype=np.float32)
        self._files = self.records["Image Index"].tolist()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        filename = self._files[i]
        path = self.image_index.get(filename)
        if path is None:
            raise FileNotFoundError(f"image {filename!r} not found in image index")
        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image)

        pathology = torch.from_numpy(self._labels[i])
        severity = torch.tensor(
            self.severity_map.get(filename, SEVERITY_IGNORE_INDEX), dtype=torch.long
        )
        return image_tensor, {"pathology": pathology, "severity": severity}


def dataset_for_node(
    manifest_nodes: dict[str, list[str]],
    node_id: str,
    data_entry: pd.DataFrame,
    image_index: dict[str, Path],
    severity_map: dict[str, int] | None = None,
    train: bool = True,
    image_size: int = 224,
) -> ChestXray14Dataset:
    """Convenience builder: dataset for one node given a partition manifest."""
    files = set(manifest_nodes[node_id])
    records = data_entry[data_entry["Image Index"].isin(files)]
    return ChestXray14Dataset(
        records=records,
        image_index=image_index,
        severity_map=severity_map,
        image_size=image_size,
        train=train,
    )
