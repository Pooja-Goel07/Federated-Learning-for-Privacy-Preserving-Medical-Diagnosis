"""Dataset loading, IID/non-IID partitioning, severity-proxy derivation.

Public API (torch-free unless noted):
    load_data_entry, label_matrix, load_official_split   -- labels.py
    derive_severity, SeverityProxy                        -- severity.py
    build_partition, PartitionManifest                    -- partition.py
    ChestXray14Dataset, build_transforms,                 -- dataset.py (needs torch)
    build_image_index, dataset_for_node
"""

from __future__ import annotations

from .labels import label_matrix, load_data_entry, load_official_split
from .partition import PartitionManifest, build_partition
from .severity import SeverityProxy, derive_severity

__all__ = [
    "load_data_entry",
    "label_matrix",
    "load_official_split",
    "derive_severity",
    "SeverityProxy",
    "build_partition",
    "PartitionManifest",
]

# dataset.py is intentionally NOT imported here so that torch-free workflows
# (partitioning, metadata prep) don't require torch. Import it explicitly:
#   from src.data.dataset import ChestXray14Dataset, build_image_index
