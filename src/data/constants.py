"""Dataset-level constants for NIH ChestX-ray14.

Single source of truth for label names, the bounding-box pathology subset used to
derive the severity proxy, and the CSV column names shipped by the NIH release.
"""

from __future__ import annotations

# The 14 pathology labels of ChestX-ray14 (canonical order). "No Finding" is NOT
# one of the 14 — it is the implicit all-negative case.
PATHOLOGIES: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NUM_PATHOLOGIES: int = len(PATHOLOGIES)
PATHOLOGY_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(PATHOLOGIES)}

# Token used in the `Finding Labels` field for the all-negative case.
NO_FINDING: str = "No Finding"

# The 8 pathologies that have hand-annotated bounding boxes in BBox_List_2017.csv
# (~984 images). These are the only images that receive a severity proxy.
BBOX_PATHOLOGIES: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
]

# BBox_List_2017.csv spells "Infiltration" as "Infiltrate"; normalise on load.
BBOX_LABEL_ALIASES: dict[str, str] = {"Infiltrate": "Infiltration"}

# Ordinal severity classes for the proxy grading head (mild < moderate < severe).
SEVERITY_CLASSES: list[str] = ["mild", "moderate", "severe"]
SEVERITY_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(SEVERITY_CLASSES)}
# Sentinel target for images without a severity proxy (all except the bbox subset).
SEVERITY_IGNORE_INDEX: int = -1

# ---- CSV schemas (NIH release) ----
# Data_Entry_2017.csv columns actually used downstream.
COL_IMAGE_INDEX = "Image Index"
COL_FINDING_LABELS = "Finding Labels"
COL_PATIENT_ID = "Patient ID"
COL_ORIG_WIDTH = "OriginalImage Width"
COL_ORIG_HEIGHT = "Height"  # NIH header is "Height" (paired with "OriginalImage Width")

# Default frontal image dimensions if the per-image size is unavailable.
DEFAULT_IMAGE_SIZE_PX: int = 1024

# Official NIH split manifests (patient-disjoint train/val vs test).
TRAINVAL_LIST_FILE = "train_val_list.txt"
TEST_LIST_FILE = "test_list.txt"
