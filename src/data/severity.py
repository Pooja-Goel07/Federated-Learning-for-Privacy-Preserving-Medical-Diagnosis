"""Heuristic severity-proxy derivation from NIH BBox_List_2017.csv.

IMPORTANT — this is a PROXY, not clinical ground truth (CLAUDE.md §4). NIH
ChestX-ray14 has no native severity labels. We approximate per-image severity by
the *relative lesion size*: the bounding-box area as a fraction of the image area,
then bin into mild / moderate / severe. Larger relative lesion -> higher severity.

Caveats (must be stated as limitations in study.md):
  * Box area is a crude stand-in for disease burden; it ignores opacity/texture.
  * We normalise by full image area, not a segmented lung field (no lung mask in
    the release), so cardiothoracic vs peripheral lesions aren't area-corrected.
  * Only ~984 images (8 pathologies) have boxes, so the severity head is trained on
    a small subset; all other images carry SEVERITY_IGNORE_INDEX.
  * When an image has multiple boxes we take the max relative area (most severe).

The binning thresholds are returned so they can be logged verbatim in study.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    BBOX_LABEL_ALIASES,
    COL_IMAGE_INDEX,
    COL_ORIG_HEIGHT,
    COL_ORIG_WIDTH,
    DEFAULT_IMAGE_SIZE_PX,
    SEVERITY_CLASSES,
    SEVERITY_TO_INDEX,
)


@dataclass
class SeverityProxy:
    """Result of severity derivation.

    Attributes:
        table: per-image DataFrame with columns
            ['Image Index', 'bbox_label', 'rel_area', 'severity_class', 'severity_index'].
        thresholds: the two cut points (low/high) on relative area separating
            mild|moderate and moderate|severe — log these in study.md.
        method: 'tercile' or 'fixed'.
    """

    table: pd.DataFrame
    thresholds: tuple[float, float]
    method: str


def _load_bbox_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load BBox_List_2017.csv robustly.

    The NIH file has a ragged header (`Bbox [x,y,w,h]` split across columns plus
    trailing empty columns). We keep only the first six meaningful columns.
    """
    df = pd.read_csv(csv_path)
    # Drop the trailing unnamed/empty columns NIH ships.
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if df.shape[1] < 6:
        raise ValueError(
            f"{csv_path}: expected >=6 columns (image, label, x, y, w, h), got {df.shape[1]}"
        )
    df = df.iloc[:, :6]
    df.columns = ["Image Index", "bbox_label", "x", "y", "w", "h"]
    df["bbox_label"] = df["bbox_label"].str.strip().replace(BBOX_LABEL_ALIASES)
    for c in ("x", "y", "w", "h"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["w", "h"])


def derive_severity(
    bbox_csv: str | Path,
    data_entry: pd.DataFrame | None = None,
    method: str = "tercile",
    fixed_thresholds: tuple[float, float] | None = None,
) -> SeverityProxy:
    """Derive the severity proxy for the bbox subset.

    Args:
        bbox_csv: path to BBox_List_2017.csv.
        data_entry: parsed Data_Entry_2017.csv DataFrame (for per-image original
            width/height). If None, image area falls back to DEFAULT_IMAGE_SIZE_PX^2.
        method: 'tercile' (data-driven 33/67 percentiles, default) or 'fixed'.
        fixed_thresholds: (low, high) relative-area cut points when method='fixed'.

    Returns:
        SeverityProxy with one row per image in the bbox subset.
    """
    boxes = _load_bbox_raw(bbox_csv)

    # Per-image image area (px^2): use NIH original dimensions when available.
    img_area = _image_area_lookup(boxes, data_entry)
    boxes["box_area"] = boxes["w"] * boxes["h"]
    boxes["rel_area"] = boxes["box_area"] / boxes["Image Index"].map(img_area)

    # One severity value per image: the largest relative lesion (most severe).
    per_image = (
        boxes.groupby("Image Index", as_index=False)
        .agg(bbox_label=("bbox_label", "first"), rel_area=("rel_area", "max"))
    )

    thresholds = _resolve_thresholds(per_image["rel_area"].to_numpy(), method, fixed_thresholds)
    per_image["severity_class"] = _bin(per_image["rel_area"].to_numpy(), thresholds)
    per_image["severity_index"] = per_image["severity_class"].map(SEVERITY_TO_INDEX).astype(int)

    cols = ["Image Index", "bbox_label", "rel_area", "severity_class", "severity_index"]
    return SeverityProxy(table=per_image[cols], thresholds=thresholds, method=method)


def _image_area_lookup(
    boxes: pd.DataFrame, data_entry: pd.DataFrame | None
) -> "pd.Series":
    """Map Image Index -> image area in px^2."""
    default_area = float(DEFAULT_IMAGE_SIZE_PX**2)
    if data_entry is None or COL_ORIG_WIDTH not in data_entry.columns:
        return pd.Series(default_area, index=boxes["Image Index"].unique())
    de = data_entry[[COL_IMAGE_INDEX, COL_ORIG_WIDTH, COL_ORIG_HEIGHT]].copy()
    de["area"] = (
        pd.to_numeric(de[COL_ORIG_WIDTH], errors="coerce")
        * pd.to_numeric(de[COL_ORIG_HEIGHT], errors="coerce")
    ).fillna(default_area)
    return de.set_index(COL_IMAGE_INDEX)["area"]


def _resolve_thresholds(
    rel_area: np.ndarray, method: str, fixed_thresholds: tuple[float, float] | None
) -> tuple[float, float]:
    if method == "tercile":
        low, high = np.quantile(rel_area, [1 / 3, 2 / 3])
        return float(low), float(high)
    if method == "fixed":
        if fixed_thresholds is None:
            raise ValueError("method='fixed' requires fixed_thresholds=(low, high)")
        return float(fixed_thresholds[0]), float(fixed_thresholds[1])
    raise ValueError(f"unknown severity binning method: {method!r}")


def _bin(rel_area: np.ndarray, thresholds: tuple[float, float]) -> "pd.Series":
    low, high = thresholds
    # digitize: <low -> mild(0), [low,high) -> moderate(1), >=high -> severe(2)
    idx = np.digitize(rel_area, bins=[low, high], right=False)
    return pd.Series([SEVERITY_CLASSES[i] for i in idx])
