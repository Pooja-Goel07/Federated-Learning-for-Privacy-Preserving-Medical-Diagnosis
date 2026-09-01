"""Torch-free smoke tests for the data pipeline (Task 2).

Run directly:  python tests/test_data_pipeline.py
(Also importable by pytest — each check is an assert-based function.)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.constants import PATHOLOGIES, SEVERITY_CLASSES
from src.data.labels import load_data_entry
from src.data.partition import PartitionManifest, build_partition
from src.data.severity import derive_severity
from src.data.synthetic import make_synthetic_dataset


def _prepare(tmp: Path):
    paths = make_synthetic_dataset(tmp, num_patients=80, seed=1, write_images=False)
    df = load_data_entry(paths["data_entry"])
    return paths, df


def test_labels(df) -> None:
    for col in PATHOLOGIES:
        assert col in df.columns, f"missing label column {col}"
    assert "dominant_label" in df.columns
    assert "Patient ID" in df.columns
    # Binary label matrix.
    assert df[PATHOLOGIES].isin([0, 1]).all().all()
    print(f"  labels: {len(df)} images, {df['Patient ID'].nunique()} patients OK")


def test_severity(paths, df) -> None:
    proxy = derive_severity(paths["bbox_csv"], data_entry=df, method="tercile")
    assert set(proxy.table["severity_class"]).issubset(set(SEVERITY_CLASSES))
    low, high = proxy.thresholds
    assert 0.0 <= low <= high, f"thresholds not ordered: {proxy.thresholds}"
    assert proxy.table["Image Index"].is_unique, "one severity row per image expected"
    assert (proxy.table["rel_area"] > 0).all()
    print(f"  severity: {len(proxy.table)} bbox imgs, thresholds={proxy.thresholds} OK")


def test_partition_invariants(df, scheme: str, **kw) -> None:
    m = build_partition(df, num_nodes=3, scheme=scheme, seed=7, **kw)
    all_train = [f for files in m.nodes.values() for f in files]
    # No image appears on two nodes.
    assert len(all_train) == len(set(all_train)), f"{scheme}: duplicate images across nodes"
    # Train nodes and test set are disjoint.
    assert set(all_train).isdisjoint(set(m.test)), f"{scheme}: train/test overlap"
    # Coverage: every image is either on a node or in test.
    assert len(set(all_train) | set(m.test)) == len(df), f"{scheme}: images unaccounted for"
    # Patient-disjoint nodes (no patient leaks across nodes).
    pid_of = dict(zip(df["Image Index"], df["Patient ID"]))
    node_of_patient: dict[int, str] = {}
    for node_id, files in m.nodes.items():
        for f in files:
            pid = pid_of[f]
            assert node_of_patient.setdefault(pid, node_id) == node_id, (
                f"{scheme}: patient {pid} split across nodes"
            )
    sizes = {k: len(v) for k, v in m.nodes.items()}
    print(f"  partition[{scheme}]: node sizes {sizes}, test={len(m.test)} OK")
    return m


def test_manifest_roundtrip(m: PartitionManifest, tmp: Path) -> None:
    path = m.save(tmp / "manifest.json")
    reloaded = PartitionManifest.load(path)
    assert reloaded.nodes == m.nodes and reloaded.test == m.test
    print("  manifest save/load roundtrip OK")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths, df = _prepare(tmp)
        test_labels(df)
        test_severity(paths, df)
        test_partition_invariants(df, "iid")
        m = test_partition_invariants(df, "dirichlet", alpha=0.3)
        test_manifest_roundtrip(m, tmp)
    print("ALL DATA-PIPELINE TESTS PASSED")


if __name__ == "__main__":
    main()
