import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import h5py

from generate_synthetic_data import SyntheticDataGenerator


def test_synthetic_dataset_has_expected_groups(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "data" / "params.yaml"
    generator = SyntheticDataGenerator(config_path)
    generator.output_dir = tmp_path
    generator.output_path = tmp_path / "test_dataset.h5"

    # Use enough scan vectors so that every split is non-empty.
    path = generator.generate(n_scan_vectors=10, n_points_per_vector=27)

    assert path.exists()
    with h5py.File(path, "r") as f:
        for split in ("train", "val", "test"):
            assert split in f
            assert "inputs" in f[split]
            assert "outputs" in f[split]
            assert "coordinates" in f[split]
            assert "time" in f[split]
            assert "scan_vectors" in f[split]
        assert "metadata" in f
        assert f["metadata"].attrs["n_total"] > 0

        # Scaler parameters should be present.
        assert "scaler_params" in f["metadata"]
        for group_name in ("scan_vectors", "coordinates"):
            assert group_name in f["metadata/scaler_params"]
            assert "mean" in f[f"metadata/scaler_params/{group_name}"]
            assert "std" in f[f"metadata/scaler_params/{group_name}"]


def test_scan_vector_ids_do_not_leak_between_splits(tmp_path):
    """Splitting by scan-vector index must keep every ID in exactly one split."""
    config_path = Path(__file__).resolve().parents[1] / "data" / "params.yaml"
    generator = SyntheticDataGenerator(config_path)
    generator.output_dir = tmp_path
    generator.output_path = tmp_path / "leakage_test.h5"

    generator.generate(n_scan_vectors=10, n_points_per_vector=27)

    with h5py.File(generator.output_path, "r") as f:
        train_ids = set(f["metadata/train_scan_ids"][:])
        val_ids = set(f["metadata/val_scan_ids"][:])
        test_ids = set(f["metadata/test_scan_ids"][:])

    assert len(train_ids) > 0
    assert len(val_ids) > 0
    assert len(test_ids) > 0
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
