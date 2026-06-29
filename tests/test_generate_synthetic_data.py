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

    path = generator.generate(n_scan_vectors=5, n_points_per_vector=27)

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
