import sys
from pathlib import Path

import h5py
import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "src" / "pinn"))

from model import PINN  # noqa: E402

from generate_synthetic_data import SyntheticDataGenerator  # noqa: E402


def test_scaler_round_trip_via_model(tmp_path):
    """A PINN built with scaler params should round-trip normalise inputs."""
    config_path = repo_root / "data" / "params.yaml"
    generator = SyntheticDataGenerator(config_path)
    generator.output_dir = tmp_path
    generator.output_path = tmp_path / "normalisation_test.h5"

    generator.generate(n_scan_vectors=10, n_points_per_vector=27)

    with h5py.File(generator.output_path, "r") as f:
        scaler = f["metadata/scaler_params"]
        scaler_params = {
            "scan_vectors": {
                "mean": scaler["scan_vectors/mean"][:],
                "std": scaler["scan_vectors/std"][:],
            },
            "coordinates": {
                "mean": scaler["coordinates/mean"][:],
                "std": scaler["coordinates/std"][:],
            },
        }
        original = torch.tensor(f["train/inputs"][:], dtype=torch.float32)

    input_dim = original.shape[1]
    model = PINN(
        input_dim=input_dim,
        output_dim=3,
        width=32,
        depth=2,
        scaler_params=scaler_params,
    )

    normalised = model.transform_inputs(original)
    recovered = model.inverse_transform_inputs(normalised)

    # The scan-vector and coordinate slices should recover exactly; time is
    # passed through unchanged.
    np.testing.assert_allclose(
        original.numpy(), recovered.numpy(), atol=1e-6, rtol=1e-5
    )

    # A model without scaler parameters should act as the identity transform.
    unscaled_model = PINN(input_dim=input_dim, output_dim=3, width=32, depth=2)
    assert torch.allclose(original, unscaled_model.transform_inputs(original))
