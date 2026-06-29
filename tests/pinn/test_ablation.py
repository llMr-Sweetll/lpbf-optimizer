import sys
from pathlib import Path

import yaml

# Add src/pinn to the path so the standalone modules can be imported.
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src" / "pinn"))

from ablation import AblationStudy, VARIANTS  # noqa: E402


def test_ablation_variants_match_spec():
    """The three ablation variants must differ only in bounds and physics weights."""
    assert set(VARIANTS.keys()) == {"baseline_mlp", "pinn_no_physics", "pinn_physics"}
    assert VARIANTS["baseline_mlp"]["apply_output_bounds"] is False
    assert VARIANTS["pinn_no_physics"]["apply_output_bounds"] is True
    assert VARIANTS["pinn_physics"]["apply_output_bounds"] is True

    assert all(
        v["lambdas"][key] == 0.0
        for v in (VARIANTS["baseline_mlp"], VARIANTS["pinn_no_physics"])
        for key in ("lambda_heat", "lambda_stress", "lambda_porosity", "lambda_geometry")
    )
    assert VARIANTS["pinn_physics"]["lambdas"] is None


def test_ablation_builds_variant_configs(tmp_path):
    """Variant configs should be isolated and carry the expected overrides."""
    config_path = Path(__file__).resolve().parents[2] / "data" / "params.yaml"
    study = AblationStudy(
        config_path=config_path,
        output_dir=tmp_path,
        epochs=2,
    )
    configs = study.build_variant_configs()

    assert set(configs.keys()) == set(VARIANTS.keys())

    for name, cfg in configs.items():
        assert cfg["model"]["apply_output_bounds"] == VARIANTS[name]["apply_output_bounds"]
        assert cfg["training"]["n_epochs"] == 2

    # Baseline and no-physics have all physics lambdas set to zero.
    for name in ("baseline_mlp", "pinn_no_physics"):
        for key in ("lambda_heat", "lambda_stress", "lambda_porosity", "lambda_geometry"):
            assert configs[name]["training"][key] == 0.0

    # Physics variant keeps the lambdas from the base config.
    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    for key in ("lambda_heat", "lambda_stress", "lambda_porosity", "lambda_geometry"):
        assert configs["pinn_physics"]["training"][key] == base_cfg["training"][key]


def test_ablation_output_dirs_are_isolated(tmp_path):
    """Each variant should write to its own ablation sub-directory."""
    config_path = Path(__file__).resolve().parents[2] / "data" / "params.yaml"
    study = AblationStudy(config_path=config_path, output_dir=tmp_path)
    configs = study.build_variant_configs()

    for name, cfg in configs.items():
        expected_suffix = Path("data/models/ablation") / name
        assert Path(cfg["training"]["output_dir"]).resolve().parts[-4:] == expected_suffix.parts
