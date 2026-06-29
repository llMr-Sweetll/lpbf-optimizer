from pathlib import Path

import yaml


def test_config_loads_and_has_required_keys():
    config_path = Path(__file__).resolve().parents[1] / "data" / "params.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert "material_properties" in cfg
    assert "model" in cfg
    assert cfg["model"]["input_dim"] == 10
    assert cfg["model"]["output_dim"] == 3
    assert "training" in cfg
    assert "data" in cfg
    assert "optimizer" in cfg
    assert "param_bounds" in cfg["optimizer"]
    assert "objectives" in cfg["optimizer"]
