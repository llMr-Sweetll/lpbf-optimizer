"""Tests for the Optuna hyperparameter search script."""

import json
import os
import subprocess
import sys

import pytest
import yaml

optuna = pytest.importorskip("optuna")


def test_hyperparameter_search_script(tmp_path):
    """Run the HPO script for two trials on a tiny dataset."""
    config = {
        'data': {'processed_data_path': str(tmp_path / 'data.h5')},
        'material_properties': {
            'rho': 4430.0, 'cp': 526.3, 'k': 6.7, 'eta': 0.35, 'r0': 0.05,
            'Hm': 286000.0, 'Ts': 1604.0, 'Tl': 1660.0,
            'E': 110.0, 'nu': 0.34, 'alpha': 8.6e-6, 'sigma_y': 1100.0,
        },
        'model': {
            'input_dim': 10, 'output_dim': 3,
            'hidden_width': 16, 'hidden_depth': 2, 'dropout_rate': 0.0,
            'apply_output_bounds': False,
        },
        'training': {
            'n_epochs': 2,
            'batch_size': 8,
            'random_seed': 42,
            'device': 'cpu',
            'optimizer': {'type': 'adam', 'learning_rate': 0.01, 'weight_decay': 0.0},
            'scheduler': {'type': 'none'},
            'lambda_heat': 0.0,
            'lambda_stress': 0.0,
            'lambda_porosity': 0.0,
            'lambda_geometry': 0.0,
            'clip_grad': False,
            'checkpoint_freq': 100,
            'output_dir': str(tmp_path / 'models'),
            'plot_freq': 100,
            'print_freq': 100,
            'hpo_epochs': 2,
        },
        'optimizer': {
            'param_bounds': {
                'P': [150.0, 400.0],
                'v': [500.0, 1500.0],
                'h': [0.05, 0.15],
                'theta': [0.0, 90.0],
                'l_island': [2.0, 10.0],
                'layer_thickness': [0.02, 0.06],
            },
            'objectives': ['residual_stress', 'porosity', 'geometric_accuracy'],
        },
    }
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Generate tiny dataset
    from generate_synthetic_data import SyntheticDataGenerator
    generator = SyntheticDataGenerator(str(config_path))
    generator.generate(n_scan_vectors=8, n_points_per_vector=16)

    output_dir = tmp_path / 'hpo'
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src'
    result = subprocess.run(
        [sys.executable, 'scripts/hyperparameter_search.py',
         '--config', str(config_path), '--n-trials', '2',
         '--output-dir', str(output_dir), '--num-threads', '1'],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert 'Best validation loss' in result.stdout

    assert (output_dir / 'best_config.yaml').exists()
    assert (output_dir / 'study_summary.json').exists()
    with open(output_dir / 'study_summary.json', 'r') as f:
        summary = json.load(f)
    assert summary['n_trials'] == 2
