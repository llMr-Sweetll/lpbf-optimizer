"""Tests for the validation/characterisation module and report generator."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

# Add src/validate to the path so the standalone modules can be imported.
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src" / "validate"))

from characterise import (  # noqa: E402
    LPBFCharacterisation,
    parse_ebsd_csv,
    parse_stress_csv,
    parse_xct_porosity_csv,
)
from report import ValidationReport  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / 'data'


@pytest.fixture
def config_path(tmp_path):
    """Create a minimal config file for LPBFCharacterisation."""
    config = {
        'material_properties': {},
        'validate': {'characterisation_dir': str(tmp_path / 'characterisation')},
    }
    config_file = tmp_path / 'params.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    return config_file


def test_parse_xct_porosity_csv():
    """XCT porosity CSV parser returns required columns and summary stats."""
    df, summary = parse_xct_porosity_csv(DATA_DIR / 'xct_porosity.csv')

    assert list(df.columns) == ['porosity', 'x', 'y', 'z']
    assert len(df) == 5
    assert summary['mean'] == pytest.approx(0.003)
    assert summary['min'] == pytest.approx(0.001)
    assert summary['max'] == pytest.approx(0.005)
    assert summary['std'] > 0


def test_parse_xct_porosity_csv_missing_column():
    """XCT porosity parser raises a clear error when porosity is missing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        f.write('x,y,z\n0,0,0\n')
        path = f.name

    with pytest.raises(ValueError, match="missing required column 'porosity'"):
        parse_xct_porosity_csv(path)

    Path(path).unlink()


def test_parse_ebsd_csv():
    """EBSD CSV parser returns required columns and preserves optional ones."""
    df = parse_ebsd_csv(DATA_DIR / 'ebsd.csv')

    required = ['phi1', 'phi', 'phi2', 'x', 'y']
    assert all(col in df.columns for col in required)
    assert 'ci' in df.columns
    assert 'phase' in df.columns
    assert len(df) == 5


def test_parse_ebsd_csv_missing_column():
    """EBSD parser raises a clear error when required columns are missing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        f.write('phi1,phi,x,y\n0,0,0,0\n')
        path = f.name

    with pytest.raises(ValueError, match='missing required columns'):
        parse_ebsd_csv(path)

    Path(path).unlink()


def test_parse_stress_csv():
    """Stress CSV parser computes von Mises stress and summary statistics."""
    df, summary = parse_stress_csv(DATA_DIR / 'stress.csv')

    assert 'von_mises' in df.columns
    assert summary['mean_xx'] == pytest.approx(375.0)
    assert summary['mean_yy'] == pytest.approx(137.5)
    assert summary['mean_zz'] == pytest.approx(72.5)
    assert summary['mean_von_mises'] > 0
    assert summary['max_von_mises'] >= summary['mean_von_mises']
    assert summary['std_von_mises'] >= 0

    # Spot-check von Mises for the first row
    expected_vm = np.sqrt(
        0.5 * (
            (300.0 - 100.0) ** 2
            + (100.0 - 50.0) ** 2
            + (50.0 - 300.0) ** 2
        )
    )
    assert df['von_mises'].iloc[0] == pytest.approx(expected_vm)


def test_parse_stress_csv_missing_column():
    """Stress parser raises a clear error when sigma_xx is missing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        f.write('x,y,z,sigma_yy\n0,0,0,100\n')
        path = f.name

    with pytest.raises(ValueError, match="missing required column 'sigma_xx'"):
        parse_stress_csv(path)

    Path(path).unlink()


def test_measure_residual_stress(config_path):
    """Residual stress analysis loads a CSV and produces expected metrics."""
    char = LPBFCharacterisation(config_path)
    metrics = char.measure_residual_stress(DATA_DIR / 'stress.csv')

    assert 'mean_von_mises' in metrics
    assert 'max_von_mises' in metrics
    assert 'std_von_mises' in metrics
    assert 'mean_xx' in metrics
    assert 'mean_yy' in metrics
    assert 'mean_zz' in metrics


def test_measure_residual_stress_unsupported_format(config_path):
    """Residual stress analysis raises for unsupported file formats."""
    char = LPBFCharacterisation(config_path)
    with pytest.raises(ValueError, match='Unsupported stress data format'):
        char.measure_residual_stress('stress.txt')


def test_load_ebsd_csv(config_path):
    """EBSD CSV loading through LPBFCharacterisation returns a data dict."""
    char = LPBFCharacterisation(config_path)
    ebsd_data = char.load_ebsd_data(DATA_DIR / 'ebsd.csv')

    assert 'phi1' in ebsd_data
    assert 'phi' in ebsd_data
    assert 'phi2' in ebsd_data
    assert 'x' in ebsd_data
    assert 'y' in ebsd_data
    assert 'ci' in ebsd_data
    assert 'phase' in ebsd_data


def test_load_ebsd_unsupported_format(config_path):
    """EBSD loader raises a clear error for unsupported formats."""
    char = LPBFCharacterisation(config_path)
    with pytest.raises(ValueError, match='Unsupported EBSD file format'):
        char.load_ebsd_data('sample.unknown')


def test_compare_with_predictions_csv(config_path):
    """Prediction comparison from CSV returns MAE/RMSE per quantity."""
    char = LPBFCharacterisation(config_path)

    experimental_data = {
        'stress': {'mean_von_mises': 330.0},
        'porosity': {'porosity': 0.0025},
        'geometric_accuracy': {'accuracy': 0.94},
    }

    comparison = char.compare_with_predictions(
        experimental_data, DATA_DIR / 'predictions.csv'
    )

    for quantity in ['stress', 'porosity', 'geometric_accuracy']:
        assert quantity in comparison
        assert 'predicted' in comparison[quantity]
        assert 'experimental' in comparison[quantity]
        assert 'mae' in comparison[quantity]
        assert 'rmse' in comparison[quantity]
        assert 'error_percent' in comparison[quantity]
        assert comparison[quantity]['mae'] >= 0
        assert comparison[quantity]['rmse'] >= comparison[quantity]['mae'] * 0.99

    # Predictions CSV has P and v columns, so r2_score should be present
    assert 'r2_score' in comparison


def test_compare_with_predictions_missing_columns(config_path):
    """Prediction comparison raises for CSV files missing required columns."""
    char = LPBFCharacterisation(config_path)

    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        f.write('P,v,stress\n200,500,300\n')
        path = f.name

    with pytest.raises(ValueError, match='missing required columns'):
        char.compare_with_predictions({}, path)

    Path(path).unlink()


def test_compare_with_predictions_unsupported_format(config_path):
    """Prediction comparison raises for unsupported file formats."""
    char = LPBFCharacterisation(config_path)
    with pytest.raises(ValueError, match='Unsupported prediction file format'):
        char.compare_with_predictions({}, 'predictions.txt')


def test_validation_report_writes_files(tmp_path):
    """ValidationReport writes both report.md and summary.json."""
    results = {
        'porosity': {
            'porosity': 0.002,
            'num_pores': 10,
            'avg_pore_size': 5.0,
        },
        'stress': {
            'mean_von_mises': 330.0,
            'max_von_mises': 450.0,
        },
        'comparison': {
            'stress': {
                'predicted': 335.0,
                'experimental': 330.0,
                'mae': 5.0,
                'rmse': 5.0,
                'error_percent': 1.5,
            },
            'porosity': {
                'predicted': 0.0022,
                'experimental': 0.002,
                'mae': 0.0002,
                'rmse': 0.0002,
                'error_percent': 10.0,
            },
            'geometric_accuracy': {
                'predicted': 0.95,
                'experimental': 0.94,
                'mae': 0.01,
                'rmse': 0.01,
                'error_percent': 1.06,
            },
            'r2_score': 0.85,
        },
    }

    report = ValidationReport(
        results,
        tmp_path,
        run_id='test-run-001',
        lineage={'xct_hash': 'abc123', 'predictions_hash': 'def456'},
    )
    report_path, summary_path = report.write()

    assert report_path.exists()
    assert summary_path.exists()

    report_text = report_path.read_text()
    assert '# LPBF Validation Report' in report_text
    assert 'test-run-001' in report_text
    assert 'abc123' in report_text
    assert '## Porosity (XCT)' in report_text
    assert '## Residual Stress' in report_text
    assert '## Prediction Comparison' in report_text
    assert '| Validation Item | Status |' in report_text

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    assert summary['run_id'] == 'test-run-001'
    assert summary['lineage']['xct_hash'] == 'abc123'
    assert summary['checklist']['xct_porosity'] is True
    assert summary['checklist']['residual_stress'] is True
    assert summary['checklist']['prediction_comparison'] is True


def test_run_full_characterisation_creates_report(config_path):
    """End-to-end characterisation run produces results suitable for reporting."""
    char = LPBFCharacterisation(config_path)

    results = char.run_full_characterisation(
        xct_file=DATA_DIR / 'xct_porosity.csv',
        ebsd_file=DATA_DIR / 'ebsd.csv',
        stress_file=DATA_DIR / 'stress.csv',
        prediction_file=DATA_DIR / 'predictions.csv',
    )

    assert 'porosity' in results
    assert 'microstructure' in results
    assert 'stress' in results
    assert 'comparison' in results

    report = ValidationReport(results, char.run_dir, run_id=char.run_id)
    report_path, summary_path = report.write()
    assert report_path.exists()
    assert summary_path.exists()
