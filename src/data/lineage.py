"""Lightweight data-versioning and lineage utilities.

This module implements a manifest-based approach (no DVC dependency) that
records the config and dataset hashes needed to reproduce an experiment.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def compute_file_hash(path, algorithm='sha256', chunk_size=65536):
    """Compute the hash of a file on disk.

    Args:
        path (str | Path): File path.
        algorithm (str): Hash algorithm supported by ``hashlib``.
        chunk_size (int): Bytes to read per chunk.

    Returns:
        str: Hexadecimal digest.
    """
    hasher = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_config_hash(config):
    """Compute a stable hash for a configuration dictionary.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        str: SHA-256 hex digest of the normalised JSON representation.
    """
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def get_git_commit(fallback='unknown'):
    """Return the current git commit hash, if available."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return fallback


def build_lineage(config, dataset_path=None):
    """Build a lineage dictionary for an artifact.

    Args:
        config (dict): Configuration dictionary.
        dataset_path (str | Path, optional): Path to the processed dataset.

    Returns:
        dict: Lineage record with config hash, dataset hash, timestamp, and git commit.
    """
    lineage = {
        'config_hash': compute_config_hash(config),
        'git_commit': get_git_commit(),
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    }
    if dataset_path is not None and Path(dataset_path).exists():
        lineage['dataset_hash'] = compute_file_hash(dataset_path)
        lineage['dataset_path'] = str(Path(dataset_path).resolve())
    return lineage


def write_manifest(config, dataset_path, output_dir):
    """Write a ``manifest.yaml`` next to the dataset or in an output directory.

    Args:
        config (dict): Configuration dictionary.
        dataset_path (str | Path): Path to the processed dataset.
        output_dir (str | Path): Directory where the manifest is written.

    Returns:
        Path: Path to the written manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_lineage(config, dataset_path)
    manifest_path = output_dir / 'manifest.yaml'
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f)
    return manifest_path


def verify_dataset_lineage(config, dataset_path, expected_lineage=None):
    """Verify that a dataset matches the lineage recorded for a config.

    Args:
        config (dict): Configuration dictionary.
        dataset_path (str | Path): Dataset file path.
        expected_lineage (dict, optional): Previously recorded lineage. If None,
            only the current config hash is compared against the dataset file.

    Returns:
        bool: True if the dataset hash matches the expected lineage.
    """
    if not Path(dataset_path).exists():
        return False
    current = build_lineage(config, dataset_path)
    if expected_lineage is None:
        return True
    return current.get('dataset_hash') == expected_lineage.get('dataset_hash')
