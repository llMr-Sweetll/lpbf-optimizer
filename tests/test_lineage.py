"""Tests for data lineage and manifest utilities."""


import yaml

from data.lineage import (
    build_lineage,
    compute_config_hash,
    compute_file_hash,
    verify_dataset_lineage,
    write_manifest,
)


def test_compute_file_hash(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("hello")
    h1 = compute_file_hash(path)
    h2 = compute_file_hash(path)
    assert h1 == h2
    assert len(h1) == 64


def test_compute_config_hash():
    cfg1 = {"a": 1, "b": [2, 3]}
    cfg2 = {"b": [2, 3], "a": 1}
    assert compute_config_hash(cfg1) == compute_config_hash(cfg2)


def test_build_lineage(tmp_path):
    dataset = tmp_path / "data.h5"
    dataset.write_bytes(b"dummy")
    config = {"model": {"width": 16}}
    lineage = build_lineage(config, dataset)
    assert "config_hash" in lineage
    assert "dataset_hash" in lineage
    assert lineage["dataset_hash"] == compute_file_hash(dataset)


def test_write_manifest(tmp_path):
    dataset = tmp_path / "data.h5"
    dataset.write_bytes(b"dummy")
    config = {"model": {"width": 16}}
    manifest_path = write_manifest(config, dataset, tmp_path / "out")
    assert manifest_path.exists()
    with open(manifest_path, "r") as f:
        loaded = yaml.safe_load(f)
    assert loaded["config_hash"] == compute_config_hash(config)


def test_verify_dataset_lineage(tmp_path):
    dataset = tmp_path / "data.h5"
    dataset.write_bytes(b"dummy")
    config = {"model": {"width": 16}}
    lineage = build_lineage(config, dataset)
    assert verify_dataset_lineage(config, dataset, lineage)

    dataset.write_bytes(b"changed")
    assert not verify_dataset_lineage(config, dataset, lineage)
