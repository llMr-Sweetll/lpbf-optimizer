"""Tests for experiment trackers."""

from pathlib import Path

import pytest
import yaml
from trackers import NoopTracker, TensorBoardTracker, build_tracker


def test_noop_tracker():
    tracker = NoopTracker()
    tracker.log_scalar('loss', 1.0, 0)
    tracker.log_image('plot', Path('nonexistent.png'), 0)
    tracker.close()


def test_build_noop_tracker():
    config = {'training': {'tracker': 'none'}}
    tracker = build_tracker(config, Path('/tmp/run'))
    assert isinstance(tracker, NoopTracker)


def test_build_tensorboard_tracker(tmp_path):
    config = {'training': {'tracker': {'type': 'tensorboard'}}}
    tracker = build_tracker(config, tmp_path)
    assert isinstance(tracker, TensorBoardTracker)
    tracker.log_scalar('loss', 1.0, 0)
    tracker.close()
    assert (tmp_path / 'tensorboard').exists()


def test_build_tracker_unknown():
    config = {'training': {'tracker': 'unknown'}}
    with pytest.raises(ValueError):
        build_tracker(config, Path('/tmp/run'))


def test_config_yaml_has_tracker_key():
    with open('data/params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    assert 'tracker' in params['training']
