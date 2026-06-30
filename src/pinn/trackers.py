"""Optional experiment trackers for the PINN training loop.

Trackers are intentionally lightweight wrappers so that switching between
TensorBoard, Weights & Biases, or no tracking only requires a config change.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTracker(ABC):
    """Abstract base class for experiment trackers."""

    @abstractmethod
    def log_scalar(self, tag, value, step):
        """Log a scalar value at a given step."""
        raise NotImplementedError

    @abstractmethod
    def log_image(self, tag, image_path, step):
        """Log an image from a file path at a given step."""
        raise NotImplementedError

    def close(self):
        """Close the tracker and flush any pending writes."""
        pass


class NoopTracker(BaseTracker):
    """Tracker that does nothing; used when tracking is disabled."""

    def log_scalar(self, tag, value, step):
        pass

    def log_image(self, tag, image_path, step):
        pass


class TensorBoardTracker(BaseTracker):
    """TensorBoard tracker using ``torch.utils.tensorboard.SummaryWriter``."""

    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, global_step=step)

    def log_image(self, tag, image_path, step):
        self.writer.add_image(
            tag,
            self._load_image(image_path),
            global_step=step,
            dataformats="HWC",
        )

    @staticmethod
    def _load_image(image_path):
        import numpy as np
        from PIL import Image

        img = Image.open(image_path)
        return np.array(img)

    def close(self):
        self.writer.close()


class WandbTracker(BaseTracker):
    """Weights & Biases tracker.

    ``wandb`` is imported lazily so that it remains an optional dependency.
    """

    def __init__(self, project, name=None, config=None, log_dir=None):
        import wandb

        self.wandb = wandb
        self.wandb.init(
            project=project,
            name=name,
            config=config,
            dir=log_dir,
        )

    def log_scalar(self, tag, value, step):
        self.wandb.log({tag: value}, step=step)

    def log_image(self, tag, image_path, step):
        self.wandb.log({tag: self.wandb.Image(str(image_path))}, step=step)

    def close(self):
        self.wandb.finish()


def build_tracker(config, run_dir):
    """Build a tracker from the training config.

    Args:
        config (dict): Training configuration dictionary.
        run_dir (Path): Directory for the current training run.

    Returns:
        BaseTracker: Configured tracker instance.
    """
    tracker_cfg = config.get('training', {}).get('tracker', 'none')
    if isinstance(tracker_cfg, str):
        tracker_type = tracker_cfg.lower()
        tracker_kwargs = {}
    elif isinstance(tracker_cfg, dict):
        tracker_type = tracker_cfg.get('type', 'none').lower()
        tracker_kwargs = {k: v for k, v in tracker_cfg.items() if k != 'type'}
    else:
        tracker_type = 'none'
        tracker_kwargs = {}

    if tracker_type in ('none', ''):
        return NoopTracker()

    if tracker_type == 'tensorboard':
        log_dir = tracker_kwargs.get('log_dir', run_dir / 'tensorboard')
        return TensorBoardTracker(log_dir=log_dir)

    if tracker_type == 'wandb':
        return WandbTracker(
            project=tracker_kwargs.get('project', 'lpbf-optimizer'),
            name=tracker_kwargs.get('name'),
            config=config,
            log_dir=tracker_kwargs.get('log_dir', str(run_dir)),
        )

    raise ValueError(f"Unknown tracker type: {tracker_type}")
