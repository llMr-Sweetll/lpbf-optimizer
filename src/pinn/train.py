import argparse
import os
import platform
import random
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loss_balancer import AdaptiveLossBalancer
from model import PINN, _DEFAULT_OUTPUT_BOUNDS
from physics import compute_physics_loss


class PINNTrainer:
    """
    Trainer for the Physics-Informed Neural Network for LPBF process optimization
    """
    def __init__(self, config_path, num_threads=4, config=None, seed=None):
        """
        Initialize the trainer with configuration.

        Args:
            config_path (str): Path to the configuration YAML file.
            num_threads (int): Number of threads for parallel operations.
            config (dict, optional): In-memory configuration dictionary. If provided,
                it overrides the YAML file.
            seed (int, optional): Random seed for reproducibility. Defaults to the
                value in ``training.random_seed``.
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        # Reproducibility
        if seed is None:
            seed = self.config['training'].get('random_seed', 42)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Handle OpenMP library conflict
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        self.num_threads = num_threads
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.num_threads)
        try:
            torch.set_num_threads(self.num_threads)
            torch.set_num_interop_threads(self.num_threads)
        except RuntimeError:
            # These calls are not allowed after PyTorch's thread pool has already
            # been used (e.g. when multiple trainers run in the same process).
            pass

        # Set device
        device_cfg = self.config['training'].get('device', 'auto')
        if device_cfg == 'cuda' or (device_cfg == 'auto' and torch.cuda.is_available()):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        # Load material properties
        self.mat_props = self.config['material_properties']

        # Determine whether physics-informed terms are active.
        train_cfg = self.config['training']
        self.physics_enabled = any(
            train_cfg.get(key, 0.0) > 0.0
            for key in ('lambda_heat', 'lambda_stress', 'lambda_porosity', 'lambda_geometry')
        )

        # Set up directories
        self.setup_directories()

        # Initialize model
        self.model = self._build_model()

        # Set up optimizer
        self.optimizer = self._build_optimizer()

        # Initialize Adaptive Loss Balancer only when physics is used.
        if self.physics_enabled:
            self.loss_balancer = AdaptiveLossBalancer(
                num_losses=4, alpha=train_cfg.get('balancer_alpha', 1.5)
            )
        else:
            self.loss_balancer = None

        # Set up learning rate scheduler
        self.scheduler = self._build_scheduler()

        # Track training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'loss_weights': {'data': [], 'heat': [], 'stress': [], 'mechanical': []}
        }

        # Test data loader, populated on demand by evaluate().
        self.test_loader = None

        # Current epoch
        self.current_epoch = 0

    def setup_directories(self):
        """Create necessary directories for saving results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.config['training']['output_dir']) / timestamp
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.log_dir = self.run_dir / 'logs'
        self.plot_dir = self.run_dir / 'plots'

        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

        # Create/update a stable symlink to the latest run
        latest_link = Path(self.config['training']['output_dir']) / 'latest'
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(self.run_dir.resolve(), target_is_directory=True)
        except OSError:
            # Symlinks may not be supported on all platforms (e.g. Windows without dev mode)
            pass

        # Save config to run directory
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)

    @staticmethod
    def _build_model_input(S_batch, coords_batch, t_batch, input_dim):
        """Build model input, trimming parameter dims if they exceed input_dim."""
        combined = torch.cat([S_batch, coords_batch, t_batch], dim=1)
        if combined.shape[1] > input_dim:
            s_keep = input_dim - coords_batch.shape[1] - t_batch.shape[1]
            if s_keep < 0:
                raise ValueError(
                    f"input_dim ({input_dim}) is smaller than coord ({coords_batch.shape[1]}) + time ({t_batch.shape[1]}) dims"
                )
            return torch.cat([S_batch[:, :s_keep], coords_batch, t_batch], dim=1)
        return combined

    def _build_model(self):
        """Initialize the PINN model."""
        model_config = self.config['model']
        model = PINN(
            in_dim=model_config['input_dim'],
            out_dim=model_config['output_dim'],
            width=model_config['hidden_width'],
            depth=model_config['hidden_depth'],
            dropout_rate=model_config.get('dropout_rate', 0.1),
            apply_output_bounds=model_config.get('apply_output_bounds', False),
            output_bounds_temperature=model_config.get('output_bounds_temperature', 100.0),
        )
        model = model.to(self.device)
        return model

    def _build_optimizer(self):
        """Set up the optimizer"""
        optim_config = self.config['training']['optimizer']
        if optim_config['type'].lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optim_config['learning_rate'],
                weight_decay=optim_config.get('weight_decay', 0)
            )
        else:
            raise NotImplementedError(f"Optimizer {optim_config['type']} not implemented.")
        return optimizer

    def _build_scheduler(self):
        """Set up the learning rate scheduler"""
        sched_config = self.config['training'].get('scheduler', {})
        if not sched_config or sched_config.get('type', '').lower() == 'none':
            return None

        if sched_config['type'].lower() in ('reduce_lr_on_plateau', 'reducelronplateau'):
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.1),
                patience=sched_config.get('patience', 10)
            )
        elif sched_config['type'].lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', 100),
                eta_min=sched_config.get('eta_min', 0)
            )
        else:
            raise NotImplementedError(f"Scheduler {sched_config['type']} not implemented.")
        return scheduler

    def load_data(self):
        """
        Load training and validation data from processed HDF5 files
        """
        data_config = self.config['data']
        data_file = Path(data_config['processed_data_path'])

        # Load data from HDF5 file
        with h5py.File(data_file, 'r') as f:
            # Process parameters (scan vectors)
            S_train = torch.tensor(f['train/scan_vectors'][:], dtype=torch.float32)
            S_val = torch.tensor(f['val/scan_vectors'][:], dtype=torch.float32)

            # Coordinate and time information
            coords_train = torch.tensor(f['train/coordinates'][:], dtype=torch.float32)
            coords_val = torch.tensor(f['val/coordinates'][:], dtype=torch.float32)
            time_train = torch.tensor(f['train/time'][:], dtype=torch.float32)
            time_val = torch.tensor(f['val/time'][:], dtype=torch.float32)

            # Ground truth values (from FEA)
            y_train = torch.tensor(f['train/outputs'][:], dtype=torch.float32)
            y_val = torch.tensor(f['val/outputs'][:], dtype=torch.float32)

        # Move data to device
        num_workers = self.config['training'].get('num_workers', None)
        if num_workers is None:
            num_workers = 0 if platform.system() == 'Darwin' else self.num_threads
        pin_memory = self.device.type == 'cuda'
        persistent = num_workers > 0

        train_dataset = torch.utils.data.TensorDataset(S_train, coords_train, time_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent
        )

        val_dataset = torch.utils.data.TensorDataset(S_val, coords_val, time_val, y_val)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent
        )

        print(f"Loaded {len(S_train)} training samples and {len(S_val)} validation samples")

    def load_test_data(self):
        """Load test data from the processed HDF5 file."""
        if self.test_loader is not None:
            return

        data_config = self.config['data']
        data_file = Path(data_config['processed_data_path'])

        with h5py.File(data_file, 'r') as f:
            S_test = torch.tensor(f['test/scan_vectors'][:], dtype=torch.float32)
            coords_test = torch.tensor(f['test/coordinates'][:], dtype=torch.float32)
            time_test = torch.tensor(f['test/time'][:], dtype=torch.float32)
            y_test = torch.tensor(f['test/outputs'][:], dtype=torch.float32)

        num_workers = self.config['training'].get('num_workers', None)
        if num_workers is None:
            num_workers = 0 if platform.system() == 'Darwin' else self.num_threads
        pin_memory = self.device.type == 'cuda'
        persistent = num_workers > 0

        test_dataset = torch.utils.data.TensorDataset(S_test, coords_test, time_test, y_test)
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent
        )
        print(f"Loaded {len(S_test)} test samples")

    def evaluate(self):
        """
        Evaluate the trained model on the held-out test set.

        Returns:
            dict: Metrics including per-output MSE/RMSE/R2, total physics
            residual, and percentage of predictions outside physical bounds.
        """
        self.load_test_data()
        self.model.eval()

        input_dim = self.config['model']['input_dim']
        mse_loss = nn.MSELoss(reduction='sum')

        total_mse_sum = 0.0
        per_output_mse_sum = torch.zeros(self.config['model']['output_dim'], device=self.device)
        per_output_var_sum = torch.zeros(self.config['model']['output_dim'], device=self.device)
        total_samples = 0

        physics_components = {'heat': 0.0, 'stress': 0.0, 'porosity': 0.0, 'geometry': 0.0}

        all_preds = []
        all_targets = []

        for S_batch, coords_batch, t_batch, y_batch in self.test_loader:
            S_batch = S_batch.to(self.device)
            coords_batch = coords_batch.to(self.device)
            t_batch = t_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            with torch.no_grad():
                model_input = self._build_model_input(S_batch, coords_batch, t_batch, input_dim)
                y_pred = self.model(model_input)

            all_preds.append(y_pred.detach().cpu())
            all_targets.append(y_batch.detach().cpu())

            batch_size = y_pred.shape[0]
            total_samples += batch_size
            total_mse_sum += mse_loss(y_pred, y_batch).item()
            per_output_mse_sum += torch.sum((y_pred - y_batch) ** 2, dim=0)

            # Physics residuals require gradients w.r.t. coordinates.
            heat_loss, stress_loss, porosity_loss, geometry_loss = compute_physics_loss(
                self.model,
                S_batch,
                coords_batch,
                t_batch,
                self.mat_props,
                return_components=True,
            )
            physics_components['heat'] += heat_loss.item() * batch_size
            physics_components['stress'] += stress_loss.item() * batch_size
            physics_components['porosity'] += porosity_loss.item() * batch_size
            physics_components['geometry'] += geometry_loss.item() * batch_size

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Per-output MSE / RMSE
        per_output_mse = per_output_mse_sum / total_samples
        per_output_rmse = torch.sqrt(per_output_mse)

        # R^2 per output
        target_mean = all_targets.mean(dim=0)
        ss_tot = torch.sum((all_targets - target_mean) ** 2, dim=0)
        ss_res = torch.sum((all_preds - all_targets) ** 2, dim=0)
        per_output_r2 = 1.0 - ss_res / (ss_tot + 1e-12)

        # Bound violations against the default physical ranges
        bounds = _DEFAULT_OUTPUT_BOUNDS
        violations = torch.zeros(self.config['model']['output_dim'])
        for i, (lo, hi) in enumerate(bounds):
            violations[i] = torch.sum((all_preds[:, i] < lo - 1e-6) | (all_preds[:, i] > hi + 1e-6)).item()
        violation_pct = 100.0 * violations.sum().item() / (total_samples * self.config['model']['output_dim'])

        # Aggregate physics residual
        for key in physics_components:
            physics_components[key] /= total_samples
        total_physics_residual = sum(physics_components.values())

        metrics = {
            'test_mse_total': total_mse_sum / total_samples,
            'test_mse_per_output': per_output_mse.cpu().numpy(),
            'test_rmse_per_output': per_output_rmse.cpu().numpy(),
            'test_r2_per_output': per_output_r2.cpu().numpy(),
            'test_physics_residual': total_physics_residual,
            'test_physics_components': physics_components,
            'test_bound_violations_pct': violation_pct,
            'test_samples': total_samples,
        }
        return metrics

    def train_step(self, S_batch, coords_batch, t_batch, y_batch):
        self.optimizer.zero_grad()

        input_dim = self.config['model']['input_dim']
        model_input = self._build_model_input(S_batch, coords_batch, t_batch, input_dim)
        y_pred = self.model(model_input)

        mse_loss = nn.MSELoss()
        data_loss = mse_loss(y_pred, y_batch)

        train_cfg = self.config['training']
        lambda_heat = train_cfg.get('lambda_heat', 0.1)
        lambda_stress = train_cfg.get('lambda_stress', 0.1)
        lambda_porosity = train_cfg.get('lambda_porosity', 0.05)
        lambda_geometry = train_cfg.get('lambda_geometry', 0.05)

        if self.physics_enabled:
            heat_loss, stress_loss, porosity_loss, geometry_loss = compute_physics_loss(
                self.model,
                S_batch,
                coords_batch,
                t_batch,
                self.mat_props,
                return_components=True,
            )

            # Shared parameter for GradNorm (last hidden layer weight matrix)
            shared_params = list(self.model.hidden[-1].parameters())[0] if hasattr(self.model, 'hidden') else list(self.model.parameters())[-2]

            # Pass unscaled losses to the balancer so adaptive weights control the
            # relative contribution of each term without double-scaling by lambdas.
            raw_losses = [
                data_loss,
                heat_loss,
                stress_loss,
                porosity_loss + geometry_loss,
            ]

            try:
                weights = self.loss_balancer.update_weights(raw_losses, shared_params)
                total_loss = (
                    weights[0] * data_loss
                    + weights[1] * lambda_heat * heat_loss
                    + weights[2] * lambda_stress * stress_loss
                    + weights[3] * (lambda_porosity * porosity_loss + lambda_geometry * geometry_loss)
                )
                current_weights = weights.detach().cpu().numpy()
                self.metrics['loss_weights']['data'].append(float(current_weights[0]))
                self.metrics['loss_weights']['heat'].append(float(current_weights[1]))
                self.metrics['loss_weights']['stress'].append(float(current_weights[2]))
                self.metrics['loss_weights']['mechanical'].append(float(current_weights[3]))
            except RuntimeError:
                # Fallback to static weighting if GradNorm graph fails
                total_loss = (
                    data_loss
                    + lambda_heat * heat_loss
                    + lambda_stress * stress_loss
                    + lambda_porosity * porosity_loss
                    + lambda_geometry * geometry_loss
                )
                self.metrics['loss_weights']['data'].append(1.0)
                self.metrics['loss_weights']['heat'].append(lambda_heat)
                self.metrics['loss_weights']['stress'].append(lambda_stress)
                self.metrics['loss_weights']['mechanical'].append((lambda_porosity + lambda_geometry) / 2.0)

            physics_term = (
                lambda_heat * heat_loss
                + lambda_stress * stress_loss
                + lambda_porosity * porosity_loss
                + lambda_geometry * geometry_loss
            ).item()
            heat_val = heat_loss.item()
            stress_val = stress_loss.item()
            porosity_val = porosity_loss.item()
            geometry_val = geometry_loss.item()
        else:
            total_loss = data_loss
            self.metrics['loss_weights']['data'].append(1.0)
            self.metrics['loss_weights']['heat'].append(0.0)
            self.metrics['loss_weights']['stress'].append(0.0)
            self.metrics['loss_weights']['mechanical'].append(0.0)
            physics_term = 0.0
            heat_val = stress_val = porosity_val = geometry_val = 0.0

        total_loss.backward()

        if train_cfg.get('clip_grad', False):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                train_cfg.get('clip_value', 1.0)
            )

        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'data': data_loss.item(),
            'physics': physics_term,
            'heat': heat_val,
            'stress': stress_val,
            'porosity': porosity_val,
            'geometry': geometry_val,
        }

    def train_epoch(self, epoch):
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            float: Average loss for the epoch
        """
        self.model.train()

        # Use the data loader that was created in load_data
        n_batches = len(self.train_loader)

        # Initialize losses
        epoch_loss = 0
        epoch_data_loss = 0
        epoch_physics_loss = 0

        # Train over batches
        for i, (S_batch, coords_batch, t_batch, y_batch) in enumerate(self.train_loader):
            # Move to device
            S_batch = S_batch.to(self.device)
            coords_batch = coords_batch.to(self.device)
            t_batch = t_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Training step
            losses = self.train_step(S_batch, coords_batch, t_batch, y_batch)

            # Update epoch losses
            epoch_loss += losses['total']
            epoch_data_loss += losses['data']
            epoch_physics_loss += losses['physics']

            # Print progress
            if (i + 1) % self.config['training'].get('print_freq', 10) == 0:
                print(f"Epoch [{epoch+1}/{self.config['training']['n_epochs']}] "
                      f"Batch [{i+1}/{n_batches}] "
                      f"Loss: {losses['total']:.6f} "
                      f"(Data: {losses['data']:.6f}, Physics: {losses['physics']:.6f})")

        # Calculate average losses
        avg_loss = epoch_loss / n_batches
        avg_data_loss = epoch_data_loss / n_batches
        avg_physics_loss = epoch_physics_loss / n_batches

        # Update metrics
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['data_loss'].append(avg_data_loss)
        self.metrics['physics_loss'].append(avg_physics_loss)

        return avg_loss

    def validate(self):
        """
        Validate the model on validation data

        Returns:
            float: Validation loss
        """
        self.model.eval()

        # Use the data loader
        n_batches = len(self.val_loader)

        # Initialize loss
        val_loss = 0

        with torch.no_grad():
            # Validate over batches
            for S_batch, coords_batch, t_batch, y_batch in self.val_loader:
                # Move to device
                S_batch = S_batch.to(self.device)
                coords_batch = coords_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                input_dim = self.config['model']['input_dim']
                model_input = self._build_model_input(S_batch, coords_batch, t_batch, input_dim)
                y_pred = self.model(model_input)

                # Compute loss
                mse_loss = nn.MSELoss()
                batch_loss = mse_loss(y_pred, y_batch)

                # Update validation loss
                val_loss += batch_loss.item()

        # Calculate average validation loss
        avg_val_loss = val_loss / n_batches

        # Update metrics
        self.metrics['val_loss'].append(avg_val_loss)

        return avg_val_loss

    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            loss: Current loss
            is_best: Whether this is the best model so far
        """
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch+1}.pt"
        best_model_path = self.checkpoint_dir / "best_model.pt"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'metrics': self.metrics
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Save best model if applicable
        if is_best:
            torch.save(checkpoint, best_model_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.metrics = checkpoint['metrics']

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")

    def plot_metrics(self):
        """Plot training and validation metrics with publication-quality formatting"""
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            pass

        # Plot loss curves
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(self.metrics['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.metrics['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        plt.title('Training and Validation Loss Evolution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'loss_curves.png', dpi=300)
        plt.close()

        # Plot data vs physics loss
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(self.metrics['data_loss'], label='Data Loss (MSE)', linewidth=2)
        plt.plot(self.metrics['physics_loss'], label='Physics Residuals', linewidth=2)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Loss Component Magnitude', fontsize=12, fontweight='bold')
        plt.title('Data vs. Physics Loss Components', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log') # Log scale to see differences better
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'loss_components.png', dpi=300)
        plt.close()

        # Plot Adaptive Weights Evolution
        # Weights are logged per step, so we might want to downsample or plot moving average
        # For now, plot all steps to show detailed dynamics
        if self.metrics['loss_weights']['data']:
            plt.figure(figsize=(10, 6), dpi=300)
            steps = range(len(self.metrics['loss_weights']['data']))
            plt.plot(steps, self.metrics['loss_weights']['data'], label=r'$\lambda_{data}$', alpha=0.8)
            plt.plot(steps, self.metrics['loss_weights']['heat'], label=r'$\lambda_{heat}$', alpha=0.8)
            plt.plot(steps, self.metrics['loss_weights']['stress'], label=r'$\lambda_{stress}$', alpha=0.8)
            plt.plot(steps, self.metrics['loss_weights']['mechanical'], label=r'$\lambda_{mechanical}$', alpha=0.8)
            plt.xlabel('Training Step', fontsize=12, fontweight='bold')
            plt.ylabel('Adaptive Weight Value', fontsize=12, fontweight='bold')
            plt.title('Evolution of Adaptive Loss Weights (GradNorm)', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plot_dir / 'adaptive_weights.png', dpi=300)
            plt.close()

    def train(self):
        """Main training loop"""
        print("Beginning training...")

        # Load data
        self.load_data()

        # Training parameters
        n_epochs = self.config['training']['n_epochs']
        best_val_loss = float('inf')

        # Train for n_epochs
        for epoch in range(self.current_epoch, n_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print epoch results
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Validation Loss: {val_loss:.6f}")

            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # Save checkpoint periodically and for best model
            if (epoch + 1) % self.config['training']['checkpoint_freq'] == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

            # Plot metrics
            if (epoch + 1) % self.config['training']['plot_freq'] == 0:
                self.plot_metrics()

            # Update current epoch
            self.current_epoch = epoch + 1

        # Final plots and checkpoints
        self.plot_metrics()
        self.save_checkpoint(n_epochs - 1, val_loss)

        print("Training complete!")


def main():
    """Main function to start training"""
    parser = argparse.ArgumentParser(description='Train PINN for LPBF optimization')
    parser.add_argument('--config', type=str, default='data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file for resuming training')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads to use for parallel operations')
    args = parser.parse_args()

    # Initialize trainer
    trainer = PINNTrainer(args.config, num_threads=args.num_threads)

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
