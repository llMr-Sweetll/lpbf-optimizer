import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import yaml
import h5py
from pathlib import Path

from model import PINN
from model import PINN
from physics import compute_physics_loss
from loss_balancer import AdaptiveLossBalancer


class PINNTrainer:
    """
    Trainer for the Physics-Informed Neural Network for LPBF process optimization
    """
    def __init__(self, config_path, num_threads=4):
        """
        Initialize the trainer with configuration
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Handle OpenMP library conflict
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        self.num_threads = num_threads
        os.environ['OMP_NUM_THREADS'] = str(self.num_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.num_threads)
        torch.set_num_threads(self.num_threads)
        torch.set_num_interop_threads(self.num_threads)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load material properties
        self.mat_props = self.config['material_properties']
        
        # Set up directories
        self.setup_directories()
        
        # Initialize model
        self.model = self._build_model()
        
        # Set up optimizer
        self.optimizer = self._build_optimizer()
        
        # Initialize Adaptive Loss Balancer (Weights: Data, Heat, Stress)
        self.loss_balancer = AdaptiveLossBalancer(num_losses=3, alpha=self.config['training'].get('balancer_alpha', 1.5))
        
        # Set up learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Track training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'loss_weights': {'data': [], 'heat': [], 'stress': []}
        }
        
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
        
        # Save config to run directory
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def _build_model(self):
        """Initialize the PINN model"""
        model_config = self.config['model']
        model = PINN(
            in_dim=model_config['input_dim'],
            out_dim=model_config['output_dim'],
            width=model_config['hidden_width'],
            depth=model_config['hidden_depth'],
            dropout_rate=model_config.get('dropout_rate', 0.1)
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
        
        if sched_config['type'].lower() == 'reducelronplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.1),
                patience=sched_config.get('patience', 10),
                verbose=True
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
        train_dataset = torch.utils.data.TensorDataset(S_train, coords_train, time_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.num_threads,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_dataset = torch.utils.data.TensorDataset(S_val, coords_val, time_val, y_val)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.num_threads,
            pin_memory=True,
            persistent_workers=True
        )
        
        print(f"Loaded {len(S_train)} training samples and {len(S_val)} validation samples")
    
    def train_step(self, S_batch, coords_batch, t_batch, y_batch):
        """
        Perform one training step
        
        Args:
            S_batch: Process parameters batch
            coords_batch: Spatial coordinates batch
            t_batch: Time coordinates batch
            y_batch: Ground truth outputs batch
        
        Returns:
            dict: Loss components
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        # Ensure input dimensions match model's expected dimensions
        print(f"DEBUG: S_batch shape: {S_batch.shape}")
        print(f"DEBUG: coords_batch shape: {coords_batch.shape}")
        print(f"DEBUG: t_batch shape: {t_batch.shape}")
        combined_input = torch.cat([S_batch, coords_batch, t_batch], dim=1)
        print(f"DEBUG: combined_input shape: {combined_input.shape}")
        input_dim = self.config['model']['input_dim']
        
        # If dimensions exceed expected input_dim, trim the excess dimensions from S_batch
        if combined_input.shape[1] > input_dim:
            # Calculate how many dimensions to keep from S_batch
            # We need to keep coords_batch (3) and t_batch (1) intact
            s_dims_to_keep = input_dim - coords_batch.shape[1] - t_batch.shape[1]
            # Create a properly sized input tensor
            model_input = torch.cat([S_batch[:, :s_dims_to_keep], coords_batch, t_batch], dim=1)
        else:
            model_input = combined_input
            
        y_pred = self.model(model_input)
        
        # Compute losses (Data, Heat, Stress)
        # Note: We need component-wise physics losses for the balancer
        
        # Data loss
        mse_loss = nn.MSELoss()
        data_loss = mse_loss(y_pred, y_batch)
        
        # Physics loss components
        heat_loss, stress_loss = compute_physics_loss(
            self.model, 
            S_batch, 
            coords_batch, 
            t_batch, 
            self.mat_props,
            return_components=True
        )
        
        # Update loss weights using Adaptive Loss Balancer
        # We need the last shared layer parameters for gradient computation
        # Typically the last layer of the shared encoder before task-specific heads
        # In our PINN, self.model.hidden[-1] is the last linear layer of the shared trunk
        shared_params = list(self.model.hidden[-1].parameters())[0] if hasattr(self.model, 'hidden') else list(self.model.parameters())[-2]
        
        try:
             # Stack losses for the balancer
            raw_losses = [data_loss, heat_loss, stress_loss]
            weights = self.loss_balancer.update_weights(raw_losses, shared_params)
            
            # Weighted total loss
            # weights[0] -> Data, weights[1] -> Heat, weights[2] -> Stress
            total_loss = (weights[0] * data_loss + 
                          weights[1] * heat_loss + 
                          weights[2] * stress_loss)
            
            # Log current weights for plotting (detach to avoid graph retention)
            # Store single value for the batch (approximated)
            current_weights = weights.detach().cpu().numpy()
            self.metrics['loss_weights']['data'].append(current_weights[0])
            self.metrics['loss_weights']['heat'].append(current_weights[1])
            self.metrics['loss_weights']['stress'].append(current_weights[2])
                          
        except Exception as e:
            # Fallback if balancer fails (e.g. graph issues)
            # print(f"Warning: Loss balancer failed ({e}), using static weights") # Reduce spam
            total_loss = data_loss + \
                         self.config['training']['lambda_heat'] * heat_loss + \
                         self.config['training']['lambda_stress'] * stress_loss
                         
            # Log static weights
            self.metrics['loss_weights']['data'].append(1.0)
            self.metrics['loss_weights']['heat'].append(self.config['training']['lambda_heat'])
            self.metrics['loss_weights']['stress'].append(self.config['training']['lambda_stress'])
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        if self.config['training'].get('clip_grad', False):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training'].get('clip_value', 1.0)
            )
        
        # Optimizer step
        self.optimizer.step()
        
        return {
            'total': total_loss.item(),
            'data': data_loss.item(),
            'physics': (heat_loss + stress_loss).item(),
            'heat': heat_loss.item(),
            'stress': stress_loss.item()
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
                # Ensure input dimensions match model's expected dimensions
                combined_input = torch.cat([S_batch, coords_batch, t_batch], dim=1)
                input_dim = self.config['model']['input_dim']
                
                # If dimensions exceed expected input_dim, trim the excess dimensions from S_batch
                if combined_input.shape[1] > input_dim:
                    # Calculate how many dimensions to keep from S_batch
                    # We need to keep coords_batch (3) and t_batch (1) intact
                    s_dims_to_keep = input_dim - coords_batch.shape[1] - t_batch.shape[1]
                    # Create a properly sized input tensor
                    model_input = torch.cat([S_batch[:, :s_dims_to_keep], coords_batch, t_batch], dim=1)
                else:
                    model_input = combined_input
                    
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
        plt.style.use('seaborn-v0_8-whitegrid') # Use a clean, scientific style if available, else defaults
        
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
            plt.plot(steps, self.metrics['loss_weights']['data'], label='$\lambda_{data}$', alpha=0.8)
            plt.plot(steps, self.metrics['loss_weights']['heat'], label='$\lambda_{heat}$', alpha=0.8)
            plt.plot(steps, self.metrics['loss_weights']['stress'], label='$\lambda_{stress}$', alpha=0.8)
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