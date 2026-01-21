"""
Training script for IMF Bz forecasting.

Supports three training strategies:
1. Full fine-tuning
2. Frozen encoder
3. LoRA

Author: Vineet Vora
Date: 2025-11-27
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import wandb
from tqdm import tqdm

from data.bz_dataset import SimpleBzDataset, BzForecastDataset
from models.bz_models import create_bz_model


class BzTrainer:
    """Trainer for Bz forecasting models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = 'cuda',
    ):
        """
        Initialize trainer.

        Args:
            model: Bz forecasting model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss function
        self.criterion = nn.MSELoss()

        # Optimizer
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-5)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Checkpointing
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # WandB logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=config.get('wandb_project', 'surya-bz-forecasting'),
                name=config.get('experiment_name', 'experiment'),
                config=config
            )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')

        for batch_idx, batch in enumerate(pbar):
            # Handle different dataset outputs
            if len(batch) == 3:
                sdo, bz_targets, metadata = batch
            else:
                sdo, bz_targets = batch

            sdo = sdo.to(self.device)
            bz_targets = bz_targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sdo)

            # Compute loss
            loss = self.criterion(predictions, bz_targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

            # Log to WandB
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Per-horizon errors
        horizon_errors = [[] for _ in range(6)]

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')

            for batch in pbar:
                # Handle different dataset outputs
                if len(batch) == 3:
                    sdo, bz_targets, metadata = batch
                else:
                    sdo, bz_targets = batch

                sdo = sdo.to(self.device)
                bz_targets = bz_targets.to(self.device)

                # Forward pass
                predictions = self.model(sdo)

                # Compute loss
                loss = self.criterion(predictions, bz_targets)
                total_loss += loss.item()
                num_batches += 1

                # Per-horizon errors
                errors = torch.abs(predictions - bz_targets)  # MAE per sample
                for h in range(min(6, errors.shape[1])):
                    horizon_errors[h].extend(errors[:, h].cpu().numpy().tolist())

                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        # Calculate per-horizon metrics
        metrics = {'val_loss': avg_loss}

        for h in range(6):
            if horizon_errors[h]:
                mae = np.mean(horizon_errors[h])
                rmse = np.sqrt(np.mean(np.array(horizon_errors[h])**2))
                metrics[f'mae_hour_{h+1}'] = mae
                metrics[f'rmse_hour_{h+1}'] = rmse

        return metrics

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a custom data loader (e.g., test set).

        Args:
            data_loader: DataLoader to evaluate on

        Returns:
            Dictionary with 'loss', 'rmse', 'mae', and per-horizon metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions = []
        all_targets = []

        # Per-horizon errors
        num_horizons = 3  # T+24h, T+48h, T+72h
        horizon_errors = [[] for _ in range(num_horizons)]

        with torch.no_grad():
            for batch in data_loader:
                # Handle different dataset outputs
                if len(batch) == 3:
                    sdo, bz_targets, metadata = batch
                else:
                    sdo, bz_targets = batch

                sdo = sdo.to(self.device)
                bz_targets = bz_targets.to(self.device)

                # Forward pass
                predictions = self.model(sdo)

                # Compute loss
                loss = self.criterion(predictions, bz_targets)
                total_loss += loss.item()
                num_batches += 1

                # Store for overall metrics
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(bz_targets.cpu().numpy())

                # Per-horizon errors
                errors = torch.abs(predictions - bz_targets)  # MAE per sample
                for h in range(min(num_horizons, errors.shape[1])):
                    horizon_errors[h].extend(errors[:, h].cpu().numpy().tolist())

        # Calculate overall metrics
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Overall MAE and RMSE (across all horizons)
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        avg_loss = total_loss / num_batches

        metrics = {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
        }

        # Add per-horizon metrics
        for h in range(num_horizons):
            if horizon_errors[h]:
                horizon_mae = np.mean(horizon_errors[h])
                horizon_rmse = np.sqrt(np.mean(np.array(horizon_errors[h])**2))
                metrics[f'mae_hour_{h+1}'] = horizon_mae
                metrics[f'rmse_hour_{h+1}'] = horizon_rmse

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")

        # Save periodic checkpoint
        if epoch % 10 == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print("="*60)

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['val_loss']

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Print metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Per-hour MAE (validation):")
            for h in range(1, 7):
                if f'mae_hour_{h}' in val_metrics:
                    print(f"    Hour {h}: {val_metrics[f'mae_hour_{h}']:.4f} nT")

            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    **val_metrics,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  ✓ New best validation loss!")

            self.save_checkpoint(epoch, val_metrics, is_best)

            print("="*60)

        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        if self.use_wandb:
            wandb.finish()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train Bz forecasting model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['full', 'frozen', 'lora'],
        required=True,
        help='Training strategy'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        help='Device to train on'
    )
    parser.add_argument(
        '--simple-dataset',
        action='store_true',
        help='Use SimpleBzDataset for testing'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['strategy'] = args.strategy

    # Set device
    device = args.device
    print(f"Using device: {device}")

    # Create datasets
    forecast_horizons = config.get('forecast_horizons', [24, 48, 72])

    if args.simple_dataset:
        print("Using SimpleBzDataset (random SDO data, real Bz)")
        train_dataset = SimpleBzDataset(
            omni_bz_csv=config['omni_bz_csv'],
            num_samples=config.get('num_train_samples', 1000),
            forecast_horizons=forecast_horizons,
            sdo_shape=tuple(config.get('sdo_shape', [13, 512, 512])),
        )
        val_dataset = SimpleBzDataset(
            omni_bz_csv=config['omni_bz_csv'],
            num_samples=config.get('num_val_samples', 200),
            forecast_horizons=forecast_horizons,
            sdo_shape=tuple(config.get('sdo_shape', [13, 512, 512])),
            seed=123,  # Different seed for val
        )
    else:
        print("Using BzForecastDataset (real SDO + OMNI data)")
        train_dataset = BzForecastDataset(
            sdo_index_csv=config['sdo_train_csv'],
            omni_bz_csv=config['omni_bz_csv'],
            forecast_horizons=forecast_horizons,
            sdo_data_dir=config.get('sdo_data_dir'),
        )
        val_dataset = BzForecastDataset(
            sdo_index_csv=config['sdo_val_csv'],
            omni_bz_csv=config['omni_bz_csv'],
            forecast_horizons=forecast_horizons,
            sdo_data_dir=config.get('sdo_data_dir'),
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=device == 'cuda',
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=device == 'cuda',
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    surya_config = config.get('surya_config', {})
    num_horizons = len(forecast_horizons)  # Number of forecast horizons

    model = create_bz_model(
        strategy=args.strategy,
        surya_config=surya_config,
        num_horizons=num_horizons,
        surya_checkpoint=config.get('surya_checkpoint'),
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.1),
    )

    # Create trainer
    trainer = BzTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Train
    num_epochs = config.get('num_epochs', 50)
    trainer.train(num_epochs)


if __name__ == '__main__':
    main()
