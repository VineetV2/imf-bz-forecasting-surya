"""
Evaluation and inference script for IMF Bz forecasting.

Evaluates trained models and generates predictions.
Focus on 6th hour ahead as per professor's requirements.

Author: Vineet Vora
Date: 2025-11-27
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from data.bz_dataset import SimpleBzDataset, BzForecastDataset
from models.bz_models import create_bz_model


class BzEvaluator:
    """Evaluator for Bz forecasting models."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = 'cuda',
        output_dir: str = './results',
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained Bz forecasting model
            test_loader: Test data loader
            device: Device to evaluate on
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_targets = []
        all_metadata = []

        print("Running inference on test set...")

        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                # Handle different dataset outputs
                if len(batch) == 3:
                    sdo, bz_targets, metadata = batch
                    all_metadata.extend(metadata)
                else:
                    sdo, bz_targets = batch

                sdo = sdo.to(self.device)
                bz_targets = bz_targets.to(self.device)

                # Forward pass
                predictions = self.model(sdo)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(bz_targets.cpu().numpy())

        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)  # Shape: (N, 6)
        targets = np.concatenate(all_targets, axis=0)  # Shape: (N, 6)

        print(f"\nTotal test samples: {len(predictions)}")

        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)

        # Save results
        self._save_results(predictions, targets, metrics, all_metadata)

        # Generate plots
        self._generate_plots(predictions, targets, metrics)

        return metrics

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        metrics = {}

        # Overall metrics
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mse = np.mean((predictions - targets) ** 2)
        correlation = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]

        metrics['overall'] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mse': float(mse),
            'correlation': float(correlation),
        }

        # Per-horizon metrics (1-6 hours)
        metrics['per_horizon'] = {}

        for h in range(6):
            h_pred = predictions[:, h]
            h_target = targets[:, h]

            h_mae = np.mean(np.abs(h_pred - h_target))
            h_rmse = np.sqrt(np.mean((h_pred - h_target) ** 2))
            h_mse = np.mean((h_pred - h_target) ** 2)
            h_corr = np.corrcoef(h_pred, h_target)[0, 1]

            metrics['per_horizon'][f'hour_{h+1}'] = {
                'mae': float(h_mae),
                'rmse': float(h_rmse),
                'mse': float(h_mse),
                'correlation': float(h_corr),
            }

        # Focus on 6th hour (as requested by professor)
        hour_6_pred = predictions[:, 5]
        hour_6_target = targets[:, 5]

        metrics['hour_6_focus'] = {
            'mae': float(np.mean(np.abs(hour_6_pred - hour_6_target))),
            'rmse': float(np.sqrt(np.mean((hour_6_pred - hour_6_target) ** 2))),
            'mse': float(np.mean((hour_6_pred - hour_6_target) ** 2)),
            'correlation': float(np.corrcoef(hour_6_pred, hour_6_target)[0, 1]),
        }

        # Metrics for negative Bz (southward - most important)
        negative_mask = targets < 0
        if negative_mask.any():
            neg_mae = np.mean(np.abs(predictions[negative_mask] - targets[negative_mask]))
            neg_rmse = np.sqrt(np.mean((predictions[negative_mask] - targets[negative_mask]) ** 2))

            metrics['negative_bz'] = {
                'count': int(negative_mask.sum()),
                'percentage': float(100 * negative_mask.sum() / negative_mask.size),
                'mae': float(neg_mae),
                'rmse': float(neg_rmse),
            }

        # Metrics for strong negative Bz (< -5 nT)
        strong_neg_mask = targets < -5
        if strong_neg_mask.any():
            strong_neg_mae = np.mean(np.abs(predictions[strong_neg_mask] - targets[strong_neg_mask]))
            strong_neg_rmse = np.sqrt(np.mean((predictions[strong_neg_mask] - targets[strong_neg_mask]) ** 2))

            metrics['strong_negative_bz'] = {
                'count': int(strong_neg_mask.sum()),
                'percentage': float(100 * strong_neg_mask.sum() / strong_neg_mask.size),
                'mae': float(strong_neg_mae),
                'rmse': float(strong_neg_rmse),
            }

        return metrics

    def _save_results(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metrics: Dict[str, Any],
        metadata: List[Dict] = None
    ):
        """Save results to files."""
        # Save predictions and targets as CSV
        results_df = pd.DataFrame()

        for h in range(6):
            results_df[f'predicted_hour_{h+1}'] = predictions[:, h]
            results_df[f'target_hour_{h+1}'] = targets[:, h]
            results_df[f'error_hour_{h+1}'] = predictions[:, h] - targets[:, h]

        if metadata:
            # Add metadata if available
            if 'sdo_time' in metadata[0]:
                results_df['sdo_time'] = [m['sdo_time'] for m in metadata]

        csv_path = os.path.join(self.output_dir, 'predictions.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved predictions to {csv_path}")

        # Save metrics as YAML
        metrics_path = os.path.join(self.output_dir, 'metrics.yaml')
        with open(metrics_path, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        print(f"Saved metrics to {metrics_path}")

        # Print metrics
        self._print_metrics(metrics)

    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in a formatted way."""
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)

        print("\nOverall Performance:")
        for key, value in metrics['overall'].items():
            print(f"  {key.upper():15s}: {value:.4f}")

        print("\n" + "-"*60)
        print("Per-Horizon Performance (1-6 hours ahead):")
        print("-"*60)
        print(f"{'Hour':^8s} {'MAE':^10s} {'RMSE':^10s} {'Correlation':^12s}")
        print("-"*60)

        for h in range(1, 7):
            h_metrics = metrics['per_horizon'][f'hour_{h}']
            print(f"{h:^8d} {h_metrics['mae']:^10.4f} {h_metrics['rmse']:^10.4f} {h_metrics['correlation']:^12.4f}")

        print("-"*60)
        print("\n6th Hour Focus (as requested):")
        for key, value in metrics['hour_6_focus'].items():
            print(f"  {key.upper():15s}: {value:.4f}")

        if 'negative_bz' in metrics:
            print("\n" + "-"*60)
            print("Negative Bz Performance (Southward - Most Important):")
            for key, value in metrics['negative_bz'].items():
                if 'percentage' in key:
                    print(f"  {key:20s}: {value:.2f}%")
                else:
                    print(f"  {key:20s}: {value:.4f}")

        if 'strong_negative_bz' in metrics:
            print("\nStrong Negative Bz Performance (< -5 nT):")
            for key, value in metrics['strong_negative_bz'].items():
                if 'percentage' in key:
                    print(f"  {key:20s}: {value:.2f}%")
                else:
                    print(f"  {key:20s}: {value:.4f}")

        print("="*60)

    def _generate_plots(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metrics: Dict[str, Any]
    ):
        """Generate visualization plots."""
        sns.set_style("whitegrid")

        # 1. Scatter plots for each horizon
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for h in range(6):
            ax = axes[h]
            h_pred = predictions[:, h]
            h_target = targets[:, h]

            ax.scatter(h_target, h_pred, alpha=0.5, s=20)
            ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()],
                   'r--', lw=2, label='Perfect prediction')

            ax.set_xlabel('Target Bz (nT)', fontsize=10)
            ax.set_ylabel('Predicted Bz (nT)', fontsize=10)
            ax.set_title(f'Hour {h+1} Ahead\nMAE={metrics["per_horizon"][f"hour_{h+1}"]["mae"]:.3f} nT',
                        fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = os.path.join(self.output_dir, 'scatter_plots.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plots to {scatter_path}")
        plt.close()

        # 2. Error distribution for 6th hour
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        hour_6_pred = predictions[:, 5]
        hour_6_target = targets[:, 5]
        hour_6_error = hour_6_pred - hour_6_target

        # Histogram
        axes[0].hist(hour_6_error, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
        axes[0].set_xlabel('Prediction Error (nT)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title(f'6th Hour Error Distribution\nMean={hour_6_error.mean():.3f} nT, Std={hour_6_error.std():.3f} nT',
                         fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Scatter for 6th hour
        axes[1].scatter(hour_6_target, hour_6_pred, alpha=0.5, s=30)
        axes[1].plot([targets.min(), targets.max()], [targets.min(), targets.max()],
                    'r--', lw=2, label='Perfect prediction')
        axes[1].set_xlabel('Target Bz (nT)', fontsize=11)
        axes[1].set_ylabel('Predicted Bz (nT)', fontsize=11)
        axes[1].set_title(f'6th Hour Predictions\nRMSE={metrics["hour_6_focus"]["rmse"]:.3f} nT',
                         fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        hour6_path = os.path.join(self.output_dir, 'hour_6_analysis.png')
        plt.savefig(hour6_path, dpi=300, bbox_inches='tight')
        print(f"Saved 6th hour analysis to {hour6_path}")
        plt.close()

        # 3. MAE vs Forecast Horizon
        fig, ax = plt.subplots(figsize=(10, 6))

        hours = np.arange(1, 7)
        maes = [metrics['per_horizon'][f'hour_{h}']['mae'] for h in hours]
        rmses = [metrics['per_horizon'][f'hour_{h}']['rmse'] for h in hours]

        ax.plot(hours, maes, marker='o', linewidth=2, markersize=8, label='MAE')
        ax.plot(hours, rmses, marker='s', linewidth=2, markersize=8, label='RMSE')

        ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
        ax.set_ylabel('Error (nT)', fontsize=12)
        ax.set_title('Prediction Error vs Forecast Horizon', fontsize=14)
        ax.set_xticks(hours)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        horizon_path = os.path.join(self.output_dir, 'error_vs_horizon.png')
        plt.savefig(horizon_path, dpi=300, bbox_inches='tight')
        print(f"Saved error vs horizon plot to {horizon_path}")
        plt.close()


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: str) -> nn.Module:
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate Bz forecasting model')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
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
        help='Device to evaluate on'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--simple-dataset',
        action='store_true',
        help='Use SimpleBzDataset for testing'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = args.device
    print(f"Using device: {device}")

    # Create test dataset
    if args.simple_dataset:
        test_dataset = SimpleBzDataset(
            omni_bz_csv=config['omni_bz_csv'],
            num_samples=config.get('num_test_samples', 500),
            forecast_hours=config.get('forecast_hours', 6),
            sdo_shape=tuple(config.get('sdo_shape', [13, 512, 512])),
            seed=456,
        )
    else:
        test_dataset = BzForecastDataset(
            sdo_index_csv=config['sdo_test_csv'],
            omni_bz_csv=config['omni_bz_csv'],
            forecast_hours=config.get('forecast_hours', 6),
            sdo_data_dir=config.get('sdo_data_dir'),
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=0,
    )

    print(f"Test samples: {len(test_dataset)}")

    # Create model
    surya_config = config.get('surya_config', {})

    model = create_bz_model(
        strategy=args.strategy,
        surya_config=surya_config,
        forecast_hours=config.get('forecast_hours', 6),
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
    )

    # Load checkpoint
    model = load_checkpoint(args.checkpoint, model, device)

    # Create evaluator
    evaluator = BzEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
    )

    # Evaluate
    metrics = evaluator.evaluate()


if __name__ == '__main__':
    main()
