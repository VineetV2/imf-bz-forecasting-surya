"""
LOFO (Leave-One-Flare-Out) Cross-Validation Training Script

For each of the 58 flare events:
- Train on 57 flares
- Test on 1 held-out flare
- Record test metrics

This provides robust evaluation on limited data and accounts for
inter-flare variability.

Author: Vineet Vora
Date: 2025-12-01
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, List

from train import BzTrainer
from data.bz_dataset import BzForecastDataset
from models.bz_models import create_bz_model
from torch.utils.data import DataLoader, Subset


def load_all_flare_indices(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and combine all flare event indices.

    Returns:
        DataFrame with all flare events (train + val + test)
    """
    # Load all CSV files
    train_df = pd.read_csv(config['sdo_train_csv'])
    val_df = pd.read_csv(config['sdo_val_csv'])
    test_df = pd.read_csv(config['sdo_test_csv'])

    # Combine them
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df['datetime'] = pd.to_datetime(all_df['datetime'])
    all_df = all_df.sort_values('datetime').reset_index(drop=True)

    print(f"Total flare events: {len(all_df)}")
    return all_df


def create_lofo_split(
    all_indices: pd.DataFrame,
    test_idx: int,
    val_fraction: float = 0.12  # ~7 out of 57 for validation
) -> tuple:
    """
    Create LOFO train/val/test split.

    Args:
        all_indices: DataFrame with all flare indices
        test_idx: Index of test flare (0-57)
        val_fraction: Fraction of training set to use for validation

    Returns:
        (train_df, val_df, test_df)
    """
    # Test set: single flare
    test_df = all_indices.iloc[[test_idx]].copy()

    # Training set: all other flares
    train_indices = list(range(len(all_indices)))
    train_indices.remove(test_idx)
    train_val_df = all_indices.iloc[train_indices].copy()

    # Split train_val into train and val
    n_val = max(1, int(len(train_val_df) * val_fraction))
    val_df = train_val_df.iloc[:n_val].copy()
    train_df = train_val_df.iloc[n_val:].copy()

    return train_df, val_df, test_df


def train_single_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, float]:
    """
    Train and evaluate a single LOFO fold.

    Returns:
        Dictionary with test metrics
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_idx + 1}/58")
    print(f"Test flare: {test_df.iloc[0]['datetime']}")
    print(f"Train flares: {len(train_df)}, Val flares: {len(val_df)}")
    print('='*70)

    # Save temporary CSV files for this fold
    fold_dir = output_dir / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_csv = fold_dir / "train.csv"
    val_csv = fold_dir / "val.csv"
    test_csv = fold_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Create datasets
    forecast_horizons = config.get('forecast_horizons', [24, 48, 72])

    try:
        train_dataset = BzForecastDataset(
            sdo_index_csv=str(train_csv),
            omni_bz_csv=config['omni_bz_csv'],
            forecast_horizons=forecast_horizons,
            sdo_data_dir=config.get('sdo_data_dir'),
        )

        val_dataset = BzForecastDataset(
            sdo_index_csv=str(val_csv),
            omni_bz_csv=config['omni_bz_csv'],
            forecast_horizons=forecast_horizons,
            sdo_data_dir=config.get('sdo_data_dir'),
        )

        test_dataset = BzForecastDataset(
            sdo_index_csv=str(test_csv),
            omni_bz_csv=config['omni_bz_csv'],
            forecast_horizons=forecast_horizons,
            sdo_data_dir=config.get('sdo_data_dir'),
        )

    except Exception as e:
        print(f"Error creating datasets for fold {fold_idx}: {e}")
        return {
            'fold': fold_idx,
            'test_flare': str(test_df.iloc[0]['datetime']),
            'error': str(e),
            'test_rmse': np.nan,
        }

    # Check if datasets are valid
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print(f"Fold {fold_idx}: Empty dataset, skipping...")
        return {
            'fold': fold_idx,
            'test_flare': str(test_df.iloc[0]['datetime']),
            'error': 'Empty dataset',
            'test_rmse': np.nan,
        }

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=False,
        num_workers=0,
    ) if len(val_dataset) > 0 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    surya_config = config.get('surya_config', {})
    num_horizons = len(forecast_horizons)

    model = create_bz_model(
        strategy=args.strategy,
        surya_config=surya_config,
        num_horizons=num_horizons,
        surya_checkpoint=config.get('surya_checkpoint'),
        lora_r=config.get('lora_r', 8),
        lora_alpha=config.get('lora_alpha', 16),
        lora_dropout=config.get('lora_dropout', 0.1),
    )

    # Train model
    trainer = BzTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
    )

    # Reduce epochs for LOFO (too expensive to run full training 58 times)
    num_epochs = config.get('lofo_epochs', 10)  # Much fewer epochs per fold

    print(f"Training for {num_epochs} epochs...")
    trainer.train(num_epochs)

    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)

    # Save fold results
    fold_results = {
        'fold': fold_idx,
        'test_flare': str(test_df.iloc[0]['datetime']),
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_loss': test_metrics['loss'],
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
    }

    # Save checkpoint
    checkpoint_path = fold_dir / f"fold_{fold_idx}_best.pt"
    torch.save({
        'fold': fold_idx,
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
    }, checkpoint_path)

    print(f"Fold {fold_idx} Test RMSE: {test_metrics['rmse']:.4f} nT")

    return fold_results


def main():
    parser = argparse.ArgumentParser(description='LOFO Cross-Validation Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['full', 'frozen', 'lora'],
                       help='Training strategy')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu/cuda/mps)')
    parser.add_argument('--output_dir', type=str, default='./lofo_results',
                       help='Output directory for results')
    parser.add_argument('--start_fold', type=int, default=0,
                       help='Start from fold N (for resuming)')
    parser.add_argument('--end_fold', type=int, default=None,
                       help='End at fold N (for partial runs)')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"Using device: {args.device}")
    print(f"Strategy: {args.strategy}")

    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load all flare indices
    all_indices = load_all_flare_indices(config)
    n_folds = len(all_indices)

    # Determine fold range
    start_fold = args.start_fold
    end_fold = args.end_fold if args.end_fold is not None else n_folds

    print(f"\nRunning LOFO CV: Folds {start_fold} to {end_fold-1} (of {n_folds} total)")

    # Run LOFO cross-validation
    all_results = []

    for fold_idx in range(start_fold, end_fold):
        # Create LOFO split
        train_df, val_df, test_df = create_lofo_split(all_indices, fold_idx)

        # Train and evaluate this fold
        fold_results = train_single_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=config,
            args=args,
            output_dir=output_dir,
        )

        all_results.append(fold_results)

        # Save incremental results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / 'lofo_results.csv', index=False)

    # Compute aggregate statistics
    results_df = pd.DataFrame(all_results)

    # Filter out failed folds
    valid_results = results_df[results_df['test_rmse'].notna()]

    print(f"\n{'='*70}")
    print("LOFO CROSS-VALIDATION RESULTS")
    print('='*70)
    print(f"Total folds: {len(results_df)}")
    print(f"Successful folds: {len(valid_results)}")
    print(f"Failed folds: {len(results_df) - len(valid_results)}")
    print()
    print(f"Average Test RMSE: {valid_results['test_rmse'].mean():.4f} ± {valid_results['test_rmse'].std():.4f} nT")
    print(f"Average Test MAE: {valid_results['test_mae'].mean():.4f} ± {valid_results['test_mae'].std():.4f} nT")
    print(f"Median Test RMSE: {valid_results['test_rmse'].median():.4f} nT")
    print(f"Min Test RMSE: {valid_results['test_rmse'].min():.4f} nT")
    print(f"Max Test RMSE: {valid_results['test_rmse'].max():.4f} nT")

    # Save summary
    summary = {
        'n_folds_total': len(results_df),
        'n_folds_successful': len(valid_results),
        'n_folds_failed': len(results_df) - len(valid_results),
        'mean_test_rmse': float(valid_results['test_rmse'].mean()),
        'std_test_rmse': float(valid_results['test_rmse'].std()),
        'median_test_rmse': float(valid_results['test_rmse'].median()),
        'min_test_rmse': float(valid_results['test_rmse'].min()),
        'max_test_rmse': float(valid_results['test_rmse'].max()),
        'mean_test_mae': float(valid_results['test_mae'].mean()),
        'std_test_mae': float(valid_results['test_mae'].std()),
    }

    with open(output_dir / 'lofo_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print('='*70)


if __name__ == '__main__':
    main()
