"""
Compare results from all three training strategies.

Usage:
    python compare_strategies.py \
        --full-results results/full/metrics.yaml \
        --frozen-results results/frozen/metrics.yaml \
        --lora-results results/lora/metrics.yaml \
        --output comparison_report.txt

Author: Vineet Vora
Date: 2025-11-27
"""

import argparse
import yaml
import pandas as pd
from typing import Dict, Any


def load_metrics(yaml_path: str) -> Dict[str, Any]:
    """Load metrics from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def create_comparison_table(metrics_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table from metrics."""

    # Overall comparison
    overall_data = []
    for strategy, metrics in metrics_dict.items():
        overall_data.append({
            'Strategy': strategy,
            'MAE': metrics['overall']['mae'],
            'RMSE': metrics['overall']['rmse'],
            'Correlation': metrics['overall']['correlation'],
        })

    overall_df = pd.DataFrame(overall_data)

    # Per-horizon comparison
    horizon_data = []
    for h in range(1, 7):
        row = {'Hour': h}
        for strategy, metrics in metrics_dict.items():
            h_metrics = metrics['per_horizon'][f'hour_{h}']
            row[f'{strategy}_MAE'] = h_metrics['mae']
            row[f'{strategy}_RMSE'] = h_metrics['rmse']
        horizon_data.append(row)

    horizon_df = pd.DataFrame(horizon_data)

    # 6th hour focus
    hour6_data = []
    for strategy, metrics in metrics_dict.items():
        hour6_data.append({
            'Strategy': strategy,
            'MAE_6h': metrics['hour_6_focus']['mae'],
            'RMSE_6h': metrics['hour_6_focus']['rmse'],
            'Correlation_6h': metrics['hour_6_focus']['correlation'],
        })

    hour6_df = pd.DataFrame(hour6_data)

    return overall_df, horizon_df, hour6_df


def print_comparison(
    overall_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    hour6_df: pd.DataFrame,
    output_file: str = None
):
    """Print formatted comparison."""

    output = []

    output.append("="*80)
    output.append("IMF Bz FORECASTING - STRATEGY COMPARISON")
    output.append("="*80)
    output.append("")

    # Overall performance
    output.append("OVERALL PERFORMANCE")
    output.append("-"*80)
    output.append(overall_df.to_string(index=False))
    output.append("")

    # Best strategy for each metric
    best_mae = overall_df.loc[overall_df['MAE'].idxmin(), 'Strategy']
    best_rmse = overall_df.loc[overall_df['RMSE'].idxmin(), 'Strategy']
    best_corr = overall_df.loc[overall_df['Correlation'].idxmax(), 'Strategy']

    output.append(f"Best MAE:         {best_mae}")
    output.append(f"Best RMSE:        {best_rmse}")
    output.append(f"Best Correlation: {best_corr}")
    output.append("")

    # 6th hour focus
    output.append("="*80)
    output.append("6TH HOUR PERFORMANCE (Primary Metric)")
    output.append("-"*80)
    output.append(hour6_df.to_string(index=False))
    output.append("")

    best_mae_6h = hour6_df.loc[hour6_df['MAE_6h'].idxmin(), 'Strategy']
    best_rmse_6h = hour6_df.loc[hour6_df['RMSE_6h'].idxmin(), 'Strategy']

    output.append(f"Best 6th hour MAE:  {best_mae_6h}")
    output.append(f"Best 6th hour RMSE: {best_rmse_6h}")
    output.append("")

    # Per-horizon
    output.append("="*80)
    output.append("PER-HORIZON COMPARISON (MAE)")
    output.append("-"*80)

    mae_cols = ['Hour'] + [col for col in horizon_df.columns if 'MAE' in col]
    output.append(horizon_df[mae_cols].to_string(index=False))
    output.append("")

    output.append("PER-HORIZON COMPARISON (RMSE)")
    output.append("-"*80)
    rmse_cols = ['Hour'] + [col for col in horizon_df.columns if 'RMSE' in col]
    output.append(horizon_df[rmse_cols].to_string(index=False))
    output.append("")

    # Recommendations
    output.append("="*80)
    output.append("RECOMMENDATIONS")
    output.append("-"*80)

    # Determine best overall strategy
    # Simple scoring: lowest MAE + RMSE for 6th hour
    hour6_df['Score'] = hour6_df['MAE_6h'] + hour6_df['RMSE_6h']
    best_overall = hour6_df.loc[hour6_df['Score'].idxmin(), 'Strategy']

    output.append(f"Recommended strategy: {best_overall}")
    output.append("")
    output.append("Rationale:")
    output.append(f"  - Based on 6th hour performance (as requested by professor)")
    output.append(f"  - {best_overall} achieves best MAE+RMSE combination")
    output.append("")

    # Strategy-specific notes
    output.append("Strategy-specific notes:")
    output.append("  - Full:   Maximum capacity, best if sufficient data and compute")
    output.append("  - Frozen: Fast baseline, good if pretrained features are optimal")
    output.append("  - LoRA:   Balanced approach, efficient adaptation with limited compute")
    output.append("")

    output.append("="*80)

    # Print to console
    report = "\n".join(output)
    print(report)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nComparison saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare results from different training strategies'
    )
    parser.add_argument(
        '--full-results',
        type=str,
        required=True,
        help='Path to full fine-tuning metrics.yaml'
    )
    parser.add_argument(
        '--frozen-results',
        type=str,
        required=True,
        help='Path to frozen encoder metrics.yaml'
    )
    parser.add_argument(
        '--lora-results',
        type=str,
        required=True,
        help='Path to LoRA metrics.yaml'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for comparison report'
    )

    args = parser.parse_args()

    # Load metrics
    metrics = {
        'Full': load_metrics(args.full_results),
        'Frozen': load_metrics(args.frozen_results),
        'LoRA': load_metrics(args.lora_results),
    }

    # Create comparison tables
    overall_df, horizon_df, hour6_df = create_comparison_table(metrics)

    # Print comparison
    print_comparison(overall_df, horizon_df, hour6_df, args.output)


if __name__ == '__main__':
    main()
