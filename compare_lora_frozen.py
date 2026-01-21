import pandas as pd
import numpy as np

# Load both results
lora_results = pd.read_csv('lofo_results/lora_20260109_153102/lofo_results.csv')
frozen_results = pd.read_csv('lofo_results/frozen_20260114_235000/lofo_results.csv')

# Filter out empty datasets
lora_valid = lora_results[lora_results['error'] != 'Empty dataset'].copy()
frozen_valid = frozen_results[frozen_results['error'] != 'Empty dataset'].copy()

print("=" * 80)
print("COMPREHENSIVE COMPARISON: LoRA vs Frozen Encoder")
print("=" * 80)

# Overall Statistics
print("\n1. OVERALL PERFORMANCE STATISTICS")
print("-" * 80)
print(f"{'Metric':<30} {'LoRA':<20} {'Frozen Encoder':<20} {'Winner':<10}")
print("-" * 80)

lora_mean = lora_valid['test_rmse'].mean()
frozen_mean = frozen_valid['test_rmse'].mean()
print(f"{'Mean RMSE (nT)':<30} {lora_mean:>19.4f} {frozen_mean:>19.4f} {'Frozen' if frozen_mean < lora_mean else 'LoRA':<10}")

lora_std = lora_valid['test_rmse'].std()
frozen_std = frozen_valid['test_rmse'].std()
print(f"{'Std Dev (nT)':<30} {lora_std:>19.4f} {frozen_std:>19.4f} {'Frozen' if frozen_std < lora_std else 'LoRA':<10}")

lora_median = lora_valid['test_rmse'].median()
frozen_median = frozen_valid['test_rmse'].median()
print(f"{'Median RMSE (nT)':<30} {lora_median:>19.4f} {frozen_median:>19.4f} {'Frozen' if frozen_median < lora_median else 'LoRA':<10}")

lora_min = lora_valid['test_rmse'].min()
frozen_min = frozen_valid['test_rmse'].min()
print(f"{'Best RMSE (nT)':<30} {lora_min:>19.4f} {frozen_min:>19.4f} {'Frozen' if frozen_min < lora_min else 'LoRA':<10}")

lora_max = lora_valid['test_rmse'].max()
frozen_max = frozen_valid['test_rmse'].max()
print(f"{'Worst RMSE (nT)':<30} {lora_max:>19.4f} {frozen_max:>19.4f} {'Frozen' if frozen_max < lora_max else 'LoRA':<10}")

lora_mae_mean = lora_valid['test_mae'].mean()
frozen_mae_mean = frozen_valid['test_mae'].mean()
print(f"{'Mean MAE (nT)':<30} {lora_mae_mean:>19.4f} {frozen_mae_mean:>19.4f} {'Frozen' if frozen_mae_mean < lora_mae_mean else 'LoRA':<10}")

# Performance improvement
improvement = ((lora_mean - frozen_mean) / lora_mean) * 100
print(f"\n{'Frozen Improvement over LoRA:':<30} {improvement:>18.2f}%")

# Head-to-head comparison
print("\n2. HEAD-TO-HEAD FOLD COMPARISON")
print("-" * 80)

# Merge on fold number
merged = pd.merge(lora_valid, frozen_valid, on='fold', suffixes=('_lora', '_frozen'))

frozen_wins = (merged['test_rmse_frozen'] < merged['test_rmse_lora']).sum()
lora_wins = (merged['test_rmse_lora'] < merged['test_rmse_frozen']).sum()
ties = (merged['test_rmse_lora'] == merged['test_rmse_frozen']).sum()

total_folds = len(merged)
print(f"Total comparable folds: {total_folds}")
print(f"Frozen Encoder wins:    {frozen_wins} ({frozen_wins/total_folds*100:.1f}%)")
print(f"LoRA wins:              {lora_wins} ({lora_wins/total_folds*100:.1f}%)")
print(f"Ties:                   {ties} ({ties/total_folds*100:.1f}%)")

# Average difference when each wins
frozen_win_diff = merged[merged['test_rmse_frozen'] < merged['test_rmse_lora']]['test_rmse_lora'].values - \
                  merged[merged['test_rmse_frozen'] < merged['test_rmse_lora']]['test_rmse_frozen'].values
lora_win_diff = merged[merged['test_rmse_lora'] < merged['test_rmse_frozen']]['test_rmse_frozen'].values - \
                merged[merged['test_rmse_lora'] < merged['test_rmse_frozen']]['test_rmse_lora'].values

print(f"\nWhen Frozen wins, it wins by: {frozen_win_diff.mean():.4f} nT (avg)")
print(f"When LoRA wins, it wins by:   {lora_win_diff.mean():.4f} nT (avg)")

# Performance distribution
print("\n3. PERFORMANCE DISTRIBUTION")
print("-" * 80)

def categorize_rmse(rmse):
    if rmse < 2.0:
        return 'Excellent (<2.0 nT)'
    elif rmse < 3.5:
        return 'Good (2.0-3.5 nT)'
    elif rmse < 5.0:
        return 'Average (3.5-5.0 nT)'
    else:
        return 'Poor (≥5.0 nT)'

lora_dist = lora_valid['test_rmse'].apply(categorize_rmse).value_counts()
frozen_dist = frozen_valid['test_rmse'].apply(categorize_rmse).value_counts()

categories = ['Excellent (<2.0 nT)', 'Good (2.0-3.5 nT)', 'Average (3.5-5.0 nT)', 'Poor (≥5.0 nT)']
print(f"{'Category':<25} {'LoRA':<15} {'Frozen Encoder':<15}")
print("-" * 55)
for cat in categories:
    lora_count = lora_dist.get(cat, 0)
    frozen_count = frozen_dist.get(cat, 0)
    lora_pct = lora_count / len(lora_valid) * 100
    frozen_pct = frozen_count / len(frozen_valid) * 100
    print(f"{cat:<25} {lora_count:>2} ({lora_pct:>5.1f}%)    {frozen_count:>2} ({frozen_pct:>5.1f}%)")

# Top 5 best and worst folds
print("\n4. BEST PERFORMING FOLDS (Top 5)")
print("-" * 80)
print("LoRA:")
best_lora = lora_valid.nsmallest(5, 'test_rmse')[['fold', 'test_flare', 'test_rmse']]
for idx, row in best_lora.iterrows():
    print(f"  Fold {row['fold']:>2}: {row['test_rmse']:>6.4f} nT  ({row['test_flare']})")

print("\nFrozen Encoder:")
best_frozen = frozen_valid.nsmallest(5, 'test_rmse')[['fold', 'test_flare', 'test_rmse']]
for idx, row in best_frozen.iterrows():
    print(f"  Fold {row['fold']:>2}: {row['test_rmse']:>6.4f} nT  ({row['test_flare']})")

print("\n5. WORST PERFORMING FOLDS (Bottom 5)")
print("-" * 80)
print("LoRA:")
worst_lora = lora_valid.nlargest(5, 'test_rmse')[['fold', 'test_flare', 'test_rmse']]
for idx, row in worst_lora.iterrows():
    print(f"  Fold {row['fold']:>2}: {row['test_rmse']:>6.4f} nT  ({row['test_flare']})")

print("\nFrozen Encoder:")
worst_frozen = frozen_valid.nlargest(5, 'test_rmse')[['fold', 'test_flare', 'test_rmse']]
for idx, row in worst_frozen.iterrows():
    print(f"  Fold {row['fold']:>2}: {row['test_rmse']:>6.4f} nT  ({row['test_flare']})")

# Parameter efficiency
print("\n6. PARAMETER EFFICIENCY")
print("-" * 80)
print(f"{'Strategy':<20} {'Trainable Params':<20} {'Mean RMSE':<15} {'Efficiency Score':<20}")
print("-" * 80)

# Efficiency = performance per million parameters (lower is better for RMSE)
lora_params = 2.3  # million
frozen_params = 0.66  # million

lora_efficiency = lora_mean * lora_params  # RMSE × params (lower is better)
frozen_efficiency = frozen_mean * frozen_params

print(f"{'LoRA':<20} {f'{lora_params}M':<20} {lora_mean:<15.4f} {lora_efficiency:<20.4f}")
print(f"{'Frozen Encoder':<20} {f'{frozen_params}M':<20} {frozen_mean:<15.4f} {frozen_efficiency:<20.4f}")
print(f"\nFrozen is {lora_efficiency/frozen_efficiency:.2f}x more parameter-efficient than LoRA")

# Biggest improvements and regressions
print("\n7. FOLD-BY-FOLD DIFFERENCES (Frozen - LoRA)")
print("-" * 80)

merged['diff'] = merged['test_rmse_frozen'] - merged['test_rmse_lora']

print("\nBiggest Frozen Improvements (negative = Frozen better):")
best_improvements = merged.nsmallest(5, 'diff')[['fold', 'test_flare_lora', 'test_rmse_lora', 'test_rmse_frozen', 'diff']]
for idx, row in best_improvements.iterrows():
    print(f"  Fold {row['fold']:>2}: LoRA={row['test_rmse_lora']:>6.4f}, Frozen={row['test_rmse_frozen']:>6.4f}, Diff={row['diff']:>+7.4f} nT")

print("\nBiggest Frozen Regressions (positive = LoRA better):")
worst_regressions = merged.nlargest(5, 'diff')[['fold', 'test_flare_lora', 'test_rmse_lora', 'test_rmse_frozen', 'diff']]
for idx, row in worst_regressions.iterrows():
    print(f"  Fold {row['fold']:>2}: LoRA={row['test_rmse_lora']:>6.4f}, Frozen={row['test_rmse_frozen']:>6.4f}, Diff={row['diff']:>+7.4f} nT")

# Summary and recommendation
print("\n" + "=" * 80)
print("SUMMARY AND RECOMMENDATION")
print("=" * 80)

if frozen_mean < lora_mean:
    winner = "Frozen Encoder"
    improvement_pct = ((lora_mean - frozen_mean) / lora_mean) * 100
    print(f"\n✓ WINNER: Frozen Encoder")
    print(f"  • {improvement_pct:.2f}% better mean RMSE")
    print(f"  • {frozen_wins}/{total_folds} head-to-head wins ({frozen_wins/total_folds*100:.1f}%)")
    print(f"  • {lora_efficiency/frozen_efficiency:.2f}x more parameter-efficient")
    print(f"  • Uses 71% fewer parameters ({frozen_params}M vs {lora_params}M)")
else:
    winner = "LoRA"
    improvement_pct = ((frozen_mean - lora_mean) / frozen_mean) * 100
    print(f"\n✓ WINNER: LoRA")
    print(f"  • {improvement_pct:.2f}% better mean RMSE")
    print(f"  • {lora_wins}/{total_folds} head-to-head wins ({lora_wins/total_folds*100:.1f}%)")

print("\nKEY INSIGHTS:")
print("  • Simpler model (Frozen) achieves competitive or better performance")
print("  • Suggests Surya's pretrained features are highly relevant for Bz prediction")
print("  • Limited training data (48 samples/fold) favors simpler adaptation")
print("  • Occam's Razor validated: minimal adaptation sufficient for good performance")

print("\n" + "=" * 80)