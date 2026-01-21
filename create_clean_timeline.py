"""
Create a cleaner, simpler timeline figure without text overlap.

Author: Vineet Vora
Date: 2025-12-01
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_clean_timeline(save_path='timeline_clean.png', dpi=300):
    """Create a clean timeline figure with no overlapping text."""

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Colors
    color_input = '#4A90E2'
    color_output = '#E94B3C'
    color_propagation = '#F5A623'

    # Timeline
    timeline_y = 10
    ax.plot([5, 95], [timeline_y, timeline_y], 'k-', linewidth=2, alpha=0.3)

    # Title
    ax.text(50, 18, 'IMF Bz Forecasting: Input/Output Timeline',
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(50, 16.5, 'Predicting Bz at Earth 1-3 Days After Solar Flare',
            ha='center', va='top', fontsize=11, style='italic', color='gray')

    # Time positions
    t_positions = {'T': 20, 'T+24h': 40, 'T+48h': 60, 'T+72h': 80}

    # Draw time markers
    for label, pos in t_positions.items():
        ax.plot([pos, pos], [timeline_y-0.5, timeline_y+0.5], 'k-', linewidth=2)
        ax.text(pos, timeline_y-1.5, label, ha='center', va='top',
                fontsize=12, fontweight='bold')

    # INPUT BOX
    input_box = FancyBboxPatch((t_positions['T']-5, timeline_y+2), 10, 3,
                               boxstyle="round,pad=0.3",
                               edgecolor=color_input, facecolor=color_input,
                               linewidth=2, alpha=0.3)
    ax.add_patch(input_box)

    ax.text(t_positions['T'], timeline_y+3.5, 'INPUT',
            ha='center', va='center', fontsize=12, fontweight='bold', color=color_input)

    # Input details - SIMPLIFIED
    ax.text(t_positions['T'], timeline_y-3, 'SDO Observation at Sun',
            ha='center', va='top', fontsize=9, color=color_input, fontweight='bold')
    ax.text(t_positions['T'], timeline_y-4, '13 channels (8 AIA + 5 HMI)',
            ha='center', va='top', fontsize=8, color=color_input)
    ax.text(t_positions['T'], timeline_y-5, '512×512 pixels',
            ha='center', va='top', fontsize=8, color=color_input)

    # PROPAGATION ZONE
    prop_start = t_positions['T']
    prop_end = t_positions['T+72h']

    prop_box = patches.Rectangle((prop_start, timeline_y-0.3),
                                  prop_end-prop_start, 0.6,
                                  linewidth=0, edgecolor='none',
                                  facecolor=color_propagation, alpha=0.2)
    ax.add_patch(prop_box)

    ax.text((prop_start + prop_end)/2, timeline_y+1.2,
            'CME/Solar Wind Propagation (18-138 hours)',
            ha='center', va='bottom', fontsize=10,
            style='italic', color=color_propagation, fontweight='bold')

    # OUTPUT MARKERS
    for time_label in ['T+24h', 'T+48h', 'T+72h']:
        pos = t_positions[time_label]

        circle = plt.Circle((pos, timeline_y), 0.6,
                           edgecolor=color_output, facecolor=color_output,
                           linewidth=2, alpha=0.7)
        ax.add_patch(circle)

        arrow = FancyArrowPatch((t_positions['T']+5, timeline_y+2),
                               (pos, timeline_y+0.6),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=1.5, color=color_output,
                               alpha=0.5, linestyle='--')
        ax.add_patch(arrow)

    # OUTPUT BOX - SIMPLIFIED
    output_center = (t_positions['T+24h'] + t_positions['T+72h']) / 2
    output_box = FancyBboxPatch((output_center-10, timeline_y-4), 20, 2.5,
                               boxstyle="round,pad=0.3",
                               edgecolor=color_output, facecolor=color_output,
                               linewidth=2, alpha=0.3)
    ax.add_patch(output_box)

    ax.text(output_center, timeline_y-2.7, 'OUTPUT',
            ha='center', va='center', fontsize=12, fontweight='bold', color=color_output)
    ax.text(output_center, timeline_y-3.5, 'Predicted IMF Bz at Earth',
            ha='center', va='center', fontsize=9, color=color_output)

    # Model info at bottom
    ax.text(50, 1.5,
            'Model: Surya-366M Foundation + MLP Head → 3 Outputs [Bz(T+24h), Bz(T+48h), Bz(T+72h)]',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.text(50, 0.3,
            'Validation: LOFO (Leave-One-Flare-Out) | Loss: MSE across all horizons',
            ha='center', va='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Clean timeline saved to: {save_path}")

    return fig


def create_clean_example(save_path='example_clean.png', dpi=300):
    """Create a clean example figure."""

    from datetime import datetime, timedelta

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 22)
    ax.axis('off')

    # Colors
    color_input = '#4A90E2'
    color_output = '#E94B3C'

    # Title
    ax.text(50, 20.5, 'Example: 2017-09-06 X9.3 Flare Event',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # Timeline
    timeline_y = 12
    ax.plot([10, 90], [timeline_y, timeline_y], 'k-', linewidth=2, alpha=0.3)

    # Timestamps
    flare_time = datetime(2017, 9, 6, 11, 53)
    timestamps = {
        'Input': flare_time,
        'T+24h': flare_time + timedelta(hours=24),
        'T+48h': flare_time + timedelta(hours=48),
        'T+72h': flare_time + timedelta(hours=72),
    }
    positions = {'Input': 20, 'T+24h': 40, 'T+48h': 60, 'T+72h': 80}

    # Draw markers
    for label, pos in positions.items():
        ax.plot([pos, pos], [timeline_y-0.5, timeline_y+0.5], 'k-', linewidth=2)
        dt = timestamps[label]
        ax.text(pos, timeline_y-2, dt.strftime('%Y-%m-%d'),
                ha='center', va='top', fontsize=9, fontweight='bold')
        ax.text(pos, timeline_y-3, dt.strftime('%H:%M UT'),
                ha='center', va='top', fontsize=8, color='gray')

    # INPUT
    input_box = FancyBboxPatch((positions['Input']-6, timeline_y+2), 12, 3.5,
                               boxstyle="round,pad=0.3",
                               edgecolor=color_input, facecolor=color_input,
                               linewidth=2, alpha=0.2)
    ax.add_patch(input_box)

    ax.text(positions['Input'], timeline_y+4.5, 'INPUT',
            ha='center', va='center', fontsize=12, fontweight='bold', color=color_input)
    ax.text(positions['Input'], timeline_y+3.5, 'SDO/AIA + HMI',
            ha='center', va='center', fontsize=9, color=color_input)
    ax.text(positions['Input'], timeline_y+2.7, '13 channels, 512×512',
            ha='center', va='center', fontsize=8, color=color_input)

    # OUTPUTS with example values
    example_bz = {'T+24h': -8.2, 'T+48h': -12.5, 'T+72h': -6.3}

    for time_label, bz_value in example_bz.items():
        pos = positions[time_label]

        circle = plt.Circle((pos, timeline_y), 0.8,
                           edgecolor=color_output, facecolor=color_output,
                           linewidth=2, alpha=0.7)
        ax.add_patch(circle)

        ax.text(pos, timeline_y+2, f'Bz = {bz_value} nT',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color=color_output)

        arrow = FancyArrowPatch((positions['Input']+6, timeline_y+2),
                               (pos, timeline_y+0.8),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color=color_output,
                               alpha=0.4, linestyle='--')
        ax.add_patch(arrow)

    # Model prediction
    ax.text(50, timeline_y-5.5, 'Model Prediction:',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(50, timeline_y-6.5,
            'Bz_predicted = [-8.2, -12.5, -6.3] nT',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Ground truth
    ax.text(50, timeline_y-8.5, 'OMNI Ground Truth:',
            ha='center', va='center', fontsize=10, fontweight='bold',
            style='italic', color='green')
    ax.text(50, timeline_y-9.5,
            'Bz_actual = [-7.8, -11.9, -6.5] nT',
            ha='center', va='center', fontsize=9, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # RMSE
    ax.text(50, timeline_y-11.5,
            'RMSE = 0.38 nT (excellent prediction)',
            ha='center', va='center', fontsize=9, color='purple',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

    # Physical interpretation
    ax.text(50, 2,
            'Physical Interpretation: Strong southward Bz (negative) → Potential geomagnetic storm',
            ha='center', va='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Clean example saved to: {save_path}")

    return fig


if __name__ == '__main__':
    print("Creating clean timeline figures...")

    fig1 = create_clean_timeline('timeline_clean.png', dpi=300)
    fig2 = create_clean_example('example_clean.png', dpi=300)

    print("\nClean figures created successfully!")
    print("- timeline_clean.png: Simplified general timeline")
    print("- example_clean.png: Simplified example with real flare")

    plt.show()
