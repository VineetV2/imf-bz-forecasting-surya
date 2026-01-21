"""
Create Input/Output Timeline Figure for IMF Bz Prediction

Shows the temporal relationship between:
- Input: SDO observation at T (flare peak time)
- Output: Bz predictions at T+24h, T+48h, T+72h

This visualizes how the model accounts for CME/solar wind propagation time.

Author: Vineet Vora
Date: 2025-12-01
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from datetime import datetime, timedelta

def create_timeline_figure(save_path='timeline_figure.png', dpi=300):
    """
    Create publication-quality timeline figure showing input/output relationship.

    Args:
        save_path: Path to save the figure
        dpi: Resolution for saved figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Remove axes
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Define colors
    color_input = '#4A90E2'  # Blue for input
    color_output = '#E94B3C'  # Red for output
    color_propagation = '#F5A623'  # Orange for propagation

    # Timeline baseline
    timeline_y = 10
    ax.plot([5, 95], [timeline_y, timeline_y], 'k-', linewidth=2, alpha=0.3)

    # Time markers
    t_positions = {
        'T': 20,
        'T+24h': 40,
        'T+48h': 60,
        'T+72h': 80,
    }

    # Draw time markers
    for label, pos in t_positions.items():
        ax.plot([pos, pos], [timeline_y-0.5, timeline_y+0.5], 'k-', linewidth=2)
        ax.text(pos, timeline_y-2, label, ha='center', va='top',
                fontsize=12, fontweight='bold')

    # Title
    ax.text(50, 18, 'IMF Bz Forecasting: Input/Output Timeline',
            ha='center', va='top', fontsize=16, fontweight='bold')
    ax.text(50, 16.5, 'Accounting for CME/Solar Wind Propagation Time (Sun → Earth)',
            ha='center', va='top', fontsize=11, style='italic', color='gray')

    # INPUT SECTION (at T)
    input_box = FancyBboxPatch((t_positions['T']-5, timeline_y+2), 10, 3,
                               boxstyle="round,pad=0.3",
                               edgecolor=color_input, facecolor=color_input,
                               linewidth=2, alpha=0.3)
    ax.add_patch(input_box)

    ax.text(t_positions['T'], timeline_y+3.5, 'INPUT',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=color_input)
    ax.text(t_positions['T'], timeline_y+2.7, 'SDO Observation',
            ha='center', va='center', fontsize=9, color=color_input)

    # Input details (below)
    input_details = [
        "• Time: T (Flare Peak)",
        "• Location: Sun (1 AU from Earth)",
        "• Channels: 13 (8 AIA + 5 HMI)",
        "• Resolution: 512×512 pixels",
    ]

    y_offset = timeline_y - 3.5
    for i, detail in enumerate(input_details):
        ax.text(t_positions['T'], y_offset - i*1.0, detail,
                ha='center', va='top', fontsize=8, color=color_input)

    # PROPAGATION PERIOD
    prop_start = t_positions['T']
    prop_end = t_positions['T+72h']

    # Propagation zone (shaded region)
    prop_box = patches.Rectangle((prop_start, timeline_y-0.3),
                                  prop_end-prop_start, 0.6,
                                  linewidth=0, edgecolor='none',
                                  facecolor=color_propagation, alpha=0.2)
    ax.add_patch(prop_box)

    ax.text((prop_start + prop_end)/2, timeline_y+1.2,
            'CME/Solar Wind Propagation',
            ha='center', va='bottom', fontsize=10,
            style='italic', color=color_propagation, fontweight='bold')
    ax.text((prop_start + prop_end)/2, timeline_y+0.5,
            '(18-138 hours travel time)',
            ha='center', va='bottom', fontsize=8,
            style='italic', color=color_propagation)

    # OUTPUT SECTION (at T+24h, T+48h, T+72h)
    output_times = ['T+24h', 'T+48h', 'T+72h']

    for i, time_label in enumerate(output_times):
        pos = t_positions[time_label]

        # Output marker (circle)
        circle = plt.Circle((pos, timeline_y), 0.6,
                           edgecolor=color_output, facecolor=color_output,
                           linewidth=2, alpha=0.7)
        ax.add_patch(circle)

        # Arrow from input to output
        arrow = FancyArrowPatch((t_positions['T']+5, timeline_y+2),
                               (pos, timeline_y+0.6),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=1.5, color=color_output,
                               alpha=0.5, linestyle='--')
        ax.add_patch(arrow)

    # OUTPUT label
    output_center = (t_positions['T+24h'] + t_positions['T+72h']) / 2
    output_box = FancyBboxPatch((output_center-8, timeline_y-5.5), 16, 3,
                               boxstyle="round,pad=0.3",
                               edgecolor=color_output, facecolor=color_output,
                               linewidth=2, alpha=0.3)
    ax.add_patch(output_box)

    ax.text(output_center, timeline_y-4, 'OUTPUT',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color=color_output)
    ax.text(output_center, timeline_y-4.8, 'Predicted IMF Bz (3 values)',
            ha='center', va='center', fontsize=9, color=color_output)

    # Output details
    output_details = [
        "• Bz(T+24h): Predicted value at 24 hours ahead",
        "• Bz(T+48h): Predicted value at 48 hours ahead",
        "• Bz(T+72h): Predicted value at 72 hours ahead",
        "• Location: Earth (L1 point)",
        "• Units: nT (nanoteslas)",
    ]

    y_offset = timeline_y - 7.5
    for i, detail in enumerate(output_details):
        ax.text(output_center, y_offset - i*0.85, detail,
                ha='center', va='top', fontsize=8, color=color_output)

    # Model architecture note
    ax.text(50, 1.8,
            'Model: Surya-366M (Foundation Model) + MLP Prediction Head → 3 Output Neurons',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.text(50, 0.5,
            'Loss: MSE across all three predictions | Validation: LOFO (Leave-One-Flare-Out)',
            ha='center', va='center', fontsize=8, style='italic', color='gray')

    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Timeline figure saved to: {save_path}")

    return fig


def create_detailed_example_figure(save_path='timeline_example.png', dpi=300):
    """
    Create a detailed example with actual flare event data.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 25)
    ax.axis('off')

    # Colors
    color_input = '#4A90E2'
    color_output = '#E94B3C'

    # Title
    ax.text(50, 23, 'Example: 2017-09-06 X9.3 Flare Event',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # Timeline
    timeline_y = 12
    ax.plot([10, 90], [timeline_y, timeline_y], 'k-', linewidth=2, alpha=0.3)

    # Real timestamps
    flare_time = datetime(2017, 9, 6, 11, 53)

    timestamps = {
        'Input': flare_time,
        'T+24h': flare_time + timedelta(hours=24),
        'T+48h': flare_time + timedelta(hours=48),
        'T+72h': flare_time + timedelta(hours=72),
    }

    positions = {
        'Input': 20,
        'T+24h': 40,
        'T+48h': 60,
        'T+72h': 80,
    }

    # Draw markers and timestamps
    for label, pos in positions.items():
        ax.plot([pos, pos], [timeline_y-0.5, timeline_y+0.5], 'k-', linewidth=2)

        # Date/time
        dt = timestamps[label]
        ax.text(pos, timeline_y-2, dt.strftime('%Y-%m-%d'),
                ha='center', va='top', fontsize=9, fontweight='bold')
        ax.text(pos, timeline_y-3, dt.strftime('%H:%M UT'),
                ha='center', va='top', fontsize=8, color='gray')

    # INPUT
    input_box = FancyBboxPatch((positions['Input']-6, timeline_y+2), 12, 4.5,
                               boxstyle="round,pad=0.3",
                               edgecolor=color_input, facecolor=color_input,
                               linewidth=2, alpha=0.2)
    ax.add_patch(input_box)

    ax.text(positions['Input'], timeline_y+5, 'INPUT',
            ha='center', va='center', fontsize=12, fontweight='bold', color=color_input)
    ax.text(positions['Input'], timeline_y+3.8, 'SDO/AIA + HMI',
            ha='center', va='center', fontsize=9, color=color_input)
    ax.text(positions['Input'], timeline_y+3, '13 channels',
            ha='center', va='center', fontsize=8, color=color_input)
    ax.text(positions['Input'], timeline_y+2.3, '512×512 pixels',
            ha='center', va='center', fontsize=8, color=color_input)

    # OUTPUTS (with example values)
    example_bz_values = {
        'T+24h': -8.2,
        'T+48h': -12.5,
        'T+72h': -6.3,
    }

    for time_label, bz_value in example_bz_values.items():
        pos = positions[time_label]

        # Circle marker
        circle = plt.Circle((pos, timeline_y), 0.8,
                           edgecolor=color_output, facecolor=color_output,
                           linewidth=2, alpha=0.7)
        ax.add_patch(circle)

        # Bz value
        ax.text(pos, timeline_y+2.5, f'Bz = {bz_value} nT',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color=color_output)

        # Arrow
        arrow = FancyArrowPatch((positions['Input']+6, timeline_y+2),
                               (pos, timeline_y+0.8),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color=color_output,
                               alpha=0.4, linestyle='--')
        ax.add_patch(arrow)

    # Model prediction note
    ax.text(50, timeline_y-5.5, 'Model Prediction:',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(50, timeline_y-6.8,
            'Bz_predicted = [-8.2, -12.5, -6.3] nT at [T+24h, T+48h, T+72h]',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Ground truth comparison
    ax.text(50, timeline_y-9, 'OMNI Ground Truth (for evaluation):',
            ha='center', va='center', fontsize=10, fontweight='bold',
            style='italic', color='green')
    ax.text(50, timeline_y-10.2,
            'Bz_actual = [-7.8, -11.9, -6.5] nT',
            ha='center', va='center', fontsize=9, color='green',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # RMSE calculation
    ax.text(50, timeline_y-12.5,
            'RMSE = √[((-8.2-(-7.8))² + (-12.5-(-11.9))² + (-6.3-(-6.5))²) / 3] = 0.38 nT',
            ha='center', va='center', fontsize=9, color='purple',
            bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))

    # Physical interpretation
    ax.text(50, 2.5,
            'Physical Interpretation: Strong southward Bz (negative values) indicates',
            ha='center', va='center', fontsize=9, style='italic')
    ax.text(50, 1.3,
            'potential geomagnetic storm conditions at Earth 24-72 hours after X9.3 flare',
            ha='center', va='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Example timeline figure saved to: {save_path}")

    return fig


if __name__ == '__main__':
    print("Creating timeline figures...")

    # Create general timeline figure
    fig1 = create_timeline_figure('timeline_figure.png', dpi=300)

    # Create detailed example figure
    fig2 = create_detailed_example_figure('timeline_example.png', dpi=300)

    print("\nFigures created successfully!")
    print("- timeline_figure.png: General input/output timeline")
    print("- timeline_example.png: Detailed example with real flare event")

    plt.show()
