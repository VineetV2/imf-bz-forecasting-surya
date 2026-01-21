"""
Create SDO index CSV from existing preprocessed .nc files.

This script scans for .nc files and creates an index CSV
that can be used with BzForecastDataset.

Author: Vineet Vora
Date: 2025-11-27
"""

import os
import pandas as pd
from datetime import datetime
import argparse


def parse_nc_filename(filename):
    """
    Parse timestamp from .nc filename.

    Expected format: YYYYMMDD_HHMM.nc
    Example: 20110120_0200.nc → 2011-01-20 02:00:00

    Args:
        filename: Name of .nc file

    Returns:
        datetime object or None if parsing fails
    """
    try:
        # Remove .nc extension
        timestamp_str = filename.replace('.nc', '')

        # Parse YYYYMMDD_HHMM format
        dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M')
        return dt
    except Exception as e:
        print(f"Warning: Could not parse {filename}: {e}")
        return None


def create_sdo_index(
    nc_directory,
    output_csv='sdo_index.csv',
    relative_paths=True
):
    """
    Create SDO index CSV from .nc files in directory.

    Args:
        nc_directory: Directory containing .nc files
        output_csv: Output CSV path
        relative_paths: If True, use relative paths in CSV

    Returns:
        DataFrame with index
    """
    print(f"Scanning for .nc files in: {nc_directory}")

    # Find all .nc files
    nc_files = []
    for root, dirs, files in os.walk(nc_directory):
        for file in files:
            if file.endswith('.nc'):
                full_path = os.path.join(root, file)
                nc_files.append((file, full_path))

    print(f"Found {len(nc_files)} .nc files")

    # Parse timestamps and create index
    data = []
    for filename, full_path in sorted(nc_files):
        dt = parse_nc_filename(filename)

        if dt is not None:
            # Use relative or absolute path
            if relative_paths:
                file_path = filename
            else:
                file_path = full_path

            data.append({
                'datetime': dt,
                'filepath': file_path
            })

    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values('datetime').reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print(f"\nCreated index with {len(df)} samples")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Saved to: {output_csv}")

    return df


def split_train_val_test(
    index_csv,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    output_dir='.'
):
    """
    Split SDO index into train/val/test sets.

    Uses temporal split (not random) to avoid data leakage.

    Args:
        index_csv: Path to full index CSV
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        output_dir: Directory to save split CSVs

    Returns:
        Tuple of (train_csv, val_csv, test_csv) paths
    """
    df = pd.read_csv(index_csv)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Split by time
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]

    # Save splits
    train_csv = os.path.join(output_dir, 'sdo_train.csv')
    val_csv = os.path.join(output_dir, 'sdo_val.csv')
    test_csv = os.path.join(output_dir, 'sdo_test.csv')

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"\nCreated train/val/test split:")
    print(f"  Train: {len(train_df)} samples ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"  Val:   {len(val_df)} samples ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
    print(f"  Test:  {len(test_df)} samples ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    print(f"\nSaved to:")
    print(f"  {train_csv}")
    print(f"  {val_csv}")
    print(f"  {test_csv}")

    return train_csv, val_csv, test_csv


def main():
    parser = argparse.ArgumentParser(
        description='Create SDO index from preprocessed .nc files'
    )
    parser.add_argument(
        '--nc-directory',
        type=str,
        default='../downstream_examples/solar_wind_forcasting/assets/infer_data',
        help='Directory containing .nc files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/sdo_index.csv',
        help='Output CSV path'
    )
    parser.add_argument(
        '--split',
        action='store_true',
        help='Also create train/val/test splits'
    )
    parser.add_argument(
        '--absolute-paths',
        action='store_true',
        help='Use absolute paths instead of relative'
    )

    args = parser.parse_args()

    # Create index
    df = create_sdo_index(
        nc_directory=args.nc_directory,
        output_csv=args.output,
        relative_paths=not args.absolute_paths
    )

    # Create splits if requested
    if args.split:
        print("\n" + "="*60)
        split_train_val_test(
            index_csv=args.output,
            output_dir=os.path.dirname(args.output) or '.'
        )

    print("\n" + "="*60)
    print("Done! You can now use this index with BzForecastDataset")
    print("="*60)


if __name__ == '__main__':
    main()
