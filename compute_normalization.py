#!/usr/bin/env python3
"""
Compute normalization parameters for PRIMER training data
Calculates min, max, mean, std for all .npy files in a directory
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def compute_normalization_stats(
    data_dir: str,
    output_file: str = "property_results.npy",
    channel_idx: int = 0,
    clip_range: tuple = (-3.0, 3.0),
):
    """
    Compute normalization statistics for .npy files

    Args:
        data_dir: Directory containing .npy files
        output_file: Output file for normalization parameters
        channel_idx: Which channel to compute stats for (0 for precip)
        clip_range: (min, max) for standard normalization clipping
    """
    data_path = Path(data_dir)
    npy_files = sorted(data_path.glob("*.npy"))

    if not npy_files:
        raise ValueError(f"No .npy files found in {data_dir}")

    print(f"Found {len(npy_files)} .npy files in {data_dir}")
    print(f"Computing statistics for channel {channel_idx}...")

    # Initialize accumulators
    sum_val = 0.0
    sum_sq = 0.0
    count = 0
    min_val = float('inf')
    max_val = float('-inf')

    # Process each file
    for file in tqdm(npy_files, desc="Computing stats"):
        try:
            data = np.load(file)

            # Handle different data shapes
            if data.ndim == 3:
                # (C, H, W) format - extract specified channel
                data = data[channel_idx]
            elif data.ndim == 2:
                # (H, W) format - use as is
                pass
            else:
                print(f"Warning: Unexpected shape {data.shape} in {file.name}, skipping")
                continue

            # Remove NaN values
            valid_data = data[~np.isnan(data)]

            if valid_data.size == 0:
                print(f"Warning: No valid data in {file.name}, skipping")
                continue

            # Update statistics
            sum_val += valid_data.sum()
            sum_sq += (valid_data ** 2).sum()
            count += valid_data.size
            min_val = min(min_val, valid_data.min())
            max_val = max(max_val, valid_data.max())

        except Exception as e:
            print(f"Error processing {file.name}: {e}")
            continue

    if count == 0:
        raise ValueError("No valid data found in any files!")

    # Compute final statistics
    mean = sum_val / count
    variance = sum_sq / count - mean ** 2
    std = np.sqrt(variance)

    # Create results dictionary
    results = {
        'min_val': float(min_val),
        'max_val': float(max_val),
        'mean': float(mean),
        'std': float(std),
        'clip_min': float(clip_range[0]),
        'clip_max': float(clip_range[1]),
        'num_files': len(npy_files),
        'num_values': count,
    }

    # Save results
    output_path = Path(output_file)
    np.save(output_path, results)

    # Print summary
    print("\n" + "="*60)
    print("NORMALIZATION PARAMETERS")
    print("="*60)
    print(f"Files processed:    {results['num_files']}")
    print(f"Total values:       {results['num_values']:,}")
    print(f"\nStatistics:")
    print(f"  Min value:        {min_val:.6f}")
    print(f"  Max value:        {max_val:.6f}")
    print(f"  Mean:             {mean:.6f}")
    print(f"  Std deviation:    {std:.6f}")
    print(f"\nClipping range (for standard normalization):")
    print(f"  Clip min:         {results['clip_min']}")
    print(f"  Clip max:         {results['clip_max']}")
    print("="*60)
    print(f"\n✓ Parameters saved to: {output_path.absolute()}")

    # Print usage instructions
    print("\nTo use in training, update your config file:")
    print("─"*60)
    print("# code/configs/training_config.py")
    print("")
    print("import numpy as np")
    print(f"norm_params = np.load('{output_file}', allow_pickle=True).item()")
    print("")
    print("data.min_val = norm_params['min_val']")
    print("data.max_val = norm_params['max_val']")
    print("data.mean = norm_params['mean']")
    print("data.std = norm_params['std']")
    print("data.clip_min = norm_params['clip_min']")
    print("data.clip_max = norm_params['clip_max']")
    print("data.normalization = 'standard'  # or 'minmax'")
    print("─"*60)

    return results


def verify_normalization(data_dir: str, params_file: str, num_samples: int = 5):
    """
    Verify normalization by applying it to sample files

    Args:
        data_dir: Directory with .npy files
        params_file: Path to normalization parameters
        num_samples: Number of files to verify
    """
    params = np.load(params_file, allow_pickle=True).item()
    data_path = Path(data_dir)
    npy_files = list(data_path.glob("*.npy"))[:num_samples]

    print(f"\nVerifying normalization on {len(npy_files)} sample files...")
    print("─"*60)

    for file in npy_files:
        data = np.load(file)
        if data.ndim == 3:
            data = data[0]

        # Apply standard normalization
        normalized = (data - params['mean']) / params['std']
        normalized = np.clip(normalized, params['clip_min'], params['clip_max'])

        valid_data = normalized[~np.isnan(normalized)]

        print(f"{file.name}:")
        print(f"  Original range:   [{data.min():.4f}, {data.max():.4f}]")
        print(f"  Normalized range: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
        print(f"  Normalized mean:  {valid_data.mean():.4f}")
        print(f"  Normalized std:   {valid_data.std():.4f}")

    print("─"*60)
    print("✓ Normalization looks correct!")
    print("  Expected: mean ≈ 0, std ≈ 1, range within clip bounds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute normalization parameters for PRIMER training data"
    )
    parser.add_argument(
        "--data-dir", "-d",
        required=True,
        help="Directory containing .npy files"
    )
    parser.add_argument(
        "--output", "-o",
        default="property_results.npy",
        help="Output file for normalization parameters (default: property_results.npy)"
    )
    parser.add_argument(
        "--channel", "-c",
        type=int,
        default=0,
        help="Channel index to compute stats for (default: 0, for gauge data with 2 channels)"
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=-3.0,
        help="Minimum clip value for standard normalization (default: -3.0)"
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=3.0,
        help="Maximum clip value for standard normalization (default: 3.0)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify normalization after computing"
    )

    args = parser.parse_args()

    # Compute normalization parameters
    results = compute_normalization_stats(
        data_dir=args.data_dir,
        output_file=args.output,
        channel_idx=args.channel,
        clip_range=(args.clip_min, args.clip_max)
    )

    # Optionally verify
    if args.verify:
        verify_normalization(args.data_dir, args.output)
