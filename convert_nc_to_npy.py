#!/usr/bin/env python3
"""
Convert NetCDF weather data to NPY format for PRIMER training
Supports ERA5, IMERG, and gauge data formats
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Tuple, Optional
import argparse
from tqdm import tqdm


def convert_era5_nc_to_npy(
    nc_file: str,
    output_dir: str,
    var_name: str = "tp",  # ERA5: total precipitation
    spatial_subset: Optional[Tuple[slice, slice]] = None,
    time_format: str = "%Y%m%d%H",
):
    """
    Convert ERA5 NetCDF to NPY format

    Args:
        nc_file: Path to input NetCDF file
        output_dir: Directory to save .npy files
        var_name: Variable name in NetCDF (e.g., 'tp', 'precip')
        spatial_subset: (lat_slice, lon_slice) for subsetting
        time_format: strftime format for output filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open NetCDF file
    ds = xr.open_dataset(nc_file)

    # Extract precipitation variable
    precip = ds[var_name]

    # Apply spatial subset if specified
    if spatial_subset:
        lat_slice, lon_slice = spatial_subset
        precip = precip.isel(latitude=lat_slice, longitude=lon_slice)

    # Process each time step
    for time_idx in tqdm(range(len(precip.time)), desc="Converting ERA5"):
        data = precip.isel(time=time_idx).values

        # Ensure shape is (1, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Generate filename from timestamp
        timestamp = precip.time.isel(time=time_idx).values
        timestamp_str = np.datetime_as_string(timestamp, unit='h').replace('-', '').replace('T', '')
        filename = f"{timestamp_str}.npy"

        # Save
        np.save(output_path / filename, data.astype(np.float32))

    print(f"✓ Converted {len(precip.time)} ERA5 timesteps to {output_path}")
    return output_path


def convert_imerg_nc_to_npy(
    nc_file: str,
    output_dir: str,
    var_name: str = "precipitationCal",
    spatial_subset: Optional[Tuple[slice, slice]] = None,
):
    """
    Convert IMERG NetCDF to NPY format
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_file)
    precip = ds[var_name]

    if spatial_subset:
        lat_slice, lon_slice = spatial_subset
        precip = precip.isel(lat=lat_slice, lon=lon_slice)

    for time_idx in tqdm(range(len(precip.time)), desc="Converting IMERG"):
        data = precip.isel(time=time_idx).values

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # IMERG filename format: YYYYMMDD_HHMM
        timestamp = precip.time.isel(time=time_idx).values
        timestamp_str = np.datetime_as_string(timestamp, unit='m').replace('-', '').replace('T', '_').replace(':', '')
        filename = f"{timestamp_str}.npy"

        np.save(output_path / filename, data.astype(np.float32))

    print(f"✓ Converted {len(precip.time)} IMERG timesteps to {output_path}")
    return output_path


def convert_gauge_nc_to_npy(
    precip_nc_file: str,
    mask_nc_file: Optional[str],
    output_dir: str,
    precip_var: str = "precip",
    mask_var: Optional[str] = "station_mask",
    spatial_subset: Optional[Tuple[slice, slice]] = None,
):
    """
    Convert gauge data to NPY format (2-channel: precipitation + mask)

    Args:
        precip_nc_file: NetCDF file with precipitation values
        mask_nc_file: NetCDF file with station mask (optional if in same file)
        output_dir: Directory to save .npy files
        precip_var: Precipitation variable name
        mask_var: Mask variable name
        spatial_subset: Spatial subsetting
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open precipitation data
    ds_precip = xr.open_dataset(precip_nc_file)
    precip = ds_precip[precip_var]

    # Open mask data
    if mask_nc_file:
        ds_mask = xr.open_dataset(mask_nc_file)
        mask = ds_mask[mask_var]
    elif mask_var in ds_precip:
        mask = ds_precip[mask_var]
    else:
        print("Warning: No mask provided. Creating mask from non-NaN values.")
        mask = None

    # Apply spatial subset
    if spatial_subset:
        lat_slice, lon_slice = spatial_subset
        precip = precip.isel(lat=lat_slice, lon=lon_slice)
        if mask is not None:
            mask = mask.isel(lat=lat_slice, lon=lon_slice)

    for time_idx in tqdm(range(len(precip.time)), desc="Converting Gauge"):
        precip_data = precip.isel(time=time_idx).values

        # Create or extract mask
        if mask is not None:
            if 'time' in mask.dims:
                mask_data = mask.isel(time=time_idx).values
            else:
                mask_data = mask.values  # Static mask
        else:
            # Create mask from non-NaN values
            mask_data = (~np.isnan(precip_data)).astype(np.float32)

        # Stack into 2-channel array: (2, H, W)
        data = np.stack([precip_data, mask_data], axis=0)

        # Generate filename
        timestamp = precip.time.isel(time=time_idx).values
        timestamp_str = np.datetime_as_string(timestamp, unit='h').replace('-', '').replace('T', '')
        filename = f"{timestamp_str}.npy"

        np.save(output_path / filename, data.astype(np.float32))

    print(f"✓ Converted {len(precip.time)} Gauge timesteps to {output_path}")
    return output_path


def batch_convert_directory(
    input_dir: str,
    output_dir: str,
    data_type: str,
    pattern: str = "*.nc",
    **kwargs
):
    """
    Batch convert all NetCDF files in a directory

    Args:
        input_dir: Directory containing .nc files
        output_dir: Output directory for .npy files
        data_type: 'era5', 'imerg', or 'gauge'
        pattern: File glob pattern
        **kwargs: Additional arguments for conversion functions
    """
    input_path = Path(input_dir)
    nc_files = sorted(input_path.glob(pattern))

    print(f"Found {len(nc_files)} NetCDF files in {input_dir}")

    converters = {
        'era5': convert_era5_nc_to_npy,
        'imerg': convert_imerg_nc_to_npy,
        'gauge': convert_gauge_nc_to_npy,
    }

    converter = converters.get(data_type.lower())
    if not converter:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'era5', 'imerg', or 'gauge'")

    for nc_file in nc_files:
        print(f"\nProcessing: {nc_file.name}")
        try:
            if data_type.lower() == 'gauge' and 'mask_nc_file' not in kwargs:
                kwargs['mask_nc_file'] = None
            converter(str(nc_file), output_dir, **kwargs)
        except Exception as e:
            print(f"✗ Error processing {nc_file.name}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NetCDF weather data to NPY format")
    parser.add_argument("--input", "-i", required=True, help="Input NetCDF file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for .npy files")
    parser.add_argument("--type", "-t", required=True, choices=['era5', 'imerg', 'gauge'],
                        help="Data type: era5, imerg, or gauge")
    parser.add_argument("--var", "-v", help="Variable name in NetCDF")
    parser.add_argument("--batch", action="store_true", help="Batch process directory")
    parser.add_argument("--lat-range", nargs=2, type=int, help="Latitude index range (start, end)")
    parser.add_argument("--lon-range", nargs=2, type=int, help="Longitude index range (start, end)")

    args = parser.parse_args()

    # Prepare spatial subset
    spatial_subset = None
    if args.lat_range and args.lon_range:
        spatial_subset = (
            slice(args.lat_range[0], args.lat_range[1]),
            slice(args.lon_range[0], args.lon_range[1])
        )

    kwargs = {}
    if args.var:
        kwargs['var_name' if args.type != 'gauge' else 'precip_var'] = args.var
    if spatial_subset:
        kwargs['spatial_subset'] = spatial_subset

    if args.batch:
        batch_convert_directory(args.input, args.output, args.type, **kwargs)
    else:
        converters = {
            'era5': convert_era5_nc_to_npy,
            'imerg': convert_imerg_nc_to_npy,
            'gauge': convert_gauge_nc_to_npy,
        }
        if args.type == 'gauge':
            kwargs['mask_nc_file'] = None
        converters[args.type](args.input, args.output, **kwargs)

    print("\n✓ Conversion complete!")
