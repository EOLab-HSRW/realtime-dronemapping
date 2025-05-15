import os
import glob
import argparse
import sys
import time
import warnings
import math
from typing import List, Optional, Tuple, Dict, Any

import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.enums import Resampling # Only if needed for reading different resolutions
import numpy as np
from scipy.ndimage import distance_transform_edt # Efficient distance calculation
from tqdm import tqdm # Progress bar

# Suppress RuntimeWarnings often encountered with nan comparisons in blending
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered in.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

def get_dtype_from_source(src: rasterio.DatasetReader) -> Optional[np.dtype]:
    """
    Attempts to get dtype robustly, checking src.dtypes[0] first,
    then falling back to reading a pixel if necessary.
    """
    dtype_str: Optional[str] = None
    try:
        # --- FIX: Use src.dtypes[0] instead of src.dtype ---
        if src.dtypes and len(src.dtypes) > 0:
            dtype_str = src.dtypes[0]
        # ----------------------------------------------------
        else:
            warnings.warn(f"Source {os.path.basename(src.name)} has empty 'dtypes' tuple. Attempting read.")

        if dtype_str:
            # Convert rasterio dtype string to numpy dtype
            return np.dtype(dtype_str)
        else:
            # Fall through to reading data if dtype_str is still None
            pass

    except AttributeError:
         warnings.warn(f"Source {os.path.basename(src.name)} lacks 'dtypes' attribute. Attempting read.")
    except IndexError:
         warnings.warn(f"Source {os.path.basename(src.name)} 'dtypes' tuple is empty. Attempting read.")
    except TypeError as e:
         warnings.warn(f"Could not interpret dtype string '{dtype_str}' for {os.path.basename(src.name)}: {e}. Attempting read.")
    except Exception as e: # Catch other potential errors during dtype access
        warnings.warn(f"Unexpected error accessing dtype for {os.path.basename(src.name)}: {e}. Attempting read.")


    # If we reach here, getting dtype from metadata failed, try reading data
    try:
        # Read a small window (e.g., first pixel or 10x10 block) from the first band
        read_window = Window(0, 0, min(10, src.width), min(10, src.height))
        if read_window.width > 0 and read_window.height > 0:
            # Ensure we read band 1 (index 0) if multiple bands exist
            band_index_to_read = 1 # Rasterio uses 1-based indexing for read()
            sample_data = src.read(band_index_to_read, window=read_window)
            return sample_data.dtype
        else:
            warnings.warn(f"Source {os.path.basename(src.name)} has zero dimensions, cannot infer dtype.")
            return None
    except Exception as e:
        warnings.warn(f"Could not read data from {os.path.basename(src.name)} to infer dtype: {e}")
        return None


def calculate_weights(data_mask: np.ndarray, blend_distance: int) -> np.ndarray:
    """
    Calculates pixel weights based on distance to the nearest False value (nodata/edge).
    (Function remains the same as before)
    """
    if blend_distance <= 0:
        return data_mask.astype(np.float32)
    distances = distance_transform_edt(data_mask)
    weights = np.clip(distances, 0, blend_distance) / blend_distance
    weights = weights.astype(np.float32)
    weights[~data_mask] = 0.0
    return weights

def create_smooth_mosaic(input_dir: str, output_file: str,
                         output_format: str = 'GTiff',
                         blend_distance: int = 100, # Pixels for blending overlap
                         nodata_val: Optional[str] = None, # Explicit nodata (read as string)
                         resampling_method: str = 'nearest', # For reading, if needed
                         block_size: int = 512, # Processing chunk size
                         verbose: bool = False):
    """
    Stitches georeferenced TIFFs with custom weighted blending for smooth transitions.
    Includes robust dtype handling using src.dtypes[0].

    Args:
        input_dir: Directory containing input GeoTIFF files (.tif, .tiff).
        output_file: Path for the resulting mosaic file.
        output_format: Output format driver ('GTiff', 'PNG', etc.).
        blend_distance: Distance in pixels for feathering/blending overlap zones.
        nodata_val: Explicit nodata value override (as string from argparse). If None, attempts auto-detect.
        resampling_method: Rasterio resampling method if reading data requires it.
        block_size: Size of processing chunks (windows) in pixels.
        verbose: If True, prints detailed progress information.
    """
    print(f"Starting smooth mosaic creation (Quality Focus)...")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Blend Distance: {blend_distance} pixels")
    print(f"Processing Block Size: {block_size}x{block_size} pixels")

    resampling_alg = Resampling[resampling_method]
    start_time = time.time()

    # --- 1. Find and Validate Input Files ---
    search_paths = [os.path.join(input_dir, '*.tif'), os.path.join(input_dir, '*.tiff')]
    input_files = sorted([f for path in search_paths for f in glob.glob(path)])

    if not input_files:
        print(f"ERROR: No .tif or .tiff files found in '{input_dir}'.")
        sys.exit(1)
    print(f"Found {len(input_files)} input TIFF files.")

    sources: List[rasterio.DatasetReader] = []
    try:
        print("Opening input files and reading metadata...")
        first_profile: Optional[Dict[str, Any]] = None
        common_crs: Optional[rasterio.crs.CRS] = None
        common_dtype: Optional[np.dtype] = None # Will be determined robustly
        common_count: Optional[int] = None
        all_bounds: List[Tuple[float, float, float, float]] = []
        detected_nodata_values = set()

        # --- Metadata Reading Loop ---
        for i, fpath in enumerate(tqdm(input_files, desc="Reading Metadata")):
            try:
                src = rasterio.open(fpath)
                sources.append(src)
                all_bounds.append(src.bounds)
                current_dtype = get_dtype_from_source(src) # Use robust function

                if i == 0:
                    # Establish baseline from first file
                    first_profile = src.profile.copy() # Get profile first
                    common_crs = src.crs
                    common_count = src.count
                    common_dtype = current_dtype # Store determined dtype

                    if common_dtype is None:
                        # Close already opened sources before exiting
                        for s in sources: s.close()
                        raise ValueError(f"Could not determine data type for the first input file: {fpath}. Cannot proceed.")
                    if not common_crs:
                         warnings.warn(f"Input {os.path.basename(fpath)} lacks CRS info!")

                    # Update profile with determined dtype NAME string
                    first_profile['dtype'] = common_dtype.name

                    if src.nodata is not None:
                        detected_nodata_values.add(src.nodata)

                    if verbose:
                        print(f"\nReference file: {os.path.basename(fpath)}")
                        print(f"  Determined common Dtype: {common_dtype}")
                        print(f"  CRS: {common_crs}")
                        print(f"  Bands: {common_count}")

                else:
                    # Check consistency with the first file
                    if src.crs != common_crs:
                        warnings.warn(f"CRS mismatch: {os.path.basename(fpath)} ({src.crs}) "
                                      f"differs from reference ({common_crs}). Results may be incorrect. "
                                      f"Preprocessing (reprojection) is recommended.")
                    if src.count != common_count:
                         # Close already opened sources before exiting
                        for s in sources: s.close()
                        raise ValueError(f"Band count mismatch: {os.path.basename(fpath)} ({src.count}) "
                                         f"vs reference ({common_count}).")
                    if current_dtype != common_dtype:
                         warnings.warn(f"Data type mismatch: {os.path.basename(fpath)} (detected {current_dtype}) "
                                       f"differs from reference ({common_dtype}). Mosaic will use reference type.")
                    if src.nodata is not None:
                        detected_nodata_values.add(src.nodata)

            except Exception as e:
                print(f"\nERROR reading or validating {fpath}: {e}")
                # Ensure cleanup if error occurs during the loop
                for s in sources:
                    if not s.closed: s.close()
                raise # Re-raise to be caught by the outer try/finally

        # --- Check if metadata initialization was successful ---
        if not first_profile or common_count is None or common_dtype is None:
             raise ValueError("Could not establish common metadata (profile/bands/dtype) from inputs.")

        # --- Determine Final Nodata Value (using common_dtype) ---
        # (Nodata determination logic remains the same as the previous version)
        final_nodata_typed: Optional[Any] = None # Will hold nodata value cast to common_dtype
        if nodata_val is None: # No explicit override from user
            if len(detected_nodata_values) > 1:
                ref_nodata = first_profile.get('nodata')
                warnings.warn(f"Multiple nodata values detected: {detected_nodata_values}. "
                              f"Using value from first file: {ref_nodata}. "
                              f"Consider using --nodata_val for explicit control.")
                if ref_nodata is not None:
                    try: final_nodata_typed = common_dtype.type(ref_nodata)
                    except (ValueError, TypeError) as e:
                         warnings.warn(f"Could not cast nodata value {ref_nodata} from first file to common dtype {common_dtype}: {e}"); final_nodata_typed = None
            elif len(detected_nodata_values) == 1:
                detected_val = list(detected_nodata_values)[0]
                try:
                    final_nodata_typed = common_dtype.type(detected_val)
                    print(f"INFO: Using consistent detected nodata value: {final_nodata_typed} (dtype: {common_dtype})")
                except (ValueError, TypeError) as e:
                     warnings.warn(f"Could not cast detected nodata value {detected_val} to common dtype {common_dtype}: {e}"); final_nodata_typed = None
            else: final_nodata_typed = None; print("INFO: No nodata value detected in input files.")
        else: # User provided --nodata_val (as string)
             if nodata_val.lower() == 'none': final_nodata_typed = None; print(f"INFO: Explicitly setting no nodata value for the output.")
             else:
                 try: final_nodata_typed = common_dtype.type(nodata_val); print(f"INFO: Using specified nodata value: {final_nodata_typed} (dtype: {common_dtype})")
                 except (ValueError, TypeError): raise ValueError(f"Could not convert provided nodata_val '{nodata_val}' to common data type '{common_dtype}'.")


        # --- Calculate Output Bounds and Transform ---
        # (Bounds and Transform calculation remains the same)
        print("Calculating output bounds and transform...")
        dst_bounds = rasterio.coords.BoundingBox(
            left=min(b.left for b in all_bounds), bottom=min(b.bottom for b in all_bounds),
            right=max(b.right for b in all_bounds), top=max(b.top for b in all_bounds)
        )
        output_transform = rasterio.transform.from_bounds(
            *dst_bounds,
            width=math.ceil((dst_bounds.right - dst_bounds.left) / first_profile['transform'].a),
            height=math.ceil((dst_bounds.top - dst_bounds.bottom) / abs(first_profile['transform'].e)) # Use abs for neg e
        )
        output_width = math.ceil((dst_bounds.right - dst_bounds.left) / output_transform.a)
        output_height = math.ceil((dst_bounds.top - dst_bounds.bottom) / abs(output_transform.e)) # Use abs
        print(f"Output dimensions: {output_width}W x {output_height}H pixels")


        # --- Prepare Output File Profile ---
        profile = first_profile.copy()
        profile.update({
            'crs': common_crs,
            'transform': output_transform,
            'width': output_width,
            'height': output_height,
            'nodata': final_nodata_typed, # Use the typed nodata value
            'driver': output_format,
            'count': common_count,
            'dtype': common_dtype.name # Ensure profile dtype string is set
        })
        if output_format == 'GTiff':
            profile['compress'] = 'lzw'; profile['predictor'] = 2
            profile['tiled'] = True; profile['blockxsize'] = block_size
            profile['blockysize'] = block_size; profile['bigtiff'] = 'yes'

        # --- Process and Write Output in Windows ---
        print(f"\nProcessing mosaic in {block_size}x{block_size} blocks...")
        with rasterio.open(output_file, 'w', **profile) as dst:
            # (Window iteration logic remains the same)
            n_windows_x = math.ceil(output_width / block_size)
            n_windows_y = math.ceil(output_height / block_size)
            total_windows = n_windows_x * n_windows_y
            dest_buffer = np.zeros((common_count, block_size, block_size), dtype=np.float64)
            weight_sum_buffer = np.zeros((block_size, block_size), dtype=np.float64)
            progress = tqdm(total=total_windows, desc="Writing Mosaic Blocks", unit="block")

            for j in range(n_windows_y):
                for i in range(n_windows_x):
                    # (Window definition, buffer reset, overlap check logic is the same)
                    window = Window(i * block_size, j * block_size,
                                    min(block_size, output_width - i * block_size),
                                    min(block_size, output_height - j * block_size))
                    current_block_height, current_block_width = window.height, window.width
                    dest_buffer_slice = dest_buffer[:, :current_block_height, :current_block_width]
                    weight_sum_buffer_slice = weight_sum_buffer[:current_block_height, :current_block_width]
                    dest_buffer_slice.fill(0)
                    weight_sum_buffer_slice.fill(0)
                    overlapping_sources_indices = [idx for idx, src in enumerate(sources)
                                                   if not rasterio.coords.disjoint_bounds(dst.window_bounds(window), src.bounds)]

                    if not overlapping_sources_indices:
                         if final_nodata_typed is not None:
                              nodata_fill = np.full((common_count, window.height, window.width), final_nodata_typed, dtype=common_dtype)
                              dst.write(nodata_fill, window=window)
                         progress.update(1)
                         continue

                    # Read data, calculate weights, accumulate
                    # (Inner loop logic for reading/blending remains largely the same,
                    # ensuring use of final_nodata_typed and common_dtype)
                    for src_idx in overlapping_sources_indices:
                        src = sources[src_idx]
                        src_nodata_typed = None
                        if src.nodata is not None:
                            try: src_nodata_typed = common_dtype.type(src.nodata)
                            except (ValueError, TypeError): pass

                        try:
                            src_window = rasterio.windows.from_bounds(*dst.window_bounds(window), src.transform).round_offsets().round_lengths()
                            src_window = src_window.intersection(Window(0, 0, src.width, src.height))
                            if src_window.width <= 0 or src_window.height <= 0: continue

                            fill_val = src_nodata_typed if src_nodata_typed is not None else final_nodata_typed

                            src_data = src.read(window=src_window, out_shape=(common_count, window.height, window.width),
                                                resampling=resampling_alg, boundless=True, fill_value=fill_val, masked=True)

                            if np.ma.is_masked(src_data): valid_data_mask = ~src_data.mask[0]
                            else:
                                valid_data_mask = np.ones((window.height, window.width), dtype=bool)
                                if fill_val is not None:
                                     if np.isnan(fill_val): valid_data_mask &= ~np.isnan(src_data[0])
                                     else: valid_data_mask &= (src_data[0] != fill_val)

                            weights = calculate_weights(valid_data_mask, blend_distance)

                            for band_idx in range(common_count):
                                band_data = np.ma.getdata(src_data[band_idx])
                                valid_pixels_in_source = valid_data_mask & (weights > 0)
                                dest_buffer_slice[band_idx][valid_pixels_in_source] += (
                                    band_data[valid_pixels_in_source].astype(np.float64) * weights[valid_pixels_in_source])

                            weight_sum_buffer_slice += weights

                        except Exception as read_err:
                            warnings.warn(f"Could not read/process window from {os.path.basename(src.name)} for output window {window}: {read_err}")
                            continue

                    # Calculate final blended value and write
                    # (Logic remains the same, ensuring final cast to common_dtype)
                    valid_pixels = weight_sum_buffer_slice > 1e-6
                    fill_value_final = final_nodata_typed if final_nodata_typed is not None else 0
                    try: fill_value_final_typed = common_dtype.type(fill_value_final)
                    except: fill_value_final_typed = common_dtype.type(0)

                    final_data = np.full((common_count, window.height, window.width), fill_value_final_typed, dtype=common_dtype)

                    for band_idx in range(common_count):
                       band_buffer = dest_buffer_slice[band_idx]
                       averaged_values = (band_buffer[valid_pixels] / weight_sum_buffer_slice[valid_pixels])
                       final_data[band_idx][valid_pixels] = averaged_values.astype(common_dtype)

                    dst.write(final_data, window=window)
                    progress.update(1)
            progress.close()
        print("\nFinalizing output file...")

    # (Error handling and finally block remain the same)
    except (ValueError, rasterio.RasterioIOError, MemoryError) as e:
        print(f"\nERROR during mosaic processing: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\nClosing input datasets...")
        for src in sources:
            if src and not src.closed:
                try: # Add try-except for closing, in case a source is already invalid
                   src.close()
                except Exception as close_err:
                   warnings.warn(f"Error closing source {getattr(src, 'name', 'unknown')}: {close_err}")
        print("Input datasets closed.")

    total_time = time.time() - start_time
    print(f"\nTotal mosaic creation time: {total_time:.2f} seconds.")
    print(f"Smoothed mosaic saved successfully to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch multiple georeferenced TIFFs into a smooth mosaic using custom blending (robust dtype handling using src.dtypes[0]).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Arguments are the same as before ---
    parser.add_argument("input_dir", type=str, help="Directory containing input GeoTIFF files (.tif, .tiff).")
    parser.add_argument("output_file", type=str, help="Path for the output mosaic file (e.g., path/to/mosaic.tif).")
    parser.add_argument("-f", "--format", type=str, default="GTiff", help="Output file format driver (GTiff, PNG, etc.).")
    parser.add_argument("--blend_dist", metavar="PIXELS", type=int, default=100, help="Distance in pixels for feathering/blending overlap zones. Set to 0 to disable blending.")
    parser.add_argument("--nodata_val", type=str, default=None, help="Specify explicit nodata value override for output (provide as string). Use 'none' for no nodata. If unset, attempts auto-detection.")
    parser.add_argument("--block_size", metavar="PIXELS", type=int, default=512, help="Size of internal processing blocks (pixels). Larger may be faster but uses more RAM.")
    parser.add_argument("--resampling", type=str, default="nearest", choices=[r.name for r in Resampling], help="Resampling method used ONLY if reading requires reprojection/resizing.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output.")

    args = parser.parse_args()
    if not os.path.isdir(args.input_dir): print(f"ERROR: Input directory not found: {args.input_dir}"); sys.exit(1)
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        try: os.makedirs(output_dir); print(f"Created output directory: {output_dir}")
        except OSError as e: print(f"ERROR: Could not create output directory '{output_dir}'. Error: {e}"); sys.exit(1)

    nodata_arg = args.nodata_val

    create_smooth_mosaic(
        input_dir=args.input_dir,
        output_file=args.output_file,
        output_format=args.format,
        blend_distance=args.blend_dist,
        nodata_val=nodata_arg,
        resampling_method=args.resampling,
        block_size=args.block_size,
        verbose=args.verbose
    )