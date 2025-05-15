import os
import glob
import tempfile
import rasterio
from rasterio.merge import merge
from rasterio.errors import RasterioIOError
from tqdm import tqdm
import logging
import numpy as np

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def is_valid_raster(src):
    try:
        if src.width <= 0 or src.height <= 0:
            return False
        if src.transform.a == 0 or src.transform.e == 0:
            return False
        if np.any(np.isnan(src.read(1, out_shape=(1, 1)))):  # quick check
            return False
        return True
    except Exception as e:
        logging.warning(f"Validation failed: {e}")
        return False

def custom_blend_merge(sources):
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling

    # Determine mosaic bounds and size
    mosaic, transform = merge(sources, method='first')
    count, height, width = mosaic.shape

    blend_mosaic = np.zeros((count, height, width), dtype=np.float32)
    weight_mask = np.zeros((height, width), dtype=np.float32)

    for src in sources:
        with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
            data = vrt.read(out_shape=(count, height, width), resampling=Resampling.bilinear)
            mask = vrt.read_masks(1, out_shape=(height, width), resampling=Resampling.nearest).astype(bool)

            for b in range(count):
                band_data = data[b].astype(np.float32)
                band_data[~mask] = 0
                blend_mosaic[b] += band_data

            weight_mask += mask.astype(np.float32)

    weight_mask[weight_mask == 0] = 1
    for b in range(count):
        blend_mosaic[b] /= weight_mask

    return blend_mosaic.astype(np.float32), transform

def stitch_geotiffs_incremental(input_folder, output_path, output_crs=None, keep_temp=False):
    """
    Stitch all GeoTIFF files in a folder into one mosaic GeoTIFF,
    using custom blending and managing memory with temp files.
    """
    tiff_files = glob.glob(os.path.join(input_folder, '*.tif'))
    if not tiff_files:
        raise FileNotFoundError("No .tif files found in the input directory.")

    tiff_files.sort()
    valid_sources = []

    logging.info(f"Found {len(tiff_files)} files. Opening and validating...")
    for fp in tqdm(tiff_files):
        try:
            src = rasterio.open(fp)
            if not is_valid_raster(src):
                logging.warning(f"Skipping invalid or corrupt raster: {fp}")
                continue
            if output_crs and src.crs != output_crs:
                logging.info(f"Reprojection is not supported in this script. Skipping: {fp}")
                continue
            valid_sources.append(src)
        except RasterioIOError as e:
            logging.warning(f"Could not open {fp}: {e}")

    if not valid_sources:
        raise RuntimeError("No valid raster files to process.")

    logging.info("Blending all rasters using custom blending algorithm...")

    try:
        mosaic, transform = custom_blend_merge(valid_sources)
    except Exception as e:
        raise RuntimeError(f"Custom blending failed: {e}")

    meta = valid_sources[0].meta.copy()
    meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "count": mosaic.shape[0],
        "dtype": 'uint16'
    })

    # Scale to 16-bit unsigned integer for storage efficiency
    mosaic = np.clip(mosaic, 0, 65535)
    mosaic = mosaic.astype(np.uint16)

    temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    logging.info(f"Writing mosaic to temporary file...")
    try:
        with rasterio.open(temp_file.name, 'w', **meta) as dst:
            dst.write(mosaic)

        with rasterio.open(temp_file.name) as final_src:
            final_data = final_src.read()
            final_meta = final_src.meta

        with rasterio.open(output_path, "w", **final_meta) as dest:
            dest.write(final_data)
    except Exception as e:
        raise RuntimeError(f"Failed to write output: {e}")
    finally:
        for src in valid_sources:
            src.close()
        if not keep_temp:
            os.remove(temp_file.name)
        else:
            logging.info(f"Intermediate file retained at: {temp_file.name}")

    logging.info("Mosaic written successfully.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Stitch GeoTIFFs into a mosaic with custom blending.")
    parser.add_argument('--input_folder', required=True, help='Folder with input .tif files')
    parser.add_argument('--output', required=True, help='Path to output mosaic GeoTIFF')
    parser.add_argument('--crs', default=None, help='Optional target CRS (e.g., EPSG:4326)')
    parser.add_argument('--keep_temp', action='store_true', help='Keep intermediate temp files for debugging')
    args = parser.parse_args()

    stitch_geotiffs_incremental(args.input_folder, args.output, args.crs, args.keep_temp)
