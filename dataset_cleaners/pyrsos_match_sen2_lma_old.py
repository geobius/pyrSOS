#!/usr/bin/env python3

from pathlib import Path
import rasterio as rio
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
import argparse
import numpy as np

def open_sentinel2(sen2_path, sen2_gsd):
    band_names = []
    band_arrays = []
    relevant_files = sorted(list(sen2_path.glob(f'GRANULE/*/IMG_DATA/R{sen2_gsd}m/*B*.jp2')))
    #I only care about band names that start with B and I short them to ensure that names start from B1 to B12 in ascending order
    for img_file in relevant_files:
        current_band_name = img_file.stem.split('_')[-2] #this is consistent with the granule naming convention
        current_band_data = rxr.open_rasterio(img_file)
        band_names.append(current_band_name)
        band_arrays.append(current_band_data)

    sen2_ds = xr.concat(band_arrays, dim='band')
    sen2_ds['band'] = band_names

    return sen2_ds


def blackout(sen2_bands, lma_bands):
    mask = np.all(lma_bands == 0, axis=0)
    sen2_bands[:, mask] = 0
    return sen2_bands


def match_satellite_images(LMA_ds, sen2_ds, interp_method):
    #crops and reprojects a sentinel2 image to fit exactly within the boundaries of an lma image
    #get the lma bbox and reproject it to sen2 crs.
    #clip sen2 according to the bbox
    #reproject the clipped sen2 image to lma crs and interpolate
    sen2_crs = sen2_ds.rio.crs
    LMA_bounds_reprojected = LMA_ds.rio.transform_bounds(sen2_crs)
    sen2_clipped = sen2_ds.rio.clip_box(*LMA_bounds_reprojected)
    sen2_ds_reprojected = sen2_clipped.rio.reproject_match(LMA_ds, resampling=interp_method)

    return sen2_ds_reprojected



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Opens a sentinel2 .safe folder, reprojects it and crops it to overlap a specified LMA raster.\
    The arguments of the script follow this order\
    out_pathname: Where to store and how to name the output sentinel2 image\
    LMA_pathname: Where to find the LMA image\
    sen2_pathname: Where to find the .safe folder\
    --sen2_gsd: The ground sampling distance (as meters) of the source sen2 raster. Copernicus offers bands in resolutions of 60m,20m,10m. The script defaults to 20m\
    --interpolation: nearest,billinear and bicubic are provided. The script defaults to bicubic.')

    parser.add_argument('out_path', type=Path)
    parser.add_argument('LMA_path', type=Path)
    parser.add_argument('sen2_path', type=Path)
    parser.add_argument('--source_sen2_gsd', type=int, default=20)
    parser.add_argument('--interpolation', default='bicubic')
    args = parser.parse_args()

    table = {'nearest': Resampling.nearest,
             'bilinear': Resampling.bilinear,
             'bicubic': Resampling.cubic}

    sen2_ds = open_sentinel2(args.sen2_path, args.source_sen2_gsd)
    LMA_ds = rxr.open_rasterio(args.LMA_path)
    realligned_sen2_ds = match_satellite_images(LMA_ds, sen2_ds, table[args.interpolation])

    LMA_bands = LMA_ds.values
    realligned_sen2_bands = realligned_sen2_ds.values
    realligned_sen2_bands_with_black_pixels = blackout(realligned_sen2_bands, LMA_bands)

    final_sen2_ds = realligned_sen2_ds
    final_sen2_ds.rio.write_nodata(None, inplace=True)
    final_sen2_ds.values = realligned_sen2_bands_with_black_pixels
    final_sen2_ds.attrs['band names'] = realligned_sen2_ds.band

    final_sen2_ds.rio.to_raster(args.out_path)
