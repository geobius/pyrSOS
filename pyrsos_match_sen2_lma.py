#!/usr/bin/env python3

from pathlib import Path
import rasterio as rio
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
import argparse

def open_sentinel2(sen2_pathname, sen2_gsd):
    sen2_ds = xr.Dataset()
    sen2_path = Path(sen2_pathname)
    relevant_files = sorted(list(sen2_path.glob(f'GRANULE/*/IMG_DATA/R{sen2_gsd}m/*B*.jp2')))
    #I only care about band names that start with B and I short them to ensure that names start from B1 to B12 in ascending order
    for img_file in relevant_files:
        band_name = img_file.stem.split('_')[-2] #this is consistent with the granule naming convention
        band = rxr.open_rasterio(img_file).squeeze() #since each image only has 1 band then I might as well squeeze it to reduce the dimensions of the array.
        sen2_ds[band_name] = band

    return sen2_ds


def rasterize_satellite_images(out_pathname, LMA_pathname, sen2_pathname,
                               source_sen2_gsd, interpolation):
    '''
    crops and reprojects a sentinel2 image to fit exactly within the boundaries of an lma image
    '''

    if interpolation == 'nearest':
        interp_method = Resampling.nearest
    elif interpolation == 'bilinear':
        interp_method = Resampling.bilinear
    else:
        interp_method = Resampling.cubic

    sen2_ds = open_sentinel2(sen2_pathname, source_sen2_gsd)
    LMA_ds = rxr.open_rasterio(LMA_pathname)

    sen2_crs = sen2_ds.rio.crs

    #get the lma bbox and reproject it to sen2 crs.
    #clip sen2 according to the bbox
    #reproject the clipped sen2 image to lma crs and interpolate

    LMA_bounds_reprojected = LMA_ds.rio.transform_bounds(sen2_crs)
    sen2_clipped = sen2_ds.rio.clip_box(*LMA_bounds_reprojected)

    sen2_ds_reprojected = sen2_clipped.rio.reproject_match(LMA_ds, resampling=interp_method)

    sen2_ds_reprojected.rio.to_raster(out_pathname)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Opens a sentinel2 .safe folder, reprojects it and crops it to overlap a specified LMA raster.\
    The arguments of the script follow this order\
    out_pathname: Where to store and how to name the output sentinel2 image\
    LMA_pathname: Where to find the LMA image\
    sen2_pathname: Where to find the .safe folder\
    --sen2_gsd: The ground sampling distance (as meters) of the source sen2 raster. Copernicus offers bands in resolutions of 60m,20m,10m. The script defaults to 20m\
    --interpolation: nearest,billinear and bicubic are provided. The script defaults to bicubic.')

    parser.add_argument('out_pathname')
    parser.add_argument('LMA_pathname')
    parser.add_argument('sen2_pathname')
    parser.add_argument('--sen2_gsd', type=int, default=20)
    parser.add_argument('--interpolation', default='bicubic')

    args = parser.parse_args()
    rasterize_satellite_images(args.out_pathname,
                               args.LMA_pathname,
                               args.sen2_pathname,
                               args.sen2_gsd,
                               args.interpolation)
