#!/usr/bin/env python3

from pathlib import Path
import rasterio as rio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds, calculate_default_transform
from rasterio.windows import from_bounds
import argparse
import numpy as np


def open_sentinel2(sen2_path, sen2_gsd):
    #make an adhoc dataset object without saving to disk. It is saved in RAM.
    #The contents of this dataset contain the concatenated images

    relevant_filenames = sorted(list(sen2_path.glob(f'GRANULE/*/IMG_DATA/R{sen2_gsd}m/*B*.jp2')))
    band_names = [x.stem.split('_')[-2] for x in relevant_filenames]
    source_datasets = [rio.open(x) for x in relevant_filenames]
    all_bands = np.concatenate([x.read() for x in source_datasets], axis=0)
    transform = source_datasets[0].transform
    crs = source_datasets[0].crs
    dtype = source_datasets[0].dtypes[0]

    count, height, width = all_bands.shape

    memfile = MemoryFile()
    dataset = memfile.open(driver='GTiff',
                           width=width,
                           height=height,
                           count=count,
                           crs=crs,
                           transform=transform,
                           dtype=dtype)

    dataset.write(all_bands)
    dataset.descriptions = band_names
    
    return dataset

def blackout(sen2_bands, lma_bands):
    mask = np.all(lma_bands == 0, axis=0)
    sen2_bands[:, mask] = 0
    return sen2_bands


def overlap_satellite_images(LMA_ds, sen2_ds, interp_method):
    #crops and reprojects a sentinel2 image to fit exactly within the boundaries of an lma image
    #get the lma bbox and reproject it to sen2 crs.
    #clip sen2 according to the bbox
    #reproject the clipped sen2 image to lma crs  and interpolate
    LMA_crs = LMA_ds.crs
    sen2_crs = sen2_ds.crs

    LMA_bounds = LMA_ds.bounds
    LMA_bounds_reprojected = transform_bounds(LMA_crs, sen2_crs, *LMA_ds.bounds)
    sen2_window = from_bounds(*LMA_bounds_reprojected, transform=sen2_ds.transform)
    
    sen2_clipped_bands = sen2_ds.read(indexes=None, window=sen2_window)
    sen2_clipped_transform = sen2_ds.window_transform(sen2_window)
    
    dummy_bands = np.zeros((sen2_ds.count, LMA_ds.height, LMA_ds.width))
    sen2_reprojected_bands, sen2_reprojected_transform = reproject(source=sen2_clipped_bands,
                                                                   destination=dummy_bands,
                                                                   src_transform=sen2_clipped_transform,
                                                                   dst_transform=LMA_ds.transform,
                                                                   src_crs=sen2_crs,
                                                                   dst_crs=LMA_ds.crs,
                                                                   resampling=interp_method)
    return sen2_reprojected_bands, sen2_reprojected_transform


table = {'nearest': Resampling.nearest,
         'bilinear': Resampling.bilinear,
         'bicubic': Resampling.cubic}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Opens a sentinel2 .safe folder, reprojects it and crops it to overlap a specified LMA raster.\
    The arguments of the script follow this order\
    out_pathname: Where to store and how to name the output sentinel2 image\
    LMA_pathname: Where to find the LMA image\
    sen2_pathname: Where to find the .safe folder\
    --sen2_gsd: The ground sampling distance (as meters) of the source sen2 raster. Copernicus offers bands in resolutions of 60m,20m,10m. The script defaults to 20m\
    --interpolation: nearest,billinear and bicubic are provided. The script defaults to bicubic.')

    parser.add_argument('LMA_path', type=Path)
    parser.add_argument('sen2_path', type=Path)
    parser.add_argument('out_path', type=Path)
    parser.add_argument('--source_sen2_gsd', type=int, default=20)
    parser.add_argument('--interpolation', default='bicubic')
    args = parser.parse_args()


    sen2_ds = open_sentinel2(args.sen2_path, args.source_sen2_gsd)
    band_descriptions = sen2_ds.descriptions
    LMA_ds = rio.open(args.LMA_path)
    
    realligned_sen2_bands, realligned_sen2_transform = overlap_satellite_images(LMA_ds, sen2_ds, table[args.interpolation])
    LMA_bands = LMA_ds.read()
    realligned_sen2_bands_with_black_pixels = blackout(realligned_sen2_bands, LMA_bands)

    count, height, width = realligned_sen2_bands_with_black_pixels.shape
    
    with rio.open(args.out_path,
                  'w',
                  driver='GTiff',
                  width=width,
                  height=height,
                  count=count,
                  crs=LMA_ds.crs,
                  transform=realligned_sen2_transform,
                  dtype=sen2_ds.dtypes[0]) as final_ds:
        
        final_ds.write(realligned_sen2_bands_with_black_pixels)
        final_ds.descriptions = band_descriptions
        
        
