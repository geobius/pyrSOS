#!/usr/bin/python3

from pathlib import Path
import rasterio as rio
import argparse
import numpy as np

#This is to be used only for prodromos1 and prodromos2 from raw_data_alt. Don't use this for prodromos from raw_data.
#The purpose of this script is to set all values of the mask to zero wherever total black is in the source lma raster.
#This is necessary because the original vector was drawn on the fused area

def blackout(mask_bands, lma_bands):
    mask = np.all(lma_bands == 0, axis=0)
    mask_bands[:, mask] = 0
    return mask_bands


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Opens a rasterized mask and sets all pixels to zero wherever the LMA raster is total black.\
    The arguments of the script follow this order\
    rastermask_filepath: Where to find the rasterized mask\
    lma_filepath: Where to find the LMA image\
    out_filepath: Where to store the corrected mask\
')

    parser.add_argument('rastermask_filepath', type=Path)
    parser.add_argument('lma_filepath', type=Path)
    parser.add_argument('out_filepath', type=Path)
    args = parser.parse_args()

    with rio.open(args.rastermask_filepath) as mask_ds:
        mask_bands = mask_ds.read()
        mask_crs = mask_ds.crs
        mask_transform = mask_ds.transform

    with rio.open(args.lma_filepath) as lma_ds:
        lma_bands = lma_ds.read()

    corrected_mask = blackout(mask_bands, lma_bands)

    count, height, width = corrected_mask.shape

    with rio.open(args.out_filepath,
                  'w',
                  driver='GTiff',
                  width=width,
                  height=height,
                  count=count,
                  crs=mask_crs,
                  transform=mask_transform,
                  dtype=np.uint8) as final_ds:

        final_ds.write(corrected_mask)
