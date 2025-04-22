#!/usr/bin/python3

import numpy as np
import rasterio as rio
import argparse
from rasterio.coords import BoundingBox
from rasterio.windows import Window, from_bounds, intersection
from rasterio.enums import ColorInterp
from pathlib import Path

#this script should be used only on the south prodromos image.
#Raw prodromos north and Raw prodromos south have a common area
#I don't want the same area to appear in both images because It will create radiometric incosistency
#I have decided to remove that reduntant information from the south image

def intersection_bounds(bbox1, bbox2):
    bbox3 = BoundingBox(
        left=max(bbox1.left, bbox2.left),
        bottom=max(bbox1.bottom, bbox2.bottom),
        right=min(bbox1.right, bbox2.right),
        top=min(bbox1.top, bbox2.top))
    return bbox3


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('north_lma_raster_filepath', type=Path)
    parser.add_argument('south_lma_raster_filepath', type=Path)
    parser.add_argument('output_raster_filepath', type=Path)
    args = parser.parse_args()

    with rio.open(args.north_lma_raster_filepath) as north_ds, rio.open(args.south_lma_raster_filepath) as south_ds:
        intersection_bbox = intersection_bounds(north_ds.bounds, south_ds.bounds)

        north_intersection_window = from_bounds(*intersection_bbox, transform=north_ds.transform)
        north_overlapping_section = north_ds.read(window=north_intersection_window)
        south_image = south_ds.read()

        mask = np.all(north_overlapping_section == 0, axis=0).astype(np.uint8)
        mask = mask[np.newaxis, :, :]

        south_intersection_window = from_bounds(*intersection_bbox, transform=south_ds.transform)
        south_image[:, 0:round(south_intersection_window.height), 0:round(south_intersection_window.width)] *= mask

        south_crs = south_ds.crs
        south_transform = south_ds.transform
        south_dtype = south_ds.dtypes[0]
        count, height, width = south_image.shape

    with rio.open(args.output_raster_filepath,
                  'w',
                  driver='Gtiff',
                  width=width,
                  height=height,
                  count=count,
                  crs=south_crs,
                  transform=south_transform,
                  dtype=south_dtype) as out_ds:
        
        out_ds.colorinterp = (ColorInterp.undefined,
                              ColorInterp.undefined,
                              ColorInterp.undefined,
                              ColorInterp.undefined)
        
        out_ds.write(south_image)
