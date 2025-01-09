#!/usr/bin/python3
import geopandas as gpd
from pyproj import transform
import rasterio as rio
from rasterio.features import rasterize
import argparse
from pathlib import Path

def rasterize_mask(source_raster_path, geopackage_path, output_path):
    vector_features = gpd.read_file(geopackage_path)
    geometries = vector_features.geometry

    with rio.open(source_raster_path) as source_ds:
        image_transform = source_ds.transform
        image_crs = source_ds.crs
        image_shape = source_ds.shape

    mask = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=image_shape,
        transform=image_transform,
        fill=0,
        all_touched=False,
        dtype='uint8'
    )

    with rio.open(output_path,
                  'w',
                  driver='GTiff',
                  height=image_shape[0],
                  width=image_shape[1],
                  count=1,
                  crs=image_crs,
                  transform=image_transform,
                  dtype='uint8') as destination_ds:
        destination_ds.write(mask, 1)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     """
                                     Input the names of the source raster and vector mask to generate
                                     a new raster mask with the desired name.
                                     """
                                     )

    parser.add_argument('source_raster_path', type=Path)
    parser.add_argument('geopackage_path', type=Path)
    parser.add_argument('destination_raster_path', type=Path)
   
    args = parser.parse_args()
    rasterize_mask(args.source_raster_path,
                   args.geopackage_path,
                   args.destination_raster_path)
