#!/usr/bin/python3
import rasterio
import numpy as np
import argparse

from rasterio import Affine
from rasterio import transform
from rasterio import dtypes
from rasterio.warp import calculate_default_transform, reproject
from collections import deque

WHITE = np.full(4, 255)
BLACK = np.full(4, 0)


def resample(source_dataset, target_resolution):
    crs = source_dataset.crs
    new_transform, new_width, new_height = calculate_default_transform(crs, crs,
                                                                       width=source_dataset.width,
                                                                       height=source_dataset.height,
                                                                       resolution=target_resolution,
                                                                       left=source_dataset.bounds.left,
                                                                       bottom=source_dataset.bounds.bottom,
                                                                       right=source_dataset.bounds.right,
                                                                       top=source_dataset.bounds.top
                                                                       )
    
    
    bands = source_dataset.read()
    dummy_bands = np.zeros((source_dataset.count, new_height, new_width))
    
    resampled_bands, new_transformation = reproject(bands, dummy_bands,
                                                    src_transform=source_dataset.transform,
                                                    src_crs=source_dataset.crs,
                                                    dst_transform=new_transform,
                                                    dst_crs=source_dataset.crs,
                                                    dst_resolution=target_resolution)
    
    return resampled_bands.astype(np.uint8), new_transformation


def find_initial_row_column(bands):
    _, height, width = bands.shape
    window_size = 11
    # Loop through the image with a sliding window. The window is anchored at topleft.
    for row in range(height-window_size):
        for column in range(width-window_size):
            window = bands[:, row:row+window_size, column:column+window_size]
            #If all the pixels in the window are white then return the topleft pixel
            comparison_matrix = np.all(window == WHITE.reshape((4, 1, 1)), axis=0)
            everything_is_white = np.all(comparison_matrix)
            if everything_is_white:
                return (row, column)

    return "NO WHITE PIXELS FOUND"


def floodfill_algorithm(bands, initial_row, initial_column):
    _, image_height, image_width = bands.shape

    white_mask = np.all(bands == WHITE.reshape((4,1,1)), axis=0)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)] #(vertical, horizontal) pairs of offsets
    queue = deque([(initial_row, initial_column)])
    #remember rows are vertical columns are horizontal. coordinate convention is first down then right

    bands[:, initial_row, initial_column] = BLACK
    while queue:
        row, column = queue.popleft()
        for down_offset, right_offset in directions:
            new_row = row + down_offset
            new_column = column + right_offset

            is_pixel_within_bounds = 0 <= new_row < image_height and 0 <= new_column < image_width

            if is_pixel_within_bounds and white_mask[new_row, new_column]:
                bands[:, new_row, new_column] = BLACK
                white_mask[new_row, new_column] = False
                queue.append((new_row, new_column))

    return bands


if __name__ == '__main__':
    parser = argparse. ArgumentParser()
    parser.add_argument('source_name')
    parser.add_argument('destination_name')
    parser.add_argument('resolution', type=float)
    args = parser.parse_args()
    
    with rasterio.open(args.source_name) as initial_dataset:
        crs = initial_dataset.crs
        resampled_bands, new_transformation = resample(initial_dataset, args.resolution)
        # I need to pad the image before passing it to the floodfill algorithm. Otherwise, it may miss some pixels in the borders.
    padded_image = np.pad(resampled_bands,
                          pad_width=((0, 0), (1, 1), (1, 1)),
                          mode='constant',
                          constant_values=255)
    del resampled_bands #to free up some space
    
    seed_row, seed_column = find_initial_row_column(padded_image)
    intermediate = floodfill_algorithm(padded_image, seed_row, seed_column)
    final = intermediate[:, 1:-1, 1:-1] #I must delete the paddings. They served their purpose.
    del intermediate # to free up some space

    with rasterio.open(args.destination_name,
                       'w',
                       driver='GTiff',
                       count=final.shape[0],
                       height=final.shape[1],
                       width=final.shape[2],
                       crs=crs,
                       transform=new_transformation,
                       dtype='uint8') as destination_raster:

        destination_raster.colorinterp = (rasterio.enums.ColorInterp.undefined,
                                          rasterio.enums.ColorInterp.undefined,
                                          rasterio.enums.ColorInterp.undefined,
                                          rasterio.enums.ColorInterp.undefined)
        destination_raster.write(final)
