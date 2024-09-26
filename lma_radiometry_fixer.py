#!/usr/bin/python3
import rasterio
import numpy as np
import argparse

from rasterio import transform
from rasterio import dtypes


WHITE = np.full(4, 255)
BLACK = np.full(4, 0)


def fix_radiometry_phase1(source_array, height, width):
    desired_window_size = 11 #I want it to be big enough as to not affect objects that actually reflect [255,255,255,255]
    destination_array = np.copy(source_array)
    # Loop through the image with a sliding window
    for row in range(0, height, desired_window_size):
        for column in range(0, width, desired_window_size):
            # Define the window (with boundary checking)
            win_height = min(desired_window_size, height - row)
            win_width = min(desired_window_size, width - column)
            window = source_array[row:row+win_height, column:column+win_width]

            # if ALL pixel values in this window are white then blackify them
            is_empty_area = np.all(window == WHITE)
            if is_empty_area:
                destination_array[row:row+win_height, column:column+win_width] = BLACK

    return destination_array


def fix_radiometry_phase2(intermediate_array, height, width):
    destination_array = np.copy(intermediate_array)
    # Loop through the image with a sliding window
    for row in range(1, height-1):
        for column in range(1, width-1):
            window = destination_array[row-1:row+1, column-1:column+1]
            central_pixel = destination_array[row, column]

            #If the current pixel is white and some other pixel next to it is black then it must become black as well
            at_least_one_neighbour_is_black = np.any(window == BLACK)
            pixel_is_white = np.array_equal(central_pixel, WHITE)
            if at_least_one_neighbour_is_black and pixel_is_white:
                destination_array[row, column] = BLACK

    return destination_array


#Phase 1, a window passes through the initial raster. If all values within the window are white make them black
#Some white pixels that border with actual data will escape the above test. That's why there's a need for phase 2
#In phase 2, If the current pixel is white and at least one of it's neighbours is black then set the current pixel as black.
#notice that the central operator in phase1 is 'all'. The central operator in phase2 is 'any'

if __name__ == '__main__':
    parser = argparse. ArgumentParser()
    parser.add_argument('source_name')
    parser.add_argument('destination_name')
    args = parser.parse_args()

    with rasterio.open(args.source_name) as initial_dataset:
        height = initial_dataset.height
        width = initial_dataset.width
        initial_bands = initial_dataset.read()

        #transpose to rows aka height,columns aka width, bands for a more typical convention
        initial_bands = np.transpose(initial_bands, (1, 2, 0))
        im = fix_radiometry_phase1(initial_bands, height, width)
        final = fix_radiometry_phase2(im, height, width)
        final = np.transpose(final, (2, 0, 1))
        # Transposed back to (bands, height, width). The way rasterio reads and writes data from/to files


        with rasterio.open(args.destination_name,
                           'w',
                           driver='GTiff',
                           height=height,
                           width=width,
                           crs=initial_dataset.crs,
                           transform=initial_dataset.transform,
                           count=initial_dataset.count,
                           dtype=initial_dataset.dtypes[0]) as destination_raster:

            destination_raster.write(final)
