#!/usr/bin/python3
import rasterio
import numpy as np
import argparse

from rasterio import transform
from rasterio import dtypes
from collections import deque

WHITE = np.full(4, 255)
BLACK = np.full(4, 0)


def find_initial_row_column(bands, height, width):
    window_size = 11
    # Loop through the image with a sliding window
    for row in range(window_size, height-window_size):
        for column in range(window_size, width-window_size):
            window = bands[row-window_size:row+window_size, column-window_size:column+window_size]

            #If all the pixels in the window are white then return the row and column of the central pixel
            everything_is_white = np.all(window == WHITE)
            if everything_is_white:
                return (row, column)

    return "NO WHITE PIXELS FOUND"



def floodfill_algorithm(bands, image_height, image_width, initial_row, initial_column):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue = deque()
    queue.append((initial_row, initial_column))

    while queue:
        cx, cy = queue.popleft()
        # If the current pixel is within bounds and has the old color, we change it
        if 0 <= cx < image_width and 0 <= cy < image_height and np.array_equal(bands[cy][cx], WHITE):
            bands[cy][cx] = BLACK

            for dx, dy in directions:
                queue.append((cx + dx, cy + dy))

    return bands


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

        # I need to pad the image before passing it to the floodfill algorithm. Otherwise, it may miss some pixels in the borders.
        padded_image = np.pad(initial_bands,
                              pad_width=((1, 1), (1, 1), (0, 0)),
                              mode='constant',
                              constant_values=255)
        initial_row, initial_column = find_initial_row_column(padded_image, height+2, width+2)
        #height+2 because 1 down +1 up and
        #width+2 because 1 right+ 1 left
        intermediate1 = floodfill_algorithm(padded_image, height+2, width+2, initial_row, initial_column)
        intermediate2 = intermediate1[1:height+1, 1:width+1]#be careful off by 1 errors

        final = np.transpose(intermediate2, (2, 0, 1))
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
