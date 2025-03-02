#!/usr/bin/python3

import json
import numpy as np
import rasterio as rio
import argparse
from pathlib import Path

#open an image
#read the min,max, mean, stdev, normalizing constant
#append the results in a file


def extract_stats(image_array):
    min_values = np.min(image_array, axis=(1, 2)).tolist()
    max_values = np.max(image_array, axis=(1, 2)).tolist()
    mean_values = np.mean(image_array, axis=(1, 2)).tolist()
    stdev_values = np.std(image_array, axis=(1, 2)).tolist()
    normalizing_value = 255 if image_array.dtype == np.uint8 else 10000

    stats_dictionary = {
        'global_minima': min_values,
        'global_maxima': max_values,
        'global_mean': mean_values,
        'global_stdev': stdev_values,
        'global_normalizing_value': normalizing_value}

    return stats_dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'dump the minimum,maximum,mean,standard deviation and normalizing constant values across each band of a raster into a json file')
    
    parser.add_argument('dataset_folder', type=Path)
    args = parser.parse_args()

    stats_table = {}

    all_events = [folder.stem for folder in args.dataset_folder.iterdir() if folder.is_dir()]
    for event in all_events:
        source_folder = args.dataset_folder/event
        event_files = source_folder.glob('*.tif')
        for current_file in event_files:
            with rio.open(current_file) as ds:
                image_name = ds.name
                bands = ds.read()
                stats = extract_stats(bands)
                new_entry = {image_name: stats}
                stats_table.update(new_entry)


    with open(args.dataset_folder/'stats_logger.json', 'w') as log:
        json.dump(stats_table, log)
