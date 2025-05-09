#!/usr/bin/python3

import json
import numpy as np
import rasterio as rio
import argparse
from pathlib import Path


def image2tabular(image_array):
    channels, height, width = image_array.shape
    transposed = np.transpose(image_array, (1, 2, 0))
    tabular_array = np.reshape(transposed, (height * width, channels))
    return tabular_array


def load_images_as_tabular_array(filepaths):
    tabular_chunks = []
    for p in filepaths:
        with rio.open(p) as ds:
            image_array = ds.read()
            tabular_chunks.append(image2tabular(image_array))

    tabular_array = np.concatenate(tabular_chunks, 0)
    return tabular_array


def extract_stats(tabular_array, source):
    min_values = np.min(tabular_array, axis=0).tolist()
    max_values = np.max(tabular_array, axis=0).tolist()
    mean_values = np.mean(tabular_array, axis=0).tolist()
    stdev_values = np.std(tabular_array, axis=0).tolist()
    normalizing_value = 255 if source == 'lma_post' else 10000

    stats_dictionary = {
        'global_minima': min_values,
        'global_maxima': max_values,
        'global_mean': mean_values,
        'global_stdev': stdev_values,
        'global_normalizing_value': normalizing_value}

    return stats_dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
                                     'dump the minimum,maximum,mean,standard deviation and normalizing constant values across each band of the training set into a json file')
    
    parser.add_argument('dataset_folder', type=Path)
    parser.add_argument('event_split_filepath', type=Path)
    args = parser.parse_args()

    stats_table = {}
    events_split = json.load(open(args.event_split_filepath, 'r'))
    train_events = events_split['training_set']
    train_folderpaths = [args.dataset_folder/x for x in train_events]

    train_filepaths = []
    for current_folder in train_folderpaths:
        train_filepaths.extend(list(current_folder.glob('*multiband.tif')))

    sources = ['lma_post', 'sen2_pre', 'sen2_post']
    for source in sources:
        source_filepaths = [path for path in train_filepaths if source in path.stem]
        source_tabular = load_images_as_tabular_array(source_filepaths)
        source_stats = extract_stats(source_tabular, source)

        new_entry = {source: source_stats}
        stats_table.update(new_entry)

    with open(args.dataset_folder/'stats_logger.json', 'w') as log:
        json.dump(stats_table, log)
