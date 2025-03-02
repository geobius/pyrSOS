#!/usr/bin/python3

from enum import unique
import numpy as np
import rasterio
from rasterio.transform import from_origin
import pickle
from pathlib import Path
from itertools import product, starmap
import gc
import argparse
import random
import pyjson5
from tqdm import tqdm

def split_dataset(base_out_path, optional_prefix, seed):
    all_events = [folder.stem for folder in base_out_path.iterdir() if folder.is_dir()]
    random.seed(seed)
    main_events = random.sample(all_events, k=3)

    test_events = [e for e in all_events if e not in main_events]
    validation_events = random.choices(main_events, k=1)
    training_events = [e for e in main_events if e not in validation_events]

    all_splits = {
        'training set': training_events,
        'validation set': validation_events,
        'testing set': test_events
    }
   
    json_split_name = f'event_splits_{optional_prefix}.json'
    pyjson5.dump(all_splits, open(base_out_path/json_split_name, 'wb'))

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
    """This program generates a json5 file denoting which areas will be used for training, validation and testing.""")

    parser.add_argument('--base_out_path', type=Path, default='/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination')
    parser.add_argument('--prefix', type=str, default='v1')
    parser.add_argument('--seed', type=int, default=29)
    args = parser.parse_args()

    split_dataset(args.base_out_path, args.prefix, args.seed)

