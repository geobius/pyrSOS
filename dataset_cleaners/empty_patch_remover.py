#!/usr/bin/python3
import numpy as np
from pathlib import Path
import argparse
import rasterio as rio
from tqdm import tqdm

def patch_is_mostly_black(patch_array):
    black_pixel_mask = np.all(patch_array==0, axis=0)
    num_black_pixels = np.sum(black_pixel_mask)
    total_pixels = patch_array.shape[1]*patch_array.shape[2]

    #0.95 because some patches may have some orphan pixels. I want to remove the orphan pixels as well
    criterion = num_black_pixels/total_pixels > 0.99
    return criterion


def partial_rename(image_name, which_components, new_values):
    naming_components = image_name.split('_')
    index_matching_table = {
        'area': 0,
        'platform': 1,
        'time': 2,
        'gsd': 3,
        'labelstatus': 4,
        'row_ID': 5,
        'column_ID': 6}

    indices = [index_matching_table[x] for x in which_components]
    for idx, val in zip(indices, new_values):
        naming_components[idx] = val
    new_name = '_'. join(naming_components)
    return new_name


def find_black_patches(area_folderpath):
    #first search for the black lma patches
    #add them to the blacklist
    #for all the items in the blacklist find their sen2 and mask equivalents by renaming
    #return the blacklist
    #delete everything in the blacklist in main

    blacklist = []
    lma_patch_paths = area_folderpath.glob('*lma*multiband*')
    for path in lma_patch_paths:
        with rio.open(path) as ds:
            patch_array = ds.read()
            if patch_is_mostly_black(patch_array):
                blacklist.append(path.name)

    #now you have found all the black lma patches
    #find the equivalent sen2pre sen2post mask patches
    sen2_pre_names = [partial_rename(lma_name, ['platform', 'time'], ['sen2', 'pre']) for lma_name in blacklist]
    sen2_post_names = [partial_rename(lma_name, ['platform', 'time'], ['sen2', 'post']) for lma_name in blacklist]

    #this is a bit awkard because some labels might be slightly positive but still useless
    label_names = []
    for name in blacklist:
        most_probable_label_name = partial_rename(name, ['labelstatus'], ['negativelabel'])
        if (area_folderpath/most_probable_label_name).exists():
            label_names.append(most_probable_label_name)
        else:
            edge_case = partial_rename(name, ['labelstatus'], ['positivelabel'])
            label_names.append(edge_case)

    blacklist.extend(sen2_pre_names)
    blacklist.extend(sen2_post_names)
    blacklist.extend(label_names)

    all_black_paths = [area_folderpath/x for x in blacklist]
    return all_black_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deletes patches that are 99% NODATA. It should take a few minutes.')
    parser.add_argument('dataset_folderpath', type=Path)

    args = parser.parse_args()

    areas = [x for x in args.dataset_folderpath.iterdir() if x.is_dir()]
    
    for area in areas:
        print(f'beginning patch removal from {area}')
        all_empty_paths = find_black_patches(area)
        for path in all_empty_paths:
            path.unlink()
