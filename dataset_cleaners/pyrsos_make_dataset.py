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

def get_padding_offset(image_width, image_height, patch_width, patch_height):
    '''
    top and bottom paddings must be equal or differ by 1 to accomplish symmetry in the vertical axis.
    Same rule applies for left and right paddings in the horizontal axis.
    '''
    horizontal_padding = patch_width - image_width % patch_width
    vertical_padding = patch_height - image_height % patch_height

    # if the batch size is bigger than the image then horizontal padding is reduced to
    # batch_width - image_width. As a consequence of modulo arithmetic
    # same applies to vertical padding

    top_padding = vertical_padding // 2
    bottom_padding = vertical_padding-top_padding
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding

    return (top_padding, bottom_padding, left_padding, right_padding)

def pad_image(image, top_padding, bottom_padding, left_padding, right_padding):

    new_image = np.pad(image,
                       pad_width=((0, 0), (top_padding, bottom_padding), (left_padding, right_padding)),
                       mode='constant',
                       constant_values=0)

    return new_image


def retransform(initial_transformation, column_pixels_offset, row_pixels_offset): # by default moves the origin down and right
    translation = initial_transformation.translation(column_pixels_offset, row_pixels_offset)
    final_transformation = initial_transformation * translation

    return final_transformation


def generate_grid(image_width, image_height, patch_width, patch_height, top_padding, bottom_padding, left_padding, right_padding):
    x_indices = range(0, left_padding+image_width+right_padding, patch_width)
    y_indices = range(0, top_padding+image_height+bottom_padding, patch_height)

    g = list(product(y_indices, x_indices))
    return g



def is_positive_or_negative_or_multiband(patch, is_source_multiband_or_label):
    #If a label is positive then the patch contains some burned areas
    #If a lable is negative then the patch contains no burned areas
    uniqs = np.unique(patch)

    if is_source_multiband_or_label == 'multiband':
        extension = 'multiband'
    elif (is_source_multiband_or_label == 'label') and (1 in uniqs):
        extension = 'positivelabel'
    else:
        extension = 'negativelabel'

    return extension


#this function defines the naming convention of all kinds of patches
def name_patch(patch, source_raster_name, current_patch_offset):

    area_of_interest, sen2_or_lma, pre_or_post, gsd_in_centimeters, multiband_or_label = source_raster_name.stem.split('_')
    positivelabel_or_negativelabel_or_multiband = is_positive_or_negative_or_multiband(patch, multiband_or_label)
    name = f'{area_of_interest}_{sen2_or_lma}_{pre_or_post}_{gsd_in_centimeters}_{positivelabel_or_negativelabel_or_multiband}_{current_patch_offset}.tif'

    return name


def crop(image, i, j, patch_width, patch_height):
    patch = image[:, i: i + patch_width, j: j + patch_height]
    return patch


def possibly_extend_nodata_patches_list(patch, patch_name, list_of_nodata_patchnames):
    if ('multiband' in patch_name) and np.all(patch == 0):
        extended_list = np.append(list_of_nodata_patchnames, patch_name)
    else:
        extended_list = list_of_nodata_patchnames

    return extended_list


def export_patches(source_folder, base_out_path, patch_width=128, patch_height=128):
    #all patches are exported as tif files
    zero_patches = []
    all_files = source_folder.glob('*.tif')

    for current_file in all_files:
        hdd_image = rasterio.open(source_folder/current_file)
        initial_transformation = hdd_image.transform

        source_raster_name = hdd_image.name
        top_pad, bottom_pad, left_pad, right_pad = get_padding_offset(hdd_image.width, hdd_image.height, patch_width, patch_height)
        crop_grid = generate_grid(hdd_image.width, hdd_image.height, patch_width, patch_height, top_pad, bottom_pad, left_pad, right_pad)
        number_of_patches = len(crop_grid)

        ram_image = hdd_image.read()
        padded_image = pad_image(ram_image, top_pad, bottom_pad, left_pad, right_pad)
        padded_image_transformation = retransform(initial_transformation, -left_pad,-top_pad) #negative here because I want the origin to move up and left
        patch_key = 0


        for i, j in crop_grid:
            patch = crop(padded_image, i, j, patch_width, patch_height)
            name = name_patch(patch, current_file, patch_key)
            zero_patches = possibly_extend_nodata_patches_list(patch, name, zero_patches)
            patch_transformation = retransform(padded_image_transformation, j, i) #positive here so the origin can move down and right
            patch_fullpath = base_out_path/'patches'/name

            patch_key += 1
            dataset = rasterio.open(
                patch_fullpath,
                'w',
                driver='GTiff',
                width=patch_width,
                height=patch_height,
                count=hdd_image.count,
                dtype=hdd_image.dtypes[0],
                crs=hdd_image.crs,
                transform=patch_transformation)

            dataset.write(patch)
            dataset.close()

            print(f'written {name} , {patch_key} out of {number_of_patches}')

        hdd_image.close()
        gc.collect()

    return zero_patches

def remove_empty_patches_and_their_labels(empty_patch_names, base_out_path):
    search_folder = base_out_path/'patches'
    for multiband_patch_name in empty_patch_names:
        area_of_interest, _, _, _, _, ID = Path(multiband_patch_name).stem.split('_')

        pattern = f'{area_of_interest}_*_{ID}.tif'
        for p in search_folder.glob(pattern):
            p.unlink()
            print(f'removed {p} as empty area\n')
    return

def find_paths_of_overlaying_patches(area, ID, search_path):
    lma = next(search_path.glob(f'{area}_lma_*_multiband_{ID}.tif'))
    sen2_pre = next(search_path.glob(f'{area}_sen2_pre_*_{ID}.tif'))
    sen2_post = next(search_path.glob(f'{area}_sen2_post_*_{ID}.tif'))
    label = next(search_path.glob(f'{area}_lma_*label_{ID}.tif'))

    dictionary = {
        'lma': lma,
        'sen2_pre': sen2_pre,
        'sen2_post': sen2_post,
        'label': label,
    }

    return dictionary

def unique_IDs(source_area_name, search_folder):
    # Remember. Patches, whose names have the same area and ID, overlap.
    # Therefore every label patch has a corresponding sen2pre,sen2post,lma patch with which they share the same ID
    # Everytime, there are 4 patches that share the same ID
    # That's why I define the glob pattern this way
    label_paths = list(search_folder.glob(f'{source_area_name}*label_*'))
    label_stems = map(lambda s: s.stem, label_paths)
    label_IDs = list(map(lambda s: s.split('_')[-1], label_stems))

    return label_IDs

def all_patches_by_area(source_area_name, search_folder):
    all_possible_IDs = unique_IDs(source_area_name, search_folder)
    all_patches = []
    for ID in all_possible_IDs:
        all_patches.append(find_paths_of_overlaying_patches(source_area_name, ID, search_folder))

    return all_patches

def split_events(all_events, seed):
    random.seed(seed)
    main_events = random.sample(all_events, k=3)
    test_events = [e for e in all_events if e not in main_events]
    validation_events = random.choices(main_events, k=1)
    training_events = [e for e in main_events if e not in validation_events]
    return (training_events, validation_events, test_events)


def split_dataset(base_out_path, optional_prefix, seed):
    all_events = ['delphoi', 'domokos', 'prodromos', 'yliki']
    training_events, validation_events, test_events = split_events(all_events, seed)

    training_set = {}
    validation_set = {}
    testing_set = {}

    for e in training_events:
        training_set.update({e: all_patches_by_area(e, base_out_path/'patches')})
    for e in validation_events:
        validation_set.update({e: all_patches_by_area(e, base_out_path/'patches')})
    for e in test_events:
        testing_set.update({e: all_patches_by_area(e, base_out_path/'patches')})

    train_pickle_name = f'TrainEvents_{optional_prefix}.pkl'
    val_pickle_name = f'ValidationEvents_{optional_prefix}.pkl'
    test_pickle_name = f'TestingEvents_{optional_prefix}.pkl'

    pickle.dump(training_set, open(base_out_path/train_pickle_name, 'wb'))
    pickle.dump(validation_set, open(base_out_path/val_pickle_name, 'wb'))
    pickle.dump(testing_set, open(base_out_path/test_pickle_name, 'wb'))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program works in 3 modes.\
    In mode 1: All raster images are split into equally sized patches.\
    In mode 2: Three pickle files are created, each containing mutually exclusive paths to the patches\
    and representing different stages in the learning pipeline.\
    Mode 3: It is mode 1 followed by mode 2. This is the default mode')

    parser.add_argument('--dataset_path', type=Path, default='/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/pyrsos_250cm_dataset')
    parser.add_argument('--base_out_path', type=Path, default='/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination')
    parser.add_argument('--patch_width', type=int, default=128)
    parser.add_argument('--patch_height', type=int, default=128)
    parser.add_argument('--prefix', type=str, default='v1')
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--mode', type=int, default=3)
    parser.add_argument('--train_percentage', type=int, default=60)
    parser.add_argument('--test_percentage', type=int, default=20)
    args = parser.parse_args()

    if args.mode == 1:
        args.base_out_path.mkdir(parents=True, exist_ok=True)
        patches_path = args.base_out_path/ 'patches'
        patches_path.mkdir(parents=True, exist_ok=True)

        empty_multiband_patches = export_patches(args.dataset_path,
                                                 args.base_out_path,
                                                 args.patch_width,
                                                 args.patch_height)

        remove_empty_patches_and_their_labels(empty_multiband_patches, args.base_out_path)

    if args.mode == 2:
        split_dataset(args.base_out_path, args.prefix, args.seed)

    if args.mode == 3:
        args.base_out_path.mkdir(parents=True, exist_ok=True)
        patches_path = args.base_out_path/ 'patches'
        patches_path.mkdir(parents=True, exist_ok=True)

        empty_multiband_patches = export_patches(args.dataset_path,
                                                 args.base_out_path,
                                                 args.patch_width,
                                                 args.patch_height)

        remove_empty_patches_and_their_labels(empty_multiband_patches, args.base_out_path)
        split_dataset(args.base_out_path, args.prefix, args.seed)
