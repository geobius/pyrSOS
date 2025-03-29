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

def get_padding_offsets(image_height, image_width, patch_height, patch_width):
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
    bottom_padding = vertical_padding - top_padding
    left_padding = horizontal_padding // 2
    right_padding = horizontal_padding - left_padding

    return (top_padding, bottom_padding, left_padding, right_padding)

def pad_image(image, top_padding, bottom_padding, left_padding, right_padding):

    new_image = np.pad(image,
                       pad_width=((0, 0), (top_padding, bottom_padding), (left_padding, right_padding)),
                       mode='constant',
                       constant_values=0)

    return new_image


def retransform(initial_transformation, row_pixels_offset, column_pixels_offset): # by default moves the origin down and right
    translation = initial_transformation.translation(column_pixels_offset, row_pixels_offset)
    final_transformation = initial_transformation * translation

    return final_transformation


def generate_grid(padded_image_height, padded_image_width, patch_height, patch_width, n_subdivisions):
    row_indices = range(0, padded_image_height, patch_height // n_subdivisions)
    column_indices = range(0, padded_image_width, patch_width // n_subdivisions)
    g = list(product(row_indices, column_indices))
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
def name_patch(patch, source_raster_name, row, column, n_subdivisions):
    _, patch_height, patch_width = patch.shape

    area_of_interest, sen2_or_lma, pre_or_post, gsd_in_centimeters, multiband_or_label = source_raster_name.split('_')
    positivelabel_or_negativelabel_or_multiband = is_positive_or_negative_or_multiband(patch, multiband_or_label)
    row_index = (n_subdivisions * row) // patch_height
    column_index = (n_subdivisions * column) // patch_width
    
    name = f'{area_of_interest}_{sen2_or_lma}_{pre_or_post}_{gsd_in_centimeters}_{positivelabel_or_negativelabel_or_multiband}_{row_index}_{column_index}.tif'

    return name


def crop(image, at_row, at_column, row_offset, column_offset):
    patch = image[:, at_row: at_row + row_offset, at_column: at_column + column_offset]
    return patch


def export_1image(filepath, destination_folder, patch_height, patch_width, n_subdivisions):

    with rasterio.open(filepath) as hdd_image:
        source_raster_transformation = hdd_image.transform
        source_raster_name = hdd_image.name
        source_raster_height = hdd_image.height
        source_raster_width = hdd_image.width
        source_raster_bandcount = hdd_image.count
        source_raster_crs = hdd_image.crs
        source_raster_dtype = hdd_image.dtypes[0]

        ram_image = hdd_image.read()


    top_pad, bottom_pad, left_pad, right_pad = get_padding_offsets(source_raster_height, source_raster_width, patch_height, patch_width)
    padded_image = pad_image(ram_image, top_pad, bottom_pad, left_pad, right_pad)
    _, padded_image_height, padded_image_width = padded_image.shape

    crop_grid = generate_grid(padded_image_height, padded_image_width, patch_height, patch_width, n_subdivisions)
    padded_image_transformation = retransform(source_raster_transformation, -top_pad, -left_pad) #negative here because I want the origin to move up and left

    for row, column in tqdm(crop_grid):
        patch = crop(padded_image, row, column, patch_height, patch_width)
        name = name_patch(patch, filepath.stem, row, column, n_subdivisions)
        patch_transformation = retransform(padded_image_transformation, row, column) #positive here so the origin can move down and right
        patch_fullpath = destination_folder/name


        with rasterio.open(
                patch_fullpath,
                'w',
                driver='GTiff',
                width=patch_width,
                height=patch_height,
                count=source_raster_bandcount,
                dtype=source_raster_dtype,
                crs=source_raster_crs,
                transform=patch_transformation) as dataset:

            dataset.write(patch)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
    """This program splits raster images into equally sized patches.""")

    parser.add_argument('--dataset_path', type=Path, default='/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/pyrsos_250cm_dataset')
    parser.add_argument('--base_out_path', type=Path, default='/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination')
    parser.add_argument('--patch_width', type=int, default=128)
    parser.add_argument('--patch_height', type=int, default=128)
    parser.add_argument('--n_subdivisions', type=int, default=1, help= """modifies the step of the moving window.
    If the value is 1 then the window moves by patch_width horizontally. If the value is 2 then the window moves by patch_width//2 horizontally.
    Values bigger than 1 are useful as an augmentation mechanism and for keeping track of some pixels at the very edges of each patch.
    BEWARE. setting n_subdivisions to 2 quadruples the number of patches but the disk space required to store them is also quadrupled """)

    args = parser.parse_args()

    args.base_out_path.mkdir(parents=True, exist_ok=True)
    all_events = [folder.stem for folder in args.dataset_path.iterdir() if folder.is_dir()]
    for event in all_events:
        source_folder = args.dataset_path/event 
        destination_folder = args.base_out_path/event
        destination_folder.mkdir(parents=True, exist_ok=True)

        event_files = source_folder.glob('*.tif')
        for current_file in event_files:
            print(f'processing {current_file}')
            export_1image(current_file, destination_folder, args.patch_height, args.patch_width, args.n_subdivisions)
