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

def get_padding_offset(image_height, image_width, patch_height, patch_width):
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


def retransform(initial_transformation, row_pixels_offset, column_pixels_offset): # by default moves the origin down and right
    translation = initial_transformation.translation(column_pixels_offset, row_pixels_offset)
    final_transformation = initial_transformation * translation

    return final_transformation


def generate_grid(image_height, image_width, patch_height, patch_width, top_padding, bottom_padding, left_padding, right_padding):
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
def name_patch(patch, source_raster_name, i, j):
    patch_height = patch.shape[1]
    patch_width = patch.shape[2]

    area_of_interest, sen2_or_lma, pre_or_post, gsd_in_centimeters, multiband_or_label = source_raster_name.split('_')
    positivelabel_or_negativelabel_or_multiband = is_positive_or_negative_or_multiband(patch, multiband_or_label)
    name = f'{area_of_interest}_{sen2_or_lma}_{pre_or_post}_{gsd_in_centimeters}_{positivelabel_or_negativelabel_or_multiband}_{i // patch_height}_{j // patch_width}.tif'

    return name


def crop(image, i, j, patch_height, patch_width):
    patch = image[:, i: i + patch_height, j: j + patch_width]
    return patch


def export_1image(filepath, destination_folder, patch_height, patch_width):

    with rasterio.open(filepath) as hdd_image:
        source_raster_transformation = hdd_image.transform
        source_raster_name = hdd_image.name
        source_raster_height = hdd_image.height
        source_raster_width = hdd_image.width
        source_raster_bandcount = hdd_image.count
        source_raster_crs = hdd_image.crs
        source_raster_dtype = hdd_image.dtypes[0]

        ram_image = hdd_image.read()


    top_pad, bottom_pad, left_pad, right_pad = get_padding_offset(source_raster_height, source_raster_width, patch_height, patch_width)
    crop_grid = generate_grid(source_raster_height, source_raster_width, patch_height, patch_width, top_pad, bottom_pad, left_pad, right_pad)

    padded_image = pad_image(ram_image, top_pad, bottom_pad, left_pad, right_pad)
    padded_image_transformation = retransform(source_raster_transformation, -top_pad, -left_pad) #negative here because I want the origin to move up and left

    for i, j in tqdm(crop_grid):
        patch = crop(padded_image, i, j, patch_height, patch_width)
        name = name_patch(patch, filepath.stem, i, j)
        patch_transformation = retransform(padded_image_transformation, i, j) #positive here so the origin can move down and right
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
            export_1image(current_file, destination_folder, args.patch_height, args.patch_width)

            
