import numpy as np
import rasterio
from rasterio.transform import from_origin


from pathlib import Path
from itertools import product
import os
import re
import gc

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

def pad_image(image, sen2_or_lma, top_padding, bottom_padding, left_padding, right_padding):
    if sen2_or_lma == 'lma':
        pad_constant = 1
    elif sen2_or_lma == 'sen2':
        pad_constant == 0

    new_image = np.pad(image,
                       pad_width=((0, 0), (top_padding, bottom_padding), (left_padding, right_padding)),
                       mode='constant',
                       constant_values=pad_constant)

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

    area_of_interest, sen2_or_lma, pre_or_post, gsd_in_meters, multiband_or_label = re.split('_', source_raster_name)[:-1] #I don't want the .tif part
    positivelabel_or_negativelabel_or_multiband = is_positive_or_negative_or_multiband(patch, multiband_or_label)
    name = f'{area_of_interest}_{sen2_or_lma}_{pre_or_post}_{gsd_in_meters}_{positivelabel_or_negativelabel_or_multiband}_{current_patch_offset}_.tif'

    return name


def crop(image, i, j, patch_width, patch_height):
    patch = image[:, i: i + patch_width, j: j + patch_height]
    return patch


def possibly_extend_nodata_patches_list(patch, patch_name, list_of_nodata_patchnames):

    if 'sen2' in patch_name:
        no_data_value = 0
    if 'lma' in patch_name:
        no_data_value = 1

    if ('multiband' in patch_name) and np.all(patch == no_data_value):
        extended_list = np.append(list_of_nodata_patchnames, patch_name)
    else:
        extended_list = list_of_nodata_patchnames

    return extended_list


def export_patches(source_folder, base_out_path, patch_width=128, patch_height=128):
    #all patches are exported as tif files
    zero_patches = []
    all_files = os.listdir(source_folder)

    for current_file in all_files:
        hdd_image = rasterio.open(os.path.join(source_folder, current_file))
        initial_transformation = hdd_image.transform

        source_raster_name = hdd_image.name

        if 'sen2' in source_raster_name:
            sen2_or_lma = 'sen2'

        if 'lma' in source_raster_name:
            sen2_or_lma = 'lma'


        top_pad, bottom_pad, left_pad, right_pad = get_padding_offset(hdd_image.width, hdd_image.height, patch_width, patch_height)
        crop_grid = generate_grid(hdd_image.width, hdd_image.height, patch_width, patch_height, top_pad, bottom_pad, left_pad, right_pad)
        number_of_patches = len(crop_grid)

        ram_image = hdd_image.read()
        padded_image = pad_image(ram_image, sen2_or_lma, top_pad, bottom_pad, left_pad, right_pad)
        padded_image_transformation = retransform(initial_transformation, -left_pad,-top_pad) #negative here because I want the origin to move up and left
        patch_key = 0


        for i, j in crop_grid:
            patch = crop(padded_image, i, j, patch_width, patch_height)
            name = name_patch(patch, current_file, patch_key)
            zero_patches = possibly_extend_nodata_patches_list(patch, name, zero_patches)
            patch_transformation = retransform(padded_image_transformation, j, i) #positive here so the origin can move down and right
            patch_fullpath = os.path.join(base_out_path, 'patches', name)

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
    for multiband_patch_name in empty_patch_names:
        search_folder = Path(base_out_path)/'patches'
        area_of_interest, sen2_or_lma, pre_or_post, gsd_in_meters, _, ID = re.split('_', multiband_patch_name)[:-1] # I don't want the .tif part
        label_patch_pattern = f'{area_of_interest}_{sen2_or_lma}_{pre_or_post}_{gsd_in_meters}_*label_{ID}_.tif'
        label_patch_name = list(search_folder.glob(label_patch_pattern))[0]
       
        multiband_fullpath = search_folder/multiband_patch_name
        label_fullpath = search_folder/label_patch_name

        multiband_fullpath.unlink()
        label_fullpath.unlink()
        print(f'removed {multiband_patch_name} and {label_patch_name} as empty areas')
    return


base_out_path = '/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination'


empty_multiband_patches = export_patches('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/lma_250_dataset',
                                         base_out_path,
                                         1024,
                                         1024)

remove_empty_patches_and_their_labels(empty_multiband_patches, base_out_path)

#this is almost done. HOWEVER. You need to fix the transformation. The multiband and the label overlap as they should
#but unfortunately they are in the wrong position.
