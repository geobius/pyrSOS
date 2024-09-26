from enum import unique
import numpy as np
import rasterio
from rasterio.transform import from_origin
import pickle
from pathlib import Path
from itertools import product, starmap
import gc
from sklearn.model_selection import train_test_split
import argparse

nodata_value = {
    'sen2': 0,
    'lma': 255
}


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

    new_image = np.pad(image,
                       pad_width=((0, 0), (top_padding, bottom_padding), (left_padding, right_padding)),
                       mode='constant',
                       constant_values=nodata_value[sen2_or_lma])

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

    #area_of_interest, sen2_or_lma, pre_or_post, gsd_in_meters, multiband_or_label = re.split('_', source_raster_name)[:-1] #I don't want the .tif part
    area_of_interest, sen2_or_lma, pre_or_post, gsd_in_centimeters, multiband_or_label = source_raster_name.stem.split('_')
    positivelabel_or_negativelabel_or_multiband = is_positive_or_negative_or_multiband(patch, multiband_or_label)
    name = f'{area_of_interest}_{sen2_or_lma}_{pre_or_post}_{gsd_in_centimeters}_{positivelabel_or_negativelabel_or_multiband}_{current_patch_offset}.tif'

    return name


def crop(image, i, j, patch_width, patch_height):
    patch = image[:, i: i + patch_width, j: j + patch_height]
    return patch


def possibly_extend_nodata_patches_list(patch, patch_name, list_of_nodata_patchnames):
    sen2_or_lma = 'sen2' if 'sen2' in patch_name else 'lma'
    if ('multiband' in patch_name) and np.all(patch == nodata_value[sen2_or_lma]):
        extended_list = np.append(list_of_nodata_patchnames, patch_name)
    else:
        extended_list = list_of_nodata_patchnames

    return extended_list


def export_patches(source_folder, base_out_path, patch_width=128, patch_height=128):
    #all patches are exported as tif files
    zero_patches = []
    all_files = source_folder.glob('*')

    for current_file in all_files:
        hdd_image = rasterio.open(source_folder/current_file)
        initial_transformation = hdd_image.transform

        source_raster_name = hdd_image.name

        sen2_or_lma = 'sen2' if 'sen2' in source_raster_name else 'lma'

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


source_folder = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/pyrsos_250cm_dataset')
base_out_path = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination')


def find_paths_of_overlaying_patches(area, ID, search_path):
    lma = list(search_path.glob(f'{area}_lma_*_multiband_{ID}.tif'))[0]
    sen2_pre = list(search_path.glob(f'{area}_sen2_pre_*_{ID}.tif'))[0]
    sen2_post = list(search_path.glob(f'{area}_sen2_post_*_{ID}.tif'))[0]
    label = list(search_path.glob(f'{area}_lma_*label_{ID}.tif'))[0]
    label_is_positive = True if 'positive' in label.stem else False

    dictionary = {
        'lma': lma,
        'sen2_pre': sen2_pre,
        'sen2_post': sen2_post,
        'label': label,
        'label_is_positive': label_is_positive
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

def make_patchpaths_table(search_path):
    areas_of_interest = ['domokos', 'prodromos', 'yliki', 'delphoi']
    table = []
    for region in areas_of_interest:
        IDs = unique_IDs(region, search_path)
        for patch_ID in IDs:
            table.append(find_paths_of_overlaying_patches(region, patch_ID, search_path))

    return table


def split_dataset(base_out_path, train_percentage, test_percentage, optional_prefix):
    main_percentage = 100 - test_percentage
    val_percentage = main_percentage - train_percentage

    all_files = make_patchpaths_table(base_out_path/'patches')
    adhoc_labels_list = [0]*len(all_files)
    main_set, test_set, _, _ = train_test_split(all_files, adhoc_labels_list, train_size=main_percentage/100, test_size=test_percentage/100, random_state=42)
    adhoc_labels_list = [0]*len(main_set)
    training_set, validation_set, _, _ = train_test_split(main_set, adhoc_labels_list, train_size=train_percentage/100, test_size=val_percentage/100, random_state=42)

    train_pickle_name = f'allEvents_{train_percentage}_{val_percentage}_{test_percentage}_{optional_prefix}_train.pkl'
    val_pickle_name = f'allEvents_{train_percentage}_{val_percentage}_{test_percentage}_{optional_prefix}_val.pkl'
    test_pickle_name = f'allEvents_{train_percentage}_{val_percentage}_{test_percentage}_{optional_prefix}_test.pkl'

    pickle.dump(training_set, open(base_out_path/train_pickle_name, 'wb'))
    pickle.dump(validation_set, open(base_out_path/val_pickle_name, 'wb'))
    pickle.dump(test_set, open(base_out_path/test_pickle_name, 'wb'))


    return (training_set, validation_set, test_set)

tr,val,tst = split_dataset(base_out_path, 60, 20, 'v1')


if __name__ == 'main':
    parser = argparse.ArgumentParser(description='This program works in two phases.\
    In phase 1: All raster images are split into equally sized patches.\
    In phase_2: Three pickle files are created, each containing mutually exclusive paths to the patches\
    and representing different stages in the learning pipeline.')

    parser.add_argument('dataset_path', default='pyrsos_250cm_dataset')
    parser.add_argument('base_out_bath', default='destination')
    parser.add_argument('patch_width', type=int, default=128)
    parser.add_argument('patch_height', type=int, default=128)

    args= parser.parse_args()
    empty_multiband_patches = export_patches(args.,
                                         base_out_path,
                                         1024,
                                         1024)

    remove_empty_patches_and_their_labels(empty_multiband_patches, base_out_path)
