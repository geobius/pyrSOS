from pathlib import Path
import numpy as np
import random
import pyjson5
from itertools import chain
from rasterio import transform
import torch
from torch.utils.data import Sampler, Dataset
import rasterio as rio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


# Seed stuff
random.seed(999)

'''

Structure of a json split file.
It is a dictionary with 3 keys.
training set, validation set, testing set
Each key associates with a list of area names

Each subfolder represents a geographic place and contains equally sized patches
Every whole raster follows this naming convention
{area}_{platform}_{pre or post}_{gsd}_{multiband or label}_{row id}_{column id}
Every patch follows this naming convention
{area}_{platform}_{pre or post}_{gsd}_{multiband or positivelabel or negativelabel}_{row id}_{column id}
patches that have the same area and row id and column id geographically overlap pixel by pixel.


The job of the dataloader is to find which patches overlap and load them as tensors.
'''

#root_folder = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/patches128')
#root2=Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/pyrsos_250cm_dataset')
#configs1 = pyjson5.load(open('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs/common_config.json', 'r'))
#configs2 = pyjson5.load(open('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs/convolutional_config_lma.json', 'r'))
#configs3 = pyjson5.load(open('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs/pixel_config_lma.json', 'r'))
#test_label = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/pyrsos_250cm_dataset_alt/northprodromos/northprodromos_lma_post_250cm_label.tif')

def extract_area(image_name):
    return image_name.split('_')[0]
def extract_platform(image_name):
    return image_name.split('_')[1]
def extract_time(image_name):
    return image_name.split('_')[2]
def extract_gsd(image_name):
    return image_name.split('_')[3]
def extract_labelstatus(image_name):
    return image_name.split('_')[4]
def extract_rowID(image_name):
    return image_name.split('_')[5]
def extract_columnID(image_name):
    return image_name.split('_')[6]


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


def patch_name2whole_name(patch_name): #I assume this function to be used on patches but I allow it to work on whole rasters
    area = extract_area(patch_name)
    platform = extract_platform(patch_name)
    time = extract_time(patch_name)
    gsd = extract_gsd(patch_name)
    label_status = extract_labelstatus(patch_name)

    source_label_status = ''
    match(label_status):
        case 'multiband':
            source_label_status = 'multiband'
        case 'positivelabel':
            source_label_status = 'label'
        case 'negativelabel':
            source_label_status = 'label'
        case 'label': #in case i call the function for the whole image
            source_label_status = 'label'

    whole_raster_name = f'{area}_{platform}_{time}_{gsd}_{source_label_status}'
    return whole_raster_name





#glob is slow so I want to use it as little as possible
#I also want to guarantee that the label_paths[i], pre_paths[i],post_paths[i] overlap for any index i


def find_arrange_paths(areas_in_the_set, root_path, pre_platform, post_platform):

    label_paths = []
    for area in areas_in_the_set:
        area_folder = root_path/area
        label_paths.extend(list(area_folder.glob('*label*')))

    path_parents = [p.parent for p in label_paths]
    
    label_names = [p.stem for p in label_paths]
    pre_patch_names = [partial_rename(x, ['platform', 'time', 'labelstatus'], [pre_platform, 'pre', 'multiband']) for x in label_names]
    post_patch_names = [partial_rename(x, ['platform', 'time', 'labelstatus'], [post_platform, 'post', 'multiband']) for x in label_names]

    pre_paths = [parent/(stem+'.tif') for stem, parent in zip(pre_patch_names, path_parents)]
    post_paths = [parent/(stem+'.tif') for stem, parent in zip(post_patch_names, path_parents)]
    
    return pre_paths, post_paths, label_paths



def augment(pre_patch, post_patch, label_patch):
    pre_aug = pre_patch
    post_aug = post_patch
    label_aug = label_patch

    if random.random() > 0.5:
        pre_aug = TF.hflip(pre_aug)
        post_aug = TF.hflip(post_aug)
        label_aug = TF.hflip(label_aug)

    if random.random() > 0.5:
        pre_aug = TF.vflip(pre_aug)
        post_aug = TF.vflip(post_aug)
        label_aug = TF.vflip(label_aug)

    if random.random() > 0.5:
        angle = random.uniform(-5, 5)
        pre_aug = TF.rotate(pre_aug, angle=angle)
        post_aug = TF.rotate(post_aug, angle=angle)
        label_aug = TF.rotate(label_aug.unsqueeze(0), angle=angle, interpolation=InterpolationMode.NEAREST).squeeze(0)
        #the default interpolation causes my label to have continuous values and I don't want this.

    return pre_aug, post_aug, label_aug





#return
#a table with 3 or 4 columns each representing a spectral band
#for example if array of shape (4, 256, 128). I want to turn it to array of shape (256*128, 4)
#so I would have to first transpose (4,256,128) to (256, 128, 4) and then reshape (256*128, 4)
def image2tabular(image_array):
    channels, height, width = image_array.shape
    transposed = np.transpose(image_array, (1, 2, 0))
    tabular_array = np.reshape(transposed, (height * width, channels))
    return tabular_array


def retrieve_statistics_from_table(patch_name, band_indices, which_statistic, table):
    #remember to lower each band index by 1 because gdal/rasterio uses 1 based indexing
    #while numpy uses 0 based indexing

    corrected_band_indices = [i-1 for i in band_indices]
    #whole_raster_name = patch_name2whole_name(patch_name)
    patch_platform = extract_platform(patch_name)
    patch_time = extract_time(patch_name)
    source = f'{patch_platform}_{patch_time}'
    all_patch_stats = table[source]
    values_for_all_bands = np.array(all_patch_stats[which_statistic])
    values_for_selected_bands = values_for_all_bands[corrected_band_indices]
    return values_for_selected_bands




def scale_patch(image_array, patch_name, band_indices, stats_table, scaling_method): #We need information from the whole raster.
        match scaling_method:
            case 'standardization':
                global_means =  retrieve_statistics_from_table(patch_name, band_indices, 'global_mean', stats_table)[:, np.newaxis, np.newaxis]
                global_stdevs = retrieve_statistics_from_table(patch_name, band_indices, 'global_stdev', stats_table)[:, np.newaxis, np.newaxis]
                standardized_image = (image_array - global_means) / global_stdevs 
                return standardized_image

            case 'minmax':
                global_min =  retrieve_statistics_from_table(patch_name, band_indices, 'global_minima', stats_table)[:, np.newaxis, np.newaxis]
                global_max =  retrieve_statistics_from_table(patch_name, band_indices, 'global_maxima', stats_table)[:, np.newaxis, np.newaxis]
                minmax_image = (image_array - global_min) / (global_max - global_min)
                return minmax_image
            
            case 'reflectance':
                division_factors = {'sen2': 10000, 'lma' : 255}
                additive_offsets = {'sen2': -1000, 'lma': 0} #this is essential for sen2 images past january 2022
                platform = extract_platform(patch_name)
                reflectance_image = np.maximum(0, (image_array + additive_offsets[platform]) / division_factors[platform])
                return reflectance_image
            case None:
                return image_array

def scale_tabular_array(image_array, patch_name, band_indices, stats_table, scaling_method): #We need information from the whole raster.
        match scaling_method:
            case 'standardization':
                global_means =  retrieve_statistics_from_table(patch_name, band_indices, 'global_mean', stats_table)
                global_stdevs = retrieve_statistics_from_table(patch_name, band_indices, 'global_stdev', stats_table)
                standardized_image = (image_array - global_means) / global_stdevs 
                return standardized_image

            case 'minmax':
                global_min =  retrieve_statistics_from_table(patch_name, band_indices, 'global_minima', stats_table)
                global_max =  retrieve_statistics_from_table(patch_name, band_indices, 'global_maxima', stats_table)
                minmax_image = (image_array - global_min) / (global_max - global_min)
                return minmax_image
            
            case 'reflectance':
                division_factors = {'sen2': 10000, 'lma' : 255}
                additive_offsets = {'sen2': -1000, 'lma': 0} #this is essential for sen2 images past january 2022
                platform = extract_platform(patch_name)
                reflectance_image = np.maximum(0, (image_array + additive_offsets[platform]) / division_factors[platform])
                return reflectance_image
            case None:
                return image_array

            

def fuse_arrays(pre, post, fusion_method):
    match fusion_method:
        case 'only_post_image':
            return post
        case 'difference':
            return pre - post
        case 'concatenation':
            return np.concatenate([pre, post], axis=1)

    return




def load_single_image_as_tabular_array(image_path, selected_bands, scale_input_method, stats_table={}):
    with rio.open(image_path) as ds:
        name = image_path.stem
        image_array = ds.read(indexes=selected_bands)
        image_array = image2tabular(image_array)
        
        scaled_tabular_array = scale_tabular_array(image_array, name, selected_bands, stats_table, scale_input_method)
        zero_mask = ~np.all(image_array == 0, axis=1)

    #zero mask keeps in memory which entries have value [0,0,0]
    return scaled_tabular_array, zero_mask



def load_dataset_as_table(which_set, configs):
    
    splits_table = pyjson5.load(open(configs['split_filepath'], 'r'))
        
    areas_in_the_set = splits_table[which_set]

    stats_table = pyjson5.load(open(configs['stats_filepath'], 'r'))

    pre_images_paths, post_images_paths, label_images_paths = find_arrange_paths(areas_in_the_set,
                                                                                 Path(configs['dataset_folderpath']),
                                                                                 configs['pre_data_source'],
                                                                                 configs['post_data_source'])

    pre_tabular = []
    post_tabular = []
    zero_mask_tabular = []
    label_tabular = []

    for p in pre_images_paths:
        tabular_array, _ = load_single_image_as_tabular_array(p, configs['pre_selected_bands'], configs['pre_scale_input_method'], stats_table)
        pre_tabular.append(tabular_array)

    for p in post_images_paths:
        tabular_array, zero_mask = load_single_image_as_tabular_array(p, configs['post_selected_bands'], configs['post_scale_input_method'], stats_table)
        post_tabular.append(tabular_array)
        zero_mask_tabular.append(zero_mask)

    for p in label_images_paths:
        tabular_array, _ = load_single_image_as_tabular_array(p, [1] , None)
        label_tabular.append(tabular_array)

        #you need to remove void entries from the training set. They are too many and they worsen any model's performance

    if which_set == 'training_set':
        for idx, mask in enumerate(zero_mask_tabular):
            pre_tabular[idx] = (pre_tabular[idx])[mask]
            post_tabular[idx] = (post_tabular[idx])[mask]
            label_tabular[idx] = (label_tabular[idx])[mask]


    pre_total = np.concatenate(pre_tabular, axis=0)
    post_total = np.concatenate(post_tabular, axis=0)
    label_total = np.concatenate(label_tabular, axis=0).flatten()

    fused_array = fuse_arrays(pre_total, post_total, configs['fusion_method'])

    return fused_array, label_total


class Pyrsos_Dataset(Dataset):
    def __init__(self, which_set, configs):
        self.configs = configs
        self.which_set = which_set
        splits_table = pyjson5.load(open(configs['split_filepath'], 'r'))
        
        areas_in_the_set = splits_table[which_set]

        self.statistics_table = pyjson5.load(open(configs['stats_filepath'], 'r'))
        
        self.pre_patches_paths, self.post_patches_paths, self.label_patches_paths = find_arrange_paths(areas_in_the_set,
                                                                                                       Path(configs['dataset_folderpath']),
                                                                                                       configs['pre_data_source'],
                                                                                                       configs['post_data_source'])

    def __len__(self):
        return len(self.label_patches_paths)


    def __getitem__(self, index):

        pre_path = self.pre_patches_paths[index]
        pre_indices = self.configs['pre_selected_bands']
        with rio.open(pre_path) as pre_ds:
            pre_patch = pre_ds.read(indexes=pre_indices)
            pre_name = pre_path.stem

        post_path = self.post_patches_paths[index]
        post_indices = self.configs['post_selected_bands']
        with rio.open(post_path) as post_ds:
            post_patch = post_ds.read(indexes=post_indices)
            post_name = post_path.stem

        label_path = self.label_patches_paths[index]
        with rio.open(label_path) as label_ds:
            label_patch = label_ds.read(1)

        # the patches are configured in such a way that they geographically overlap pixel by pixel

        pre_patch = scale_patch(pre_patch, pre_name, pre_indices, self.statistics_table, self.configs['pre_scale_input_method'])
        post_patch = scale_patch(post_patch, post_name, post_indices, self.statistics_table, self.configs['post_scale_input_method'])

        pre_patch = torch.from_numpy(pre_patch).to(dtype=torch.float32)
        post_patch = torch.from_numpy(post_patch).to(dtype=torch.float32)
        label_patch = torch.from_numpy(label_patch).to(dtype=torch.int64)

        if self.configs['augment?'] and self.which_set == 'training set':
            pre_patch, post_patch, label_patch = augment(pre_patch, post_patch, label_patch)

        return pre_patch, post_patch, label_patch


class Burned_Area_Sampler(Sampler):
    def __init__(self, dataset):
        self.positive_samples = [index for index, label in enumerate(dataset.label_patches_paths) if 'positive' in label.stem]
    def __len__(self):
        return len(self.positive_samples)
    def __iter__(self):
        random.shuffle(self.positive_samples)
        return iter(self.positive_samples)
