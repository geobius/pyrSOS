from pathlib import Path
import numpy as np
import random
import pyjson5
from rasterio import transform
import torch
from torch.utils.data import Sampler, Dataset
import rasterio as rio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torchvision.transforms as T
import torchvision.transforms.functional as TF


# Seed stuff
random.seed(999)

'''

Structure of a json split file.
It is a dictionary with 3 keys.
training set, validation set, testing set
Each key associates with a list of area names

Each subfolder represents a geographic place and contains equally sized patches
Every patch follows this naming convention
{area}_{platform}_{pre or post}_{gsd}_{multiband or label}_{row id}_{column id}.tif
patches that have the same area and row id and column id geographically overlap pixel by pixel.


The job of the dataloader is to find which patches overlap and load them as tensors.

'''

#configs = pyjson5.load(open('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs/general_config.json', 'r'))


def substitute_names(label_patches_paths, platform, pre_or_post):
    new_patches_paths = []
    for path in label_patches_paths:
        naming_components = path.stem.split('_')
        root = path.parent

        naming_components[1] = platform
        naming_components[2] = pre_or_post
        naming_components[4] = 'multiband'

        new_stem = "_".join(naming_components)
        new_path = root/f'{new_stem}.tif'
        new_patches_paths.append(new_path)

    return new_patches_paths



class Pyrsos_Dataset(Dataset):
    def __init__(self, mode, configs):

        ds_path = Path(configs['dataset_folderpath'])
        splits = pyjson5.load(open(configs['event_split_filepath'], 'r'))

        self.configs = configs
        self.mode = mode
        self.areas_in_the_set = splits[mode]

        label_paths_per_area = [list((ds_path/area).glob('*label_*.tif')) for area in self.areas_in_the_set]
        merged_label_paths = [item for sublist in label_paths_per_area for item in sublist]

        self.pre_patches_paths = substitute_names(merged_label_paths, configs['pre_data_source'], 'pre')
        self.post_patches_paths = substitute_names(merged_label_paths, configs['post_data_source'], 'post')
        self.label_patches_paths = merged_label_paths
       
        self.samples = list(zip(self.pre_patches_paths, self.post_patches_paths, self.label_patches_paths))



    def augment(self, pre_patch, post_patch, label_patch):
        '''
        Applies the following augmentations:
        - Random horizontal flipping (possibility = 0.5)
        - Random vertical flipping (possibility = 0.5)
        - Random rotation (-15 to +15 deg)
        '''

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
            label_aug = TF.rotate(label_aug.unsqueeze(0), angle=angle).squeeze()


        return pre_aug, post_aug, label_aug



    def load_images(self, sample):
        '''
        Each sample is a list of 3 paths to overlapping patches, pre, post, label in this order.
        Find the appropriate paths, load the .tif images, strip away the geocoordinates
        and load the requested bands as tensors.
        '''
        pre_image_path = sample[0]
        post_image_path = sample[1]
        label_path = sample[2]

        with rio.open(pre_image_path) as pre_ds:
            pre_image = torch.from_numpy(pre_ds.read(self.configs['pre_selected_bands']))

        with rio.open(post_image_path) as post_ds:
            post_image = torch.from_numpy(post_ds.read(self.configs['post_selected_bands']))

        with rio.open(label_path) as label_ds:
            label_image = torch.from_numpy(np.squeeze(label_ds.read())).to(dtype=torch.long)
            #a tensor with dimensions (patch_height, patch_width). The elements are class indices
            transform = label_ds.transform

        return pre_image, post_image, label_image, transform


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        sample = self.samples[index]
        pre_image, post_image, label_image, transform = self.load_images(sample)

        if self.configs['augment?'] and self.mode == 'training set':
            pre_image, post_image, label_image = self.augment(pre_image, post_image, label_image)
        if self.configs['pre_normalize?']:
            pre_image = pre_image / 255
        if self.configs['post_normalize?']:
            post_image = post_image / 255

        return pre_image, post_image, label_image, transform


class Burned_Area_Sampler(Sampler):

    def __init__(self, dataset):
        self.positive_samples = [index for index, sample in enumerate(dataset.samples) if 'positive' in sample[2].stem]

    def __len__(self):
        return len(self.positive_samples)

    def __iter__(self):
        random.shuffle(self.positive_samples)
        return iter(self.positive_samples)


#return
#a table with 4 columns each represent a spectral band
#for example if array of shape (4, 256, 128). I want to turn it to array of shape (256*128, 4)
#so I would have to first transpose (4,256,128) to (256, 128, 4) and then reshape (256*128, 4)
def image2tabular(image_array):
    channels, height, width = image_array.shape
    transposed = np.transpose(image_array, (1, 2, 0))
    tabular_array = np.reshape(transposed, (height * width, channels))

    if channels == 1:
        tabular_array = tabular_array.flatten()

    return tabular_array

def tabular2image(tabular_array, height, width):
    image_array = []
    if tabular_array.ndim == 1:
        image_array = np.reshape(tabular_array, (1, height, width))
    else:
        channels = tabular_array.shape[1]
        transposed = np.transpose(tabular_array, (1, 0))
        image_array = np.reshape(transposed, (channels, height, width))

    return image_array


def load_dataset_as_table(mode, pixel_configs):

    source_dataset_path = Path(pixel_configs['dataset_path'])
    split_filename = pixel_configs['split_filename']

    splits = pyjson5.load(open(source_dataset_path / split_filename, 'r'))
    areas_in_the_set = splits[mode]

    label_paths_per_area = [list((source_dataset_path/area).glob('*label.tif')) for area in areas_in_the_set]
    merged_label_paths = [item for sublist in label_paths_per_area for item in sublist]

    pre_platform = pixel_configs['pre_data_source']
    pre_selected_bands = pixel_configs['pre_selected_bands']
    post_platform = pixel_configs['post_data_source']
    post_selected_bands = pixel_configs['post_selected_bands']

    pre_images_paths = substitute_names(merged_label_paths, pre_platform, 'pre')
    post_images_paths = substitute_names(merged_label_paths, post_platform, 'post')
    label_images_paths = merged_label_paths

    pre_tabular = []
    post_tabular = []
    label_tabular = []

    for p in pre_images_paths:
        pre_tabular = []
        image_array = rio.open(p).read(pre_selected_bands)
        pre_tabular.append(image2tabular(image_array))

    for p in post_images_paths:
        post_tabular = []
        image_array = rio.open(p).read(post_selected_bands)
        post_tabular.append(image2tabular(image_array))

    for p in label_images_paths:
        label_tabular = []
        image_array = rio.open(p).read()
        label_tabular.append(image2tabular(image_array))


    pre_concatenated = np.concatenate(pre_tabular, 0)
    post_concatenated = np.concatenate(post_tabular, 0)
    label_concatenated = np.concatenate(label_tabular, 0)
    difference_concatenated = pre_concatenated - post_concatenated

    if pixel_configs['use_only_post_image?']:
        return post_concatenated, label_concatenated
    else:
        return difference_concatenated, label_concatenated
