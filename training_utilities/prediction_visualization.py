#!/usr/bin/python3
from os import mkdir
from os.path import exists
from pathlib import Path
import numpy as np
import rasterio as rio
from rasterio import transform
import torch
from tqdm import tqdm
from rasterio.merge import merge
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pyjson5

from training_utilities.dataloaders import substitute_names, Pyrsos_Dataset


#configs_filepath = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs/pixel_config.json')
configs2 = pyjson5.load(open(Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs/convolutional_config.json'), 'r'))



def patch_to_prediction_pixel(trained_model, pre_patch, post_patch, use_only_post):
    patch_array = []
    if use_only_post:
        patch_array = post_patch
    else:
        patch_array = pre_patch - post_patch

    channels, height, width = patch_array.shape
    transposed = np.transpose(patch_array, (1, 2, 0))
    table_shape = np.reshape(transposed, (height * width, channels))
    class_indices = trained_model.predict(table_shape)
    mask = (np.reshape(class_indices, (height, width))).astype(np.uint8)

    return mask


def patch_to_prediction_convolutional(trained_model, pre_patch, post_patch, device):
    pre_tensor = pre_patch.unsqueeze(0).float().to(device=device)
    post_tensor = post_patch.unsqueeze(0).float().to(device=device)

    with torch.no_grad():
        trained_model.eval()
        prediction_tensor = trained_model(pre_tensor, post_tensor)
        probabilities = torch.softmax(prediction_tensor, dim=1)
        mask = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()
    return mask


class pixel_visualizer():
    def __init__(self, which_set, configs, trained_model=None):
        self.which_set = which_set
        self.configs = configs
        self.trained_model = trained_model

        self.fig, self.axes = 0, 0
        if trained_model is None:
            self.fig, self.axes = plt.subplots(1, 3, figsize=(20, 10))
        else:
            self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 10))

        self.index = 0

        ds_path = Path(self.configs['patches_path'])
        splits = pyjson5.load(open(ds_path / self.configs['split_filename'], 'r'))

        self.areas_in_the_set = splits[which_set]

        label_paths_per_area = [list((ds_path/area).glob('*label_*.tif')) for area in self.areas_in_the_set]
        merged_label_paths = [item for sublist in label_paths_per_area for item in sublist]

        self.pre_patches_paths = substitute_names(merged_label_paths, self.configs['pre_data_source'], 'pre')
        self.post_patches_paths = substitute_names(merged_label_paths, self.configs['post_data_source'], 'post')
        self.label_patches_paths = merged_label_paths

        self.samples = list(zip(self.pre_patches_paths, self.post_patches_paths, self.label_patches_paths))
        self.dataset_size = len(self.samples)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.display()
        plt.show()


    def display(self):

        pre_patch, post_patch, label_patch, transform = self.load_patches(self.samples[self.index])
        prediction_mask = []
        if self.trained_model is None:
            prediction_mask = None
        else:
            prediction_mask = patch_to_prediction_pixel(self.trained_model, pre_patch, post_patch,
                                                        self.configs['use_only_post_image?'])

        burnt_scheme = ListedColormap(['black', 'orange'])
        _, height, width = post_patch.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        image_extent = (left, right, bottom, top)

        for ax in self.axes:
            ax.clear()

        pre_patch_visual = np.transpose(pre_patch,
                                        (1, 2, 0))[:, :, [3, 1, 0]]
        self.axes[0].set_title('pre patch')
        self.axes[0].imshow(pre_patch_visual, extent=image_extent)


        post_patch_visual = np.transpose(post_patch,
                                         (1, 2, 0))[:, :, [3, 1, 0]]
        self.axes[1].set_title('post patch')
        self.axes[1].imshow(post_patch_visual, extent=image_extent)


        self.axes[2].set_title('manual label')
        self.axes[2].imshow(label_patch,
                            extent=image_extent, cmap=burnt_scheme)

        if prediction_mask is not None:
            self.axes[3].set_title('model prediction')
            self.axes[3].imshow(prediction_mask,
                                extent=image_extent, cmap=burnt_scheme)

        self.fig.canvas.draw()

    def load_patches(self, sample):
        pre_image_path = sample[0]
        post_image_path = sample[1]
        label_path = sample[2]

        with rio.open(pre_image_path) as pre_ds:
            pre_image = pre_ds.read(self.configs['pre_selected_bands'])

        with rio.open(post_image_path) as post_ds:
            post_image = post_ds.read(self.configs['post_selected_bands'])

        with rio.open(label_path) as label_ds:
            label_image = label_ds.read(1)
            #a tensor with dimensions (patch_height, patch_width). The elements are class indices
            transform = label_ds.transform

        return pre_image, post_image, label_image, transform


    def on_key(self, event):
        match(event.key):
            case 'right':
                self.index = (self.index + 1) % self.dataset_size
            case 'left':
                self.index = (self.index - 1) % self.dataset_size
               
        self.display()

#vis = pixel_visualizer('testing set', configs_filepath)

##################
##################
##################
##################
##################
##################

class convolutional_visualizer():
    def __init__(self, configs, trained_model=None):
        self.which_set = configs['visualization_set']
        self.configs = configs
        self.trained_model = trained_model

        self.fig, self.axes = 0, 0
        if trained_model is None:
            self.fig, self.axes = plt.subplots(1, 3, figsize=(20, 10))
        else:
            self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 10))

        self.index = 0
        self.dataset = Pyrsos_Dataset(self.which_set, self.configs)
        self.dataset_size = len(self.dataset)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.display()
        plt.show()


    def display(self):
        pre_patch, post_patch, label_patch, transform = self.dataset[self.index]
        prediction_mask = []
        if self.trained_model is None:
            prediction_mask = None
        else:
            prediction_mask = patch_to_prediction_convolutional(self.trained_model, pre_patch, post_patch,
                                                        self.configs['device'])

        burnt_scheme = ListedColormap(['black', 'orange'])
        _, height, width = post_patch.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        image_extent = (left, right, bottom, top)

        for ax in self.axes:
            ax.clear()

        pre_patch_visual = np.transpose(pre_patch,
                                        (1, 2, 0))[:, :, [3, 1, 0]]
        self.axes[0].set_title('pre patch')
        self.axes[0].imshow(pre_patch_visual, extent=image_extent)


        post_patch_visual = np.transpose(post_patch,
                                         (1, 2, 0))[:, :, [3, 1, 0]]
        self.axes[1].set_title('post patch')
        self.axes[1].imshow(post_patch_visual, extent=image_extent)


        self.axes[2].set_title('manual label')
        self.axes[2].imshow(label_patch,
                            extent=image_extent, cmap=burnt_scheme)

        if prediction_mask is not None:
            self.axes[3].set_title('model prediction')
            self.axes[3].imshow(prediction_mask,
                                extent=image_extent, cmap=burnt_scheme)

        self.fig.canvas.draw()


    def on_key(self, event):
        match(event.key):
            case 'right':
                self.index = (self.index + 1) % self.dataset_size
            case 'left':
                self.index = (self.index - 1) % self.dataset_size


        self.display()

#vis = convolutional_visualizer(configs2)
