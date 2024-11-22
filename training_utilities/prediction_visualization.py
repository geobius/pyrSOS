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
from dataset_cleaners.pyrsos_make_dataset import get_padding_offset, pad_image
import pickle

"""
def predict_whole_mask(model, loader, patch_height, patch_width):

    target_height = post_raster.shape[1]
    target_width = post_raster.shape[2]

    top, bottom, left, right = get_padding_offset(target_width, target_height, patch_width, patch_height)
    pre_raster_padded = pad_image(pre_raster, top, bottom, left, right)
    post_raster_padded = pad_image(post_raster, top, bottom, left, right)

    padded_height = post_raster.shape[1]
    padded_width = post_raster.shape[2]

    padded_prediction = np.zeros((1, padded_height, padded_width))
#no need for bounds checks since I have ensured that the padded image dimensions are divisible by the patch dimensions.
    for row in range(0, padded_height, patch_height):
        for column in range(0, padded_width, patch_width):
            end_row = row + patch_height
            end_column = column + patch_width
            pre_input_patch = pre_raster_padded[:, row:end_row, column: end_column]
            post_input_patch = post_raster_padded[:, row:end_row, column: end_column]

            pre_tensor = torch.from_numpy(pre_input_patch).float().unsqueeze(0)
            post_tensor = torch.from_numpy(post_input_patch).float().unsqueeze(0)

            with torch.no_grad():
                model.eval()
                prediction_tensor = model(pre_tensor, post_tensor)

            prediction_patch = prediction_tensor.squeeze().numpy()

            padded_prediction[:, row:end_row, column:end_column] = prediction_patch

    unpadded_prediction = padded_prediction[:,
                                            top: padded_width - bottom,
                                            left: padded_width - right]

    return unpadded_prediction

"""


def load_lma_patches_multiband(sample):
    post_image_path = sample['lma']
    label_path = sample['label']

    with rio.open(post_image_path) as post_ds:
        post_patch = post_ds.read()

    with rio.open(label_path) as label_ds:
        label_image = label_ds.read(1)
        transform = label_ds.transform

    return post_patch, label_image, transform



#this function is only for pixel by pixel classifiers
def patch_to_prediction(trained_model, patch_array, pca=None):
    channels, height, width = patch_array.shape
    transposed = np.transpose(patch_array, (1, 2, 0))
    table_shape = np.reshape(transposed, (height * width, channels))

    if pca is not None:
        table_shape = pca.transform(table_shape)

    class_indices = trained_model.predict(table_shape)
    patch_shape_prediction = (np.reshape(class_indices, (height, width))).astype(np.uint8)

    return patch_shape_prediction





class lma_pixel_classifier_visualizer():
    def __init__(self, trained_model, path_to_pickle_file, pca_object=None):
        self.trained_model = trained_model
        self.pca_object = pca_object
        self.fig, self.axes = plt.subplots(1, 3, figsize=(20, 10))
        self.index = 0

        ds_path = Path(path_to_pickle_file)
        samples_by_areas = pickle.load(open(path_to_pickle_file, 'rb'))
        self.mixed_samples = [item for sublist in samples_by_areas.values() for item in sublist]
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.display()
        plt.show()

    def display(self):
        sample = self.mixed_samples[self.index]
        post_patch, label_image, transform = load_lma_patches_multiband(sample)
        prediction = patch_to_prediction(self.trained_model, post_patch, self.pca_object)

        burnt_scheme = ListedColormap(['black', 'orange'])
        _, height, width = post_patch.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        image_extent = (left, right, bottom, top)

        for ax in self.axes:
            ax.clear()

        post_patch_visual = np.transpose(post_patch, (1, 2, 0))[:, :, [3, 1, 0]]
        self.axes[0].set_title('post patch')
        self.axes[0].imshow(post_patch_visual, extent=image_extent)

        self.axes[1].set_title('manual label')
        self.axes[1].imshow(label_image, extent=image_extent, cmap=burnt_scheme)

        self.axes[2].set_title('model prediction')
        self.axes[2].imshow(prediction, extent=image_extent, cmap=burnt_scheme)

        self.fig.canvas.draw()


    def on_key(self, event):
        match(event.key):
            case 'right':
                self.index = (self.index + 1) % len(self.mixed_samples)
            case 'left':
                self.index = (self.index - 1) % len(self.mixed_samples)

        self.display()
