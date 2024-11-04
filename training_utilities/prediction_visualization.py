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

from dataset_cleaners.pyrsos_make_dataset import get_padding_offset, pad_image


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


def merge_predictions_by_area(model, area_loader, device):
    model.eval()
    with torch.no_grad():
    merged = {}
    for (pre_image, post_image, _, transform) in loader.dataset.patches_by_areas[area]:
        output = model(pre_image, post_image)
