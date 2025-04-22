#!/usr/bin/python3

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

from training_utilities.dataloaders import image2tabular, fuse_arrays, find_arrange_paths,scale_patch



class visualizer():
    def __init__(self, trained_model, configs):
        self.which_set = configs['visualization_set']
        self.configs = configs
        splits_table = pyjson5.load(open(configs['split_filepath'], 'r'))
        
        areas_in_the_set = splits_table[self.which_set]

        self.statistics_table = pyjson5.load(open(configs['stats_filepath'], 'r'))
        
        self.pre_patches_paths, self.post_patches_paths, self.label_patches_paths = find_arrange_paths(areas_in_the_set,
                                                                                                       Path(configs['patches_folderpath']),
                                                                                                       configs['pre_data_source'],
                                                                                                       configs['post_data_source'])
        self.trained_model = trained_model


        self.dataset_size = len(self.label_patches_paths)
        self.index = 0

        self.fig, self.axes = plt.subplots(1, 4, figsize=(20, 5))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        #self.display()
        #plt.show()

    def load_patches(self): #those patches are just for show
        pre_path = self.pre_patches_paths[self.index]
        pre_indices = self.configs['pre_selected_bands']
        with rio.open(pre_path) as pre_ds:
            pre_patch = pre_ds.read(indexes=pre_indices)

        if self.configs['pre_data_source'] == 'sen2':
            pre_patch = (np.floor(pre_patch*(255/10000))).astype(np.uint8)
        pre_patch = np.transpose(pre_patch,(1, 2, 0))[:, :, [2, 1, 0]]
         
        post_path = self.post_patches_paths[self.index]
        post_indices = self.configs['post_selected_bands']
        with rio.open(post_path) as post_ds:
            post_patch = post_ds.read(indexes=post_indices)

        if self.configs['post_data_source'] == 'sen2':
            post_patch = (np.floor(post_patch*(255/10000))).astype(np.uint8)
        post_patch = np.transpose(post_patch,(1, 2, 0))[:, :, [2, 1, 0]]

        label_path = self.label_patches_paths[self.index]
        with rio.open(label_path) as label_ds:
            label_patch = label_ds.read(1)

        return pre_patch, post_patch, label_patch

    def load_transformed_patches(self): #those patches are to be pushed to the model. The model might expect a specific transformation to be performed on the data first
        pre_path = self.pre_patches_paths[self.index]
        pre_indices = self.configs['pre_selected_bands']
        with rio.open(pre_path) as pre_ds:
            transformed_pre_patch = pre_ds.read(indexes=pre_indices)
            pre_name = pre_path.stem

        transformed_pre_patch = scale_patch(transformed_pre_patch, pre_name, pre_indices, self.statistics_table, self.configs['pre_scale_input_method'])
        
        post_path = self.post_patches_paths[self.index]
        post_indices = self.configs['post_selected_bands']
        with rio.open(post_path) as post_ds:
            transformed_post_patch = post_ds.read(indexes=post_indices)
            post_name = post_path.stem

        transformed_post_patch = scale_patch(transformed_post_patch, post_name, post_indices, self.statistics_table, self.configs['post_scale_input_method'])

        return transformed_pre_patch, transformed_post_patch

    def generate_mask(transformed_pre_patch, transformed_post_patch): #this method acts differently depending on whether it was called from a convolutional or a pixel model
        return

    def display(self):
        pre_patch, post_patch, label_patch = self.load_patches()
        transformed_pre_patch, transformed_post_patch = self.load_transformed_patches()
        prediction_mask = self.generate_mask(transformed_pre_patch, transformed_post_patch)
        
        burnt_scheme = ListedColormap(['black', 'orange'])
        _, height, width = post_patch.shape

        for ax in self.axes:
            ax.clear()

        self.axes[0].set_title('pre patch')
        self.axes[0].imshow(pre_patch)

        self.axes[1].set_title('post patch')
        self.axes[1].imshow(post_patch)

        self.axes[2].set_title('manual label')
        self.axes[2].imshow(label_patch, cmap=burnt_scheme, vmin=0, vmax=1)

        self.axes[3].set_title('model prediction')
        self.axes[3].imshow(prediction_mask, cmap=burnt_scheme, vmin=0, vmax=1)

        plt.tight_layout(pad=0.0)

    def next(self):
        self.index = (self.index + 1) % self.dataset_size
    def previous(self):
        self.index = (self.index - 1) % self.dataset_size
        
    def on_key(self, event):
        match(event.key):
            case 'right':
                self.next()
            case 'left':
                self.previous()
        self.display()
        #plt.show()
        self.fig.canvas.draw_idle()

    def on_print(self, k):
        base_out_path = Path(self.configs['save_state_folderpath'])
        for i in range(k):
            self.display()
            plt.savefig(base_out_path/f'{i}.png')
            self.next()

        
class pixel_visualizer(visualizer):
    def generate_mask(self, transformed_pre_patch, transformed_post_patch):
        _, height, width = transformed_post_patch.shape
        pre_tabular_array = image2tabular(transformed_pre_patch)
        post_tabular_array = image2tabular(transformed_post_patch)
        fused_tabular_array = fuse_arrays(pre_tabular_array, post_tabular_array, self.configs['fusion_method'])
        class_indices = self.trained_model.predict(fused_tabular_array).astype(np.uint8) #shape (n_samples,)
        mask = np.reshape(class_indices, (height, width))
        return mask


class convolutional_visualizer(visualizer):
    def generate_mask(self, transformed_pre_patch, transformed_post_patch):
        pre_tensor = torch.from_numpy(transformed_pre_patch)
        post_tensor = torch.from_numpy(transformed_post_patch)

        pre_tensor = pre_tensor.unsqueeze(0).float().to(device=self.configs['device'])
        post_tensor = post_tensor.unsqueeze(0).float().to(device=self.configs['device'])

        with torch.no_grad():
            self.trained_model.eval()
            prediction_tensor = self.trained_model(pre_tensor, post_tensor)
            probabilities = torch.softmax(prediction_tensor, dim=1)
            mask = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()
        return mask
