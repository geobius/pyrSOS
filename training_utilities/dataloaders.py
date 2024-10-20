from pathlib import Path
import numpy as np
import random
import pickle
import rioxarray as rxr
import torch
from torch.utils.data import Sampler


# Seed stuff
np.random.seed(999)
random.seed(999)


class Pyrsos_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, configs):

        self.configs = configs

        # Read the pickle files containing information on the splits
        ds_path = Path(self.configs['dataset_path'])
        patches_by_areas = pickle.load(open(ds_path / configs[mode], 'rb'))
        self.merged_patches = [item for sublist in patches_by_areas.values() for item in sublist]

        # Keep the positive indices in a separate list (useful for under/oversampling)
        self.positives_indices = [index for index, patch in enumerate(self.merged_patches) if 'positive' in patch['label'].name]


    def scale_image(self, image, data_source, must_normalize):
        scaling_constant = 1
        match data_source:
            case 'sen2_pre':
                scaling_constant = 10000
            case 'sen2_post':
                scaling_constant = 10000
            case 'lma':
                scaling_constant = 255

        scaled_image = image

        if must_normalize:
            scaled_image = image.to(torch.float32) / scaling_constant

        return scaled_image


    def load_images(self, sample):
        '''
        Each sample is a dictionary that maps keys to patch paths.
        Find the appropriate paths, load the .tif images, strip away the geocoordinates
        and load the requested bands as tensors.
        '''
        pre_image_path = sample[self.configs['pre_data_source']]
        post_image_path = sample[self.configs['post_data_source']]
        label_path = sample['label']

        pre_ds = (rxr.open_rasterio(pre_image_path)).sel(band=self.configs['pre_selected_bands'])
        pre_image = torch.from_numpy(pre_ds.values)
        pre_ds.close()

        post_ds = (rxr.open_rasterio(post_image_path)).sel(band=self.configs['post_selected_bands'])
        post_image = torch.from_numpy(post_ds.values)
        post_ds.close()

        label_ds = rxr.open_rasterio(label_path)
        label_image = torch.from_numpy(label_ds.values)
        label_ds.close()

        return pre_image, post_image, label_image


    def __len__(self):
        return len(self.merged_patches)


    def __getitem__(self, event_id):
        sample = self.merged_patches[event_id]
        pre_image, post_image, label_image = self.load_images(sample)
        scaled_pre_image = self.scale_image(pre_image, self.configs['pre_data_source'], self.configs['pre_normalize?'])
        scaled_post_image = self.scale_image(post_image, self.configs['post_data_source'], self.configs['post_normalize?'])

        return scaled_pre_image, scaled_post_image, label_image


class OverSampler(Sampler):
    '''
    A Sampler which performs oversampling in imbalanced datasets.
    '''
    def __init__(self, dataset, positive_prc=0.5):
        self.dataset = dataset
        self.positive_prc = positive_prc
        self.n_samples = len(dataset)


    def __iter__(self):
        positives = self.dataset.events_df[self.dataset.events_df['positive_flag']].index.values
        pos = np.random.choice(positives, int(self.positive_prc * self.n_samples), replace=True)
        neg = np.random.choice(list(set(self.dataset.events_df.index.values) - set(positives)), int(((1 - self.positive_prc) * self.n_samples) + 1))

        idx = np.hstack([pos, neg])
        np.random.shuffle(idx)

        idx = idx[:self.n_samples]

        pos_cnt = len([i for i in idx if i in pos])
        print(f'Using {pos_cnt} POS and {len(idx) - pos_cnt} NEG (1:{((len(idx) - pos_cnt) / pos_cnt):.2f}).')

        return iter(idx)


    def __len__(self):
        return len(self.dataset)
