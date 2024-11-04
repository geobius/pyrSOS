from pathlib import Path
import numpy as np
import random
import pickle
from rasterio import transform
import torch
from torch.utils.data import Sampler, Dataset, Subset
import rasterio as rio

# Seed stuff
np.random.seed(999)
random.seed(999)

'''
Structure of a pickle file.
There are 3 pickle files in total. The training set, The validation set, The testing set
Each pickle file contains a single dictionary. Its keys correspond to the area of interest available in the set in question
for example the training set could contain 'domokos', 'yliki'.
By choosing a key you are transfered to an ordered list of 'samples' from that area.
Sample is a dictionary that maps keys to patch paths.
It has 4keys in total: sen2_pre, sen2_post, lma_post, label
The patch_paths refer to small subsets of the original downloaded images. They all share the same bounding box coordinates and thus overlap.
You can check by loading them to gis program.

The job of the dataloader is to convert the contents of the paths into tensors.
'''

configs = pyjson5.load(open('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs/general_config.json', 'r'))

class Pyrsos_Dataset(Dataset):
    def __init__(self, mode, configs):

        self.configs = configs

        # Read the pickle files containing information on the splits
        ds_path = Path(self.configs['dataset_path'])
        self.samples_by_areas = pickle.load(open(ds_path / configs[mode], 'rb'))
       
        self.areas_in_the_set = self.samples_by_areas.keys()
        self.number_of_samples_per_area = {area: len(samples) for area, samples in self.samples_by_areas.items()}
        self.total_length = sum(self.number_of_samples_per_area.values())

        self.mixed_samples = [item for sublist in self.samples_by_areas.values() for item in sublist]

        # Keep the positive indices in a separate list (useful for under/oversampling)
        self.positives_samples = [index for index, sample in enumerate(self.mixed_samples) if 'positive' in sample['label'].name]



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
        return self.total_length


    def __getitem__(self, event_id):
        sample = self.mixed_samples[event_id]
        pre_image, post_image, label_image, transform = self.load_images(sample)
        scaled_pre_image = self.scale_image(pre_image, self.configs['pre_data_source'], self.configs['pre_normalize?'])
        scaled_post_image = self.scale_image(post_image, self.configs['post_data_source'], self.configs['post_normalize?'])

        return scaled_pre_image, scaled_post_image, label_image, transform

'''
def find_indices_for_area(dataset, area):

def create_dataloader_from_area(dataset, area):
    patches_from_the_same_image = dataset.patches_by_areas[area]

    # return a dataloader that only contains the patches that belong to the area in question
'''


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
