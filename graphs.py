#!/usr/bin/python3
import rasterio as rio
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from training_utilities import dataloaders
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

common_folder = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/pyrsos_250cm_dataset')
"""
test1 = common_folder / 'delphoi_lma_post_250cm_multiband.tif'
test1_label = common_folder / 'delphoi_lma_post_250cm_label.tif'

test2 = common_folder / 'prodromos_lma_post_250cm_multiband.tif'
test2_label = common_folder / 'prodromos_lma_post_250cm_label.tif'

test3 = common_folder / 'yliki_lma_post_250cm_multiband.tif'
test3_label = common_folder / 'yliki_lma_post_250cm_label.tif'

test4 = common_folder / 'domokos_lma_post_250cm_multiband.tif'
test4_label = common_folder / 'domokos_lma_post_250cm_label.tif'
"""



def histogram_burnt(common_folder, area):
    lma_path = next(common_folder.glob(f'{area}*lma*multiband.tif'))
    mask_path = next(common_folder.glob(f'{area}*lma*label.tif'))

    with rio.open(lma_path) as multiband_ds:
        area_image = multiband_ds.read()
    with rio.open(mask_path) as mask_ds:
        area_mask = mask_ds.read(1)

    burnt_subset = area_image[:, area_mask == 1]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ['green', 'red', 'brown', 'purple']
    xlabels = ['green', 'red', 'red edge', 'near infared']
    fig.suptitle(f"{area} Burnt Histograms per channel")

    for i in range(4):
        axes[i].hist(burnt_subset[i, :], bins=256, density=True, color=colors[i])
        axes[i].set_xlabel(xlabels[i])
        axes[i].set_ylabel("density")

    plt.tight_layout()
    plt.show()

    return


def histogram_burnt_all_areas(common_folder):
    burnt_subsets = []
    for area in ['domokos', 'yliki', 'prodromos', 'delphoi']:
        lma_path = next(common_folder.glob(f'{area}*lma*multiband.tif'))
        mask_path = next(common_folder.glob(f'{area}*lma*label.tif'))

        with rio.open(lma_path) as multiband_ds:
            area_image = multiband_ds.read()

        with rio.open(mask_path) as mask_ds:
            area_mask = mask_ds.read(1)

        burnt_subsets.append(area_image[:, area_mask == 1])

    burnt_subsets = np.concatenate(burnt_subsets, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ['green', 'red', 'brown', 'purple']
    xlabels = ['green', 'red', 'red edge', 'near infared']
    fig.suptitle("Concatenated Burnt Area Histograms per channel")

    for i in range(4):
        axes[i].hist(burnt_subsets[i, :], bins=256, density=True, color=colors[i])
        axes[i].set_xlabel(xlabels[i])
        axes[i].set_ylabel("density")

    plt.tight_layout()
    plt.show()

    return

def scatterplot_burnt_all_areas(common_folder):
    burnt_subsets = []
    for area in ['domokos', 'yliki', 'prodromos', 'delphoi']:
        lma_path = next(common_folder.glob(f'{area}*lma*multiband.tif'))
        mask_path = next(common_folder.glob(f'{area}*lma*label.tif'))

        with rio.open(lma_path) as multiband_ds:
            area_image = multiband_ds.read()

        with rio.open(mask_path) as mask_ds:
            area_mask = mask_ds.read(1)

        burnt_subsets.append(area_image[:, area_mask == 1].T)

    burnt_subsets = np.concatenate(burnt_subsets, 0)
    column_combinations = list(combinations(range(4), 2))


    fig, axes = plt.subplots(2, 3)
    axes = axes.ravel()
    labels = ['Green', 'Red', 'Red Edge', 'Near Infared']
    fig.suptitle("Scatterplots between channels for burnt pixels")

    number_of_rows_to_select = 10000
    np.random.seed(103)
    random_rows = np.random.choice(burnt_subsets.shape[0], number_of_rows_to_select, replace=False)

    for i, (x, y) in enumerate(column_combinations):
        axes[i].scatter(burnt_subsets[[random_rows], x], burnt_subsets[[random_rows], y], alpha=0.7, color='red')
        axes[i].set_xlabel(labels[x])
        axes[i].set_ylabel(labels[y])

    plt.tight_layout()
    plt.show()

    return


def scatterplot_all_classes_all_areas(common_folder):
    pixel_features = []
    pixel_labels = []
    for area in ['domokos', 'yliki', 'prodromos', 'delphoi']:
        lma_path = next(common_folder.glob(f'{area}*lma*multiband.tif'))
        label_path = next(common_folder.glob(f'{area}*lma*label.tif'))

        with rio.open(lma_path) as post_ds:
            height = post_ds.height
            width = post_ds.width
            channels = post_ds.count
            post_image = post_ds.read()
            transposed = np.transpose(post_image, (1, 2, 0))
            table_shape = np.reshape(transposed, (height * width, channels))
            pixel_features.append(table_shape)

        with rio.open(label_path) as label_ds:
            height = label_ds.height
            width = label_ds.width
            label_image = label_ds.read(1).flatten()
            pixel_labels.append(label_image)

    concatenated_pixel_features = np.concatenate(pixel_features, 0)
    concatenated_pixel_labels = np.concatenate(pixel_labels, 0)

    all_burnt = concatenated_pixel_features[concatenated_pixel_labels == 1, :]
    all_unburnt = concatenated_pixel_features[concatenated_pixel_labels == 0, :]

    column_combinations = list(combinations(range(4), 2))

    fig, axes = plt.subplots(2, 3)
    axes = axes.ravel()
    labels = ['Green', 'Red', 'Red Edge', 'Near Infared']
    fig.suptitle("Scatterplots between channels with every class")

    number_of_rows_to_select = 10000
    np.random.seed(103)
    burnt_random_rows = np.random.choice(all_burnt.shape[0], number_of_rows_to_select, replace=False)
    unburnt_random_rows = np.random.choice(all_unburnt.shape[0], number_of_rows_to_select, replace=False)

    for i, (x, y) in enumerate(column_combinations):
        axes[i].scatter(all_burnt[[burnt_random_rows], x], all_burnt[[burnt_random_rows], y], alpha=0.8, color='red', label='burnt')
        axes[i].scatter(all_unburnt[[unburnt_random_rows], x], all_unburnt[[unburnt_random_rows], y], alpha=0.7, color='blue', label='unburnt')
        axes[i].set_xlabel(labels[x])
        axes[i].set_ylabel(labels[y])

    plt.legend()
    plt.tight_layout()
    plt.show()

    return


def pca(common_folder):
    pixel_features = []
    pixel_labels = []
    for area in ['domokos', 'yliki', 'prodromos', 'delphoi']:
        lma_path = next(common_folder.glob(f'{area}*lma*multiband.tif'))
        label_path = next(common_folder.glob(f'{area}*lma*label.tif'))

        with rio.open(lma_path) as post_ds:
            height = post_ds.height
            width = post_ds.width
            channels = post_ds.count
            post_image = post_ds.read()
            transposed = np.transpose(post_image, (1, 2, 0))
            table_shape = np.reshape(transposed, (height * width, channels))
            pixel_features.append(table_shape)

        with rio.open(label_path) as label_ds:
            height = label_ds.height
            width = label_ds.width
            label_image = label_ds.read(1).flatten()
            pixel_labels.append(label_image)

        concatenated_pixel_features = np.concatenate(pixel_features, 0)
        concatenated_pixel_labels = np.concatenate(pixel_labels, 0)

        scaler = StandardScaler()
        standardized_image = scaler.fit_transform(concatenated_pixel_features)
        pca = PCA(n_components=2)
        pca_transformed = pca.fit_transform(standardized_image)

        all_burnt_pca = pca_transformed[concatenated_pixel_labels == 1, :]
        all_unburnt_pca = pca_transformed[concatenated_pixel_labels == 0, :]


        fig, axes = plt.subplots(2, 1)
        axes = axes.ravel()
        labels = ['Green', 'Red', 'Red Edge', 'Near Infared']
        fig.suptitle("Scatterplots between channels with every class")

        number_of_rows_to_select = 50000
        np.random.seed(103)
        burnt_random_rows = np.random.choice(all_burnt_pca.shape[0], number_of_rows_to_select, replace=False)
        unburnt_random_rows = np.random.choice(all_unburnt_pca.shape[0], number_of_rows_to_select, replace=False)

        axes[0].scatter(all_burnt_pca[[burnt_random_rows], 0], all_burnt_pca[[burnt_random_rows], 1], alpha=0.8, color='red', label='burnt')
        axes[0].set_title('PCA Burnt Only')
        axes[0].set_xlabel('Principal Component 1')
        axes[0].set_ylabel('Principal Component 2')

        axes[1].scatter(all_burnt_pca[[burnt_random_rows], 0], all_burnt_pca[[burnt_random_rows], 1], alpha=0.8, color='red', label='burnt')
        axes[1].scatter(all_unburnt_pca[[unburnt_random_rows], 0], all_unburnt_pca[[unburnt_random_rows], 1], alpha=0.7, color='blue', label='unburnt')
        axes[1].set_title('PCA all classes')
        axes[1].set_xlabel('Principal Component 1')
        axes[1].set_ylabel('Principal Component 2')

        print(pca.explained_variance_ratio_)

        plt.legend()
        plt.tight_layout()
        plt.show()

        return
