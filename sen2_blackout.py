#!/usr/bin/python3
import numpy as np
import rasterio as rio
import argparse
from pathlib import Path

from rasterio import transform



def blackout(sen2, lma, sen2_band_count, height, width):
    LMA_BLACK = np.full(4, 0)
    SEN2_BLACK = np.full(sen2_band_count, 0)

    corrected_sen2 = np.copy(sen2)
    for row in range(height):
        for column in range(width):
            if np.array_equal(lma[row, column], LMA_BLACK):
                corrected_sen2[row, column] = SEN2_BLACK

    return corrected_sen2


sen2_filename = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/pyrsos_250cm_dataset/delphoi_sen2_pre_250cm_multiband.tif')
lma_filename = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/pyrsos_250cm_dataset/test2.tif')
dst_filename_sen2 = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/pyrsos_250cm_dataset/test3.tif')

sen2 = rio.open(sen2_filename)
lma = rio.open(lma_filename)

sen2_bands = sen2.read()
sen2_bands = np.transpose(sen2_bands, (1, 2, 0))

lma_bands = lma.read()
lma_bands = np.transpose(lma_bands, (1, 2, 0))

new_sen2 = blackout(sen2_bands, lma_bands, sen2.count, sen2.height, sen2.width)
new_sen2 = np.transpose(new_sen2, (2, 0, 1))


new_ds = rio.open(
    dst_filename_sen2,
    'w',
    driver='GTiff',
    width=sen2.width,
    height=sen2.height,
    transform=sen2.transform,
    count=sen2.count,
    crs=sen2.crs,
    dtype=sen2.dtypes[0])

new_ds.write(new_sen2)

new_ds.close()
sen2.close()
lma.close()
