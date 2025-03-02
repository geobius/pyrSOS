#!/usr/bin/bash

#I am bored of constantly having to manually type in the terminal my commands from dataset_cleaners
#for each subfolder in source_folderpath
#call lma radiometry fixer on the lma image
#call sen2 radiometry fixer on the sen2 pre image
#call sen2 radiometry fixer on the sen2 post_image
#call mask rasterizer for the vector file
#save all the results in the destination folder
#go to the destination folder
#for each subfolder 
#call statistics extractor for each image. Maybe I should remake statistics_extractor.py to work on an entire directory
#patch cutter and dataset splitter should not be called

cd /mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos

mkdir pyrsos_250cm_dataset
mkdir pyrsos_250cm_dataset/delphoi
mkdir pyrsos_250cm_dataset/domokos
mkdir pyrsos_250cm_dataset/prodromos
mkdir pyrsos_250cm_dataset/yliki


lma_radiometry_fixer.py ./raw_data/delphoi/lma.tif ./pyrsos_250cm_dataset/delphoi/delphoi_lma_post_250cm_multiband.tif 2.5
lma_radiometry_fixer.py ./raw_data/domokos/lma.tif ./pyrsos_250cm_dataset/domokos/domokos_lma_post_250cm_multiband.tif 2.5
lma_radiometry_fixer.py ./raw_data/prodromos/lma.tif ./pyrsos_250cm_dataset/prodromos/prodromos_lma_post_250cm_multiband.tif 2.5
lma_radiometry_fixer.py ./raw_data/yliki/lma.tif ./pyrsos_250cm_dataset/yliki/yliki_lma_post_250cm_multiband.tif 2.5


sen2_radiometry_fixer.py ./pyrsos_250cm_dataset/delphoi/delphoi_lma_post_250cm_multiband.tif ./raw_data/delphoi/S2A_MSIL2A_20220622T092041_N0400_R093_T34SFH_20220622T135920.SAFE ./pyrsos_250cm_dataset/delphoi/delphoi_sen2_pre_250cm_multiband.tif
sen2_radiometry_fixer.py ./pyrsos_250cm_dataset/delphoi/delphoi_lma_post_250cm_multiband.tif ./raw_data/delphoi/S2B_MSIL2A_20230413T092029_N0509_R093_T34SFH_20230413T125012.SAFE ./pyrsos_250cm_dataset/delphoi/delphoi_sen2_post_250cm_multiband.tif

sen2_radiometry_fixer.py ./pyrsos_250cm_dataset/domokos/domokos_lma_post_250cm_multiband.tif ./raw_data/domokos/S2B_MSIL2A_20220717T091559_N0400_R093_T34SFJ_20220717T105427.SAFE ./pyrsos_250cm_dataset/domokos/domokos_sen2_pre_250cm_multiband.tif
sen2_radiometry_fixer.py ./pyrsos_250cm_dataset/domokos/domokos_lma_post_250cm_multiband.tif ./raw_data/domokos/S2B_MSIL2A_20230413T092029_N0509_R093_T34SFJ_20230413T125012.SAFE ./pyrsos_250cm_dataset/domokos/domokos_sen2_post_250cm_multiband.tif

sen2_radiometry_fixer.py ./pyrsos_250cm_dataset/prodromos/prodromos_lma_post_250cm_multiband.tif ./raw_data/prodromos/S2B_MSIL2A_20230722T091559_N0509_R093_T34SFH_20230722T105251.SAFE ./pyrsos_250cm_dataset/prodromos/prodromos_sen2_pre_250cm_multiband.tif
sen2_radiometry_fixer.py ./pyrsos_250cm_dataset/prodromos/prodromos_lma_post_250cm_multiband.tif ./raw_data/prodromos/S2B_MSIL2A_20231010T091819_N0509_R093_T34SFH_20231010T124013.SAFE ./pyrsos_250cm_dataset/prodromos/prodromos_sen2_post_250cm_multiband.tif

sen2_radiometry_fixer.py ./pyrsos_250cm_dataset/yliki/yliki_lma_post_250cm_multiband.tif ./raw_data/yliki/S2B_MSIL2A_20230722T091559_N0509_R093_T34SFH_20230722T105251.SAFE ./pyrsos_250cm_dataset/yliki/yliki_sen2_pre_250cm_multiband.tif
sen2_radiometry_fixer.py ./pyrsos_250cm_dataset/yliki/yliki_lma_post_250cm_multiband.tif ./raw_data/yliki/S2B_MSIL2A_20231010T091819_N0509_R093_T34SFH_20231010T124013.SAFE ./pyrsos_250cm_dataset/yliki/yliki_sen2_post_250cm_multiband.tif


mask_rasterizer.py ./pyrsos_250cm_dataset/delphoi/delphoi_lma_post_250cm_multiband.tif ./raw_data/delphoi/delphoi_burned_areas.gpkg ./pyrsos_250cm_dataset/delphoi/delphoi_lma_post_250cm_label.tif
mask_rasterizer.py ./pyrsos_250cm_dataset/domokos/domokos_lma_post_250cm_multiband.tif ./raw_data/domokos/domokos_burned_areas.gpkg ./pyrsos_250cm_dataset/domokos/domokos_lma_post_250cm_label.tif
mask_rasterizer.py ./pyrsos_250cm_dataset/prodromos/prodromos_lma_post_250cm_multiband.tif ./raw_data/prodromos/prodromos_burned_areas.gpkg ./pyrsos_250cm_dataset/prodromos/prodromos_lma_post_250cm_label.tif
mask_rasterizer.py ./pyrsos_250cm_dataset/yliki/yliki_lma_post_250cm_multiband.tif ./raw_data/yliki/yliki_burned_areas.gpkg ./pyrsos_250cm_dataset/yliki/yliki_lma_post_250cm_label.tif

