#!/usr/bin/python3

'''
This script is used to train a model on the given data. The user must
specify the necessary data paths, the model and its hyperparameters.
'''

import argparse
from pathlib import Path
import pyjson5
from tqdm import tqdm
import copy
import wandb
import pickle
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from pyrsos_utils1 import (
    font_colors,
    resume_or_restart,
    init_model,
    compute_class_weights,
    read_learning_configs
)
from dataset_utils import Dataset, OverSampler
from cd_experiments_utils import (
    train_change_detection,
    eval_change_detection,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_folder', type=Path, default='configs/', required=False,
                        help='The config folder to use. Default "configs/.')

    args = parser.parse_args()


    # Read and validate general configuration file
    general_configs_path = args.config_folder/'general_config'.json
    configs = read_learning_configs(general_configs_path)
    #open the configuration file for the desired model
    model_configs_path = args.config_folder/'method'/f'{configs["method"]}.json'
    model_configs = pyjson5.load(open(model_configs_path,'r'))

    run_path, resume_from_checkpoint, init_epoch = resume_or_restart(configs, model_configs)
    #εδώ αρχίζει το training
    # Print informative message
    if configs['learning_stage'] == 'train':
        if (resume_from_checkpoint is None) or ((isinstance(resume_from_checkpoint, list)) and not any(resume_from_checkpoint)):
            print(f'{font_colors.CYAN}--- Training a new model ---{font_colors.ENDC}')
            print(f'{font_colors.CYAN}--- Model path: {run_path} ---{font_colors.ENDC}')
        else:
            print(f'{font_colors.CYAN}--- Resuming training from {resume_from_checkpoint} ---{font_colors.ENDC}')

            if not configs['resume_training_from_checkpoint?']:
                print(f'{font_colors.CYAN}--- New model path: {run_path} ---{font_colors.ENDC}')



    device = configs['device']


    # Load checkpoint
    checkpoint = None
    if configs['learning_stage'] == 'train' and resume_from_checkpoint is not None:
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)

    # Get data sources and GSDs
    tmp = configs['dataset_type'].split('_')
    # format: "sen2_xx_mod_yy"
    gsd = {tmp[0]: tmp[1], tmp[2]: tmp[3]}

    # Update the configs with the SEN2 or MODIS bands to be used
    data_source = configs['datasets']['data_source']
    for band in configs['datasets']['selected_bands'][data_source].keys():
        configs['datasets']['selected_bands'][data_source][band] = configs['datasets'][f'{data_source}_bands'][gsd[data_source]][band]

    # Compute total number of input channels
    inp_channels = len(configs['datasets']['selected_bands'][data_source])

    # Compute class weights for the specific dataset
    class_weights = compute_class_weights(configs)

    configs['paths']['run_path'] = run_path

    if configs['learning_stage'] == 'train':
        # --- Train model ---

        # Initialize datasets and dataloaders
        train_dataset = Dataset('train', configs)
        val_dataset = Dataset('val', configs, clc=True)

        if isinstance(configs['datasets']['oversampling'], float):
            train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True, sampler=OverSampler(train_dataset, positive_prc=configs['datasets']['oversampling']))
        else:
            train_loader = DataLoader(train_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, num_workers=configs['datasets']['num_workers'], batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True)

        # Get a validation image id for visualization
        validation_id = get_positive_sample('val', val_dataset, configs)

        print(f'{font_colors.CYAN}Using {configs["train"]["loss_function"]} with class weights: {class_weights["train"]}.{font_colors.ENDC}')

        # Begin training
        for rep in range(configs['#training_repetitions']):
            model = init_model(configs, model_configs, checkpoint, inp_channels, device, run_path=run_path, init_epoch=init_epoch)

            # Initialize wandb
            if configs['wandb_activate?'] and (rep == 0):
                wandb = init_wandb(model, run_path, configs, model_configs)


            train_change_detection(model, device, class_weights, run_path, init_epoch, train_loader, val_loader, validation_id,
                                   gsd, checkpoint, configs, model_configs, rep_i, wandb)
'''μέχρι εδώ είναι το training'''
    # --- Test model ---
    print(f'{font_colors.CYAN}--- Testing model for {resume_from_checkpoint} ---{font_colors.ENDC}')

    # Begin testing
    results = {'f1': [], 'iou': [], 'accuracy': [], 'precision': [], 'recall': []}

    # Initialize datasets and dataloaders
    test_dataset = Dataset('test', configs, clc=True)

    validation_id = get_positive_sample('test', test_dataset, configs)

    test_loader = DataLoader(test_dataset, batch_size=configs['datasets']['batch_size'], shuffle=False, pin_memory=True, num_workers=configs['datasets']['num_workers'])

    # Initialize model
    model = init_model(configs, model_configs, None, inp_channels, device, run_path=run_path, init_epoch=init_epoch)

    for rep in range(configs['#training_repetitions']):
        # Initialize wandb
        if (configs['learning_stage'] == 'eval') and configs['wandb']['activate'] and rep == 0:
            wandb = init_wandb(model, run_path, configs, model_configs)

        ckpt_path = run_path / 'checkpoints' / f'{rep_i}' / 'best_segmentation.pt'
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(f'\n{font_colors.CYAN}Loading {ckpt_path}...{font_colors.ENDC}')

        model.load_state_dict(checkpoint['model_state_dict'])
        res = eval_change_detection(model, device, class_weights, init_epoch, test_loader, validation_id,
                                    gsd, 'test', configs, model_configs, rep_i, wandb, run_path)

        for k, v in res.items():
            results[k].append(v)

    # Print final results
    print('\n ===============\n')

    for k, v in results.items():
        if k == 'lc_stats':
            continue
        else:
            print(f'{k} (burnt): {round(np.mean([i[1] for i in v]), 2)} ({round(np.std([i[1] for i in v]), 2)})')
            print(f'{k} (unburnt): {round(np.mean([i[0] for i in v]), 2)} ({round(np.std([i[0] for i in v]), 2)})')

    mean_f1 = np.mean([np.mean([i[0] for i in results['f1']]), np.mean([i[1] for i in results['f1']])])
    mean_f1_std = np.mean([np.std([i[0] for i in results['f1']]), np.std([i[1] for i in results['f1']])])
    print(f'Mean f-score: {round(mean_f1, 2)} ({round(mean_f1_std, 2)})')

    mean_iou = np.mean([np.mean([i[0] for i in results['iou']]), np.mean([i[1] for i in results['iou']])])
    mean_iou_std = np.mean([np.std([i[0] for i in results['iou']]), np.std([i[1] for i in results['iou']])])
    print(f'Mean IoU: {round(mean_iou, 2)} ({round(mean_iou_std, 2)})')
