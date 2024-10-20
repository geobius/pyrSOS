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

from training_utilities.config_parsers import read_learning_configs


from training_utilities.initializers import (
    possibly_create_checkpoints_folder,
    font_colors,
    parse_epoch_from_checkpoint_filename,
    resume_or_restart,
    init_model,
    compute_class_weights,
    save_checkpoint,
    load_checkpoint
)
from training_utilities.dataloaders import Pyrsos_Dataset
from training_utilities.learning_loops import (
    load_checkpoint,
    train_change_detection,
    #eval_change_detection,
)


parser = argparse.ArgumentParser()

parser.add_argument('--config_folder', type=Path, default='configs/', required=False,
                        help='The config folder to use. Default "configs/.')

args = parser.parse_args()

general_configs_path = args.config_folder/'general_config.json'
configs = read_learning_configs(general_configs_path)

model_configs_path = args.config_folder/'method'/f'{configs["method"]}.json'
model_configs = pyjson5.load(open(model_configs_path,'r'))

new_checkpoints_folder = init_new_checkpoints_folder(configs) # according to some setiings it will not generate a new folder
old_checkpoint = init_old_checkpoint(configs) #according to some settings it will not return a state dictionary
first_epoch = init_epoch(configs) #according to some settings it could be zero

device = configs['device']
number_of_channels = len(configs['pre_selected_bands'])
class_weights = compute_class_weights(configs)

save_every = configs['save_every_n_epochs']
save_last_epoch = configs['save_last_epoch']
if configs['mixed_precision?']:
    scaler = torch.cuda.amp.GradScaler()

last_epoch = first_epoch + configs['#epochs']
print_frequency = configs['print_frequency']


if configs['learning_stage'] == 'train':
    train_dataset = Pyrsos_Dataset('train', configs)
    val_dataset = Pyrsos_Dataset('val', configs)

    train_loader = DataLoader(train_dataset, num_workers=configs['#workers'], batch_size=configs['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, num_workers=configs['#workers'], batch_size=configs['batch_size'], shuffle=True, pin_memory=True)

    print(f'{font_colors.CYAN}Using {configs["loss_function"]} with class weights: {class_weights["train"]}.{font_colors.ENDC}')

    for rep in range(configs['#training_repetitions']):
        print(f'\n===== REP {rep_i} =====\n')
        for epoch in range(init_epoch, last_epoch):
            print(f'=== Epoch: {epoch} ===')
            model = init_model(configs, model_configs, checkpoint, number_of_channels, device, run_path=run_path, init_epoch=init_epoch)
            criterion = init_loss(configs, 'train', device, class_weights, model_configs=model_configs)
            optimizer = init_optimizer(model, checkpoint, configs, model_configs)
            lr_scheduler = init_lr_scheduler(optimizer, checkpoint, configs, model_configs)
            wandb = init_wandb(model, run_path, configs, model_configs)

            print('---BackPropagation---')
            model = train1epoch(model, train_loader, criterion, optimizer, lr_scheduler, print_frequency) #update the weights
            learning_rate = (lr_scheduler.get_last_lr())[0]

            print('---Validating for Underfitting---')
            train_loss, train_metrics = eval1epoch(model. train_loader, criterion, device, print_frequency)  #metrics for underfitting checks.
            print(f'Mean Train Loss: {train_loss:.6f}')
            wandb_log_metrics(train_loss, train_metrics, learning_rate, epoch, rep_i, 'train')

            print('---Validating for Overfitting---')
            val_loss, val_metrics = eval1epoch(model, val_loader, criterion, device, print_frequency)  #metrics for overfitting checks.
            print(f'Mean Validation Loss: {val_loss:.6f}')
            wandb_log_metrics(val_loss, val_metrics, learning_rate, epoch, rep_i, 'validation')


            if (save_every != -1) and ((epoch >= save_last_epoch) or (epoch % save_every == 0)):
                save_checkpoint() #in case I want multiple checkpoints during a repetition

        save_checkpoint() # Save at least once at the end of this repetition




if configs['learning_stage'] == 'test':
    pass


