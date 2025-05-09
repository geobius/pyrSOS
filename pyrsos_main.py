#!/usr/bin/python3

'''
This script is used to train a model on the given data. The user must
specify the necessary data paths, the model and its hyperparameters.
'''

import argparse
from os.path import exists
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

from training_utilities.config_parsers import (
    read_learning_configs,
    font_colors)


from training_utilities.initializers import (
    init_optimizer,
    save_checkpoint,
    init_lr_scheduler,
    reset_or_continue,
    init_model,
    compute_class_weights,
    init_train_loader,
    init_loss,
    init_wandb
)
from training_utilities.dataloaders import Pyrsos_Dataset, Burned_Area_Sampler
from training_utilities.learning_loops import (
    train1epoch,
    eval1epoch,
    wandb_log_metrics
)

from training_utilities.prediction_visualization import convolutional_visualizer


parser = argparse.ArgumentParser()

parser.add_argument('configs_filepath', type=Path, default='configs/convolutional_config.json',
                    required=False,
                    help='The config file to use. Default "configs/convolutional_config.json"')

args = parser.parse_args()

general_configs_filepath = args.configs_filepath
configs = read_learning_configs(general_configs_filepath) #this does checks
#configs = pyjson5.load(open(general_configs_filepath)) #this does not check

model_configs_path = Path(configs['specific_hyperparameters_filepath'])#that way I can have multiple files of hyperparameters
model_configs = pyjson5.load(open(model_configs_path, 'r'))

save_folder, state_dictionaries, starting_epoch = reset_or_continue(configs) #now I can choose where checkpoints are saved

model_name = configs['model']
patch_width = configs['patch_width']
number_of_channels = len(configs['pre_selected_bands'])
loss_function_name = configs['loss_function']

class_weights = compute_class_weights(configs)

save_every = configs['save_every_n_epochs']
last_epoch = starting_epoch + configs['#epochs']

train_dataset = Pyrsos_Dataset('training_set', configs)
val_dataset = Pyrsos_Dataset('validation_set', configs)
test_dataset = Pyrsos_Dataset('testing_set', configs)


if configs['learning_stage'] == 'train':
    train_loader = init_train_loader(train_dataset, configs)
    val_loader = DataLoader(val_dataset, num_workers=configs['#workers'], batch_size=configs['batch_size'], shuffle=True, pin_memory=True)

    print(f'{font_colors.CYAN}Using {configs["loss_function"]} with class weights: {class_weights["training_set"]}.{font_colors.ENDC}')

    model = init_model(model_name, model_configs, state_dictionaries, patch_width, number_of_channels).to(device=configs['device'])
    train_criterion = init_loss(loss_function_name, class_weights['training_set'], model_configs).to(device=configs['device'])
    val_criterion = init_loss(loss_function_name, class_weights['validation_set'], model_configs).to(device=configs['device'])
    optimizer = init_optimizer(model, state_dictionaries, configs, model_configs)
    lr_scheduler = init_lr_scheduler(optimizer, state_dictionaries, configs, model_configs)
    wandb = init_wandb(configs, model_configs)

    best_model = {}
    best_f1_score = 0.0 #in percentage
    best_epoch = 0

    for epoch in range(starting_epoch, last_epoch):
        print(f'=== Epoch: {epoch} ===')
        print('---BackPropagation---')
        train_loss, train_metrics = train1epoch(model, train_loader, train_criterion, optimizer, lr_scheduler, configs['device']) #update the weights
        learning_rate = (lr_scheduler.get_last_lr())[0]
        print(f'training Loss: {train_loss:.6f}')
        wandb_log_metrics(train_loss, train_metrics, learning_rate, epoch, 'training', configs['wandb_activate?'])

        print('---Validating for Overfitting---')
        val_loss, val_metrics = eval1epoch(model, val_loader, val_criterion, configs['device'])  #metrics for overfitting checks.
        print(f'Mean Validation Loss: {val_loss:.6f}')
        wandb_log_metrics(val_loss, val_metrics, learning_rate, epoch, 'validation', configs['wandb_activate?'])

            #this is a saving mechanism for keeping track of the current best state of the model as its weights change.
        f1_score = val_metrics['f1'].item()*100 #in percentage
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_model = model
            best_epoch = epoch

        if (save_every > 0 and epoch % save_every == 0) or (epoch == last_epoch - 1):
            new_checkpoint_path = save_folder / f'best_epoch={best_epoch}.pt'
            save_checkpoint(new_checkpoint_path, val_loss, best_model, optimizer, lr_scheduler)



if configs['learning_stage'] == 'eval':
    #It is assumed that you have already trained the model and have kept the best weights.
    #Now we are checking the metrics across every set to log them in my technical essay.
    #and visualizing the model output masks
    train_loader = init_train_loader(train_dataset, configs)
    val_loader = DataLoader(val_dataset, num_workers=configs['#workers'], batch_size=configs['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, num_workers=configs['#workers'], batch_size=configs['batch_size'], shuffle=True, pin_memory=True)

    train_criterion = init_loss(loss_function_name, class_weights['training_set'], model_configs).to(device=configs['device'])
    val_criterion = init_loss(loss_function_name, class_weights['validation_set'], model_configs).to(device=configs['device'])
    test_criterion = init_loss(loss_function_name, class_weights['testing_set'], model_configs).to(device=configs['device'])
    model = init_model(model_name, model_configs, state_dictionaries, patch_width, number_of_channels).to(device=configs['device'])

    #training_loss, training_metrics = eval1epoch(model, train_loader, train_criterion, configs['device'])  #metrics for overfitting checks.
    #validation_loss, validation_metrics = eval1epoch(model, val_loader, val_criterion, configs['device'])  #metrics for overfitting checks.
    #test_loss, test_metrics = eval1epoch(model, test_loader, test_criterion, configs['device'])  #metrics for overfitting checks.

    #print(f'Mean Training Loss: {training_loss:.6f} metrics: {training_metrics}')
    #print(f'Mean Validation Loss: {validation_loss:.6f} metrics: {validation_metrics}')
    #print(f'Mean Testing Loss: {test_loss:.6f} metrics: {test_metrics}')
    vis = convolutional_visualizer(model, configs)

    #I am too bored to manually press right and left buttons so I send the event
    #programmaticaly
    vis.on_print(2500)
