from pathlib import Path
from datetime import datetime
import pickle
from numpy import average
import torch
import torch.nn as nn
from torch.utils import checkpoint
from itertools import chain
import rasterio as rio
import wandb
from torch.utils.data import DataLoader
import pyjson5

from models.fc_ef_conc import FC_EF_conc
from models.fc_ef_diff import FC_EF_diff
from models.unet import Unet
from models.snunet import SNUNet_ECAM
from models.hfanet import HFANet
from models.changeformer import ChangeFormerV6
from models.bit_cd import define_G
from models.adhr_cdnet import ADHR
from models.transunet_cd import TransUNet_CD
from models.bam_cd.model import BAM_CD

from losses.dice import DiceLoss
from losses.bce_and_dice import BCEandDiceLoss

from training_utilities.dataloaders import Burned_Area_Sampler

def save_checkpoint(checkpoint_path, loss, model, optimizer, lr_scheduler):

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss},
               checkpoint_path)

    return checkpoint_path

def load_checkpoint(checkpoint_path):

    if checkpoint_path.is_file():
        state_dictionaries = torch.load(checkpoint_path)
        return state_dictionaries
    else:
        return {}

def parse_epoch_from_checkpoint_filename(checkpoint_path):
    epoch = int(checkpoint_path.stem.split('epoch=')[1]) + 1
    return epoch


def reset_or_continue(configs):
    #return a path, the state dictionaries and an epoch value
    save_folder = Path(configs['save_state_folderpath'])
    save_folder.mkdir(exist_ok=True, parents=True)

    if configs['load_state_filepath'] is None:
        return save_folder, {}, 0
    else:
        checkpoint_filepath = Path(configs['load_state_filepath'])
        state_dictionaries = load_checkpoint(checkpoint_filepath)
        starting_epoch = parse_epoch_from_checkpoint_filename(checkpoint_filepath)

        return save_folder, state_dictionaries, starting_epoch


def init_wandb(configs, model_configs):
    all_configs = configs
    all_configs.update({'model_configs': model_configs})

    if configs['wandb_activate?']:
        wandb.init(project=configs['wandb_project'],
                   config=all_configs,
                   reinit=True)

    return

def init_train_loader(train_dataset, configs):
    if configs['use_only_burned_patches?']:
        return DataLoader(train_dataset, num_workers=configs['#workers'], batch_size=configs['batch_size'], sampler=Burned_Area_Sampler(train_dataset), pin_memory=True)
    else:
        return DataLoader(train_dataset, num_workers=configs['#workers'], batch_size=configs['batch_size'], shuffle=True, pin_memory=True)


def init_model(model_name, model_configs, checkpoint, patch_width, inp_channels):
    '''initiate the appropriate model, send it to the device and load a state dictionary'''
    model = None
    match model_name:
        case 'fc_ef_conc':
            model = FC_EF_conc(input_nbr=inp_channels, label_nbr=2)#label_nbr means number of labels
        case 'fc_ef_diff':
            model = FC_EF_diff(input_nbr=inp_channels, label_nbr=2)
        case 'unet':
            model = Unet(n_channels=inp_channels, n_labels=2, depth=model_configs['depth'], fusion_method=model_configs['fusion_method'])
        case 'adhr_cdnet':
            model = ADHR(in_channels=inp_channels, num_classes=2)
        case 'snunet':
            model = SNUNet_ECAM(inp_channels, 2, base_channel=model_configs['base_channel'])
        case 'hfanet':
            model = HFANet(input_channel=inp_channels, input_size=patch_width, num_classes=2)
        case 'bit_cd':
            model = define_G(model_configs, num_classes=2, in_channels=inp_channels)
        case 'bam_cd':
            model = BAM_CD(
                encoder_name=model_configs['backbone'],
                encoder_weights=model_configs['encoder_weights'],
                in_channels=inp_channels,
                classes=2,
                fusion_mode='conc',
                activation=model_configs['activation'],
                siamese=model_configs['siamese'],
                decoder_attention_type=model_configs["decoder_attention_type"],
                decoder_use_batchnorm=model_configs['decoder_use_batchnorm'])
        case 'changeformer':
            model = ChangeFormerV6(
                embed_dim=model_configs['embed_dim'],
                input_nc=inp_channels,
                output_nc=2,
                decoder_softmax=model_configs['decoder_softmax'])
        case 'transunet_cd':
            model = TransUNet_CD(
                img_dim= patch_width,
                in_channels=inp_channels,
                out_channels=model_configs['out_channels'],
                head_num=model_configs['head_num'],
                mlp_dim=model_configs['mlp_dim'],
                block_num=model_configs['block_num'],
                patch_dim=model_configs['patch_dim'],
                class_num=2,
                siamese=model_configs['siamese'])

    if checkpoint != {}:
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


def compute_class_weights(configs):
    '''
    Computes the number of pixels per class (burnt/unburnt), then computes the weights of each class
    based on these counts and returns the weights.
    '''
    if configs['weighted_loss?']:
        patch_width = configs['patch_width']
        patch_height = configs['patch_height']

        burnt = {'training_set': 0, 'validation_set': 0, 'testing_set': 0}
        unburnt = {'training_set': 0, 'validation_set': 0, 'testing_set': 0}

        for mode in ['training_set', 'validation_set', 'testing_set']:
            ds_path = Path(configs['dataset_folderpath'])
            splits = pyjson5.load(open(configs['split_filepath'], 'r'))
            areas_in_the_set = splits[mode]

            label_paths = []
            for area in areas_in_the_set:
                area_folder = ds_path/area
                label_paths.extend(list(area_folder.glob('*label*')))

            for patch_path in label_paths:
                if 'positive' in patch_path.stem:
                    with rio.open(patch_path) as label:
                        mask_band = label.read(1).flatten()
                        patch_unburnt = sum(mask_band == 0) #count zeros
                        patch_burnt = patch_width * patch_height - patch_unburnt #the remaining pixels

                        unburnt[mode] += patch_unburnt
                        burnt[mode] += patch_burnt

                else:
                    unburnt[mode] += (patch_width * patch_height)

        return {
            'training_set': ((burnt['training_set'] + unburnt['training_set']) / (2 * unburnt['training_set']),
                      (burnt['training_set'] + unburnt['training_set']) / (2 * burnt['training_set'])),
            'validation_set': ((burnt['validation_set'] + unburnt['validation_set']) / (2 * unburnt['validation_set']),
                    (burnt['validation_set'] + unburnt['validation_set']) / (2 * burnt['validation_set'])),
            'testing_set': ((burnt['testing_set'] + unburnt['testing_set']) / (2 * unburnt['testing_set']),
                     (burnt['testing_set'] + unburnt['testing_set']) / (2 * burnt['testing_set']))
        }
    else:
        return {
            'training_set': (1, 1),
            'validation_set': (1, 1),
            'testing_set': (1, 1)
        }





def init_loss(function_name, class_weights, model_configs):

    match function_name:
        case 'cross_entropy':
            return nn.CrossEntropyLoss(weight=torch.Tensor(class_weights), ignore_index=2)
        case 'focal':
            return torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                alpha=torch.Tensor(class_weights),
                gamma=2,
                reduction='mean',
                force_reload=False,
                ignore_index=2)
        case 'dice':
            if ('activation' in model_configs.keys()) and (model_configs['activation'] is not None):
                use_softmax = False
            else:
                use_softmax = True
            return DiceLoss(ignore_index=2, use_softmax=use_softmax)

        case 'dice+ce':
            if ('activation' in model_configs.keys()) and (model_configs['activation'] is not None):
                use_softmax = False
            else:
                use_softmax = True
            return BCEandDiceLoss(weights=torch.Tensor(class_weights), ignore_index=2, use_softmax=use_softmax)

        case _:
            raise NotImplementedError(f'Loss {function_name} is not implemented!')


def init_optimizer(model, checkpoint, configs, model_configs, model_name=None):
    '''
    Initialize the optimizer.
    '''
    if model_name is None:
        lr = model_configs['optimizer']['learning_rate']
        optim_args = model_configs['optimizer']
    else:
        lr = model_configs['optimizer'][model_name]['learning_rate']
        optim_args = model_configs['optimizer'][model_name]

    if optim_args['name'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=optim_args['weight_decay'])
    elif optim_args['name'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=optim_args['weight_decay'])
    elif optim_args['name'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=optim_args['weight_decay'], momentum=optim_args['momentum'])
    else:
        raise NotImplementedError(f'Optimizer {optim_args["name"]} is not implemented!')

    # Load checkpoint (if any)
    if checkpoint != {}:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def init_lr_scheduler(optimizer, checkpoint, configs, model_configs, model_name=None):
    # Get the required LR scheduling
    if model_name is not None:
        lr_schedule = model_configs['optimizer'][model_name]['lr_schedule']
        optim_args = model_configs['optimizer'][model_name]
    else:
        lr_schedule = model_configs['optimizer']['lr_schedule']
        optim_args = model_configs['optimizer']

    # Initialize the LR scheduler
    match lr_schedule:
        case 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, optim_args['lr_schedule_steps'])
        case None:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1, last_epoch=-1)
        case 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - epoch / float(configs['#epochs'] + 1)
                return lr_l
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        case 'step':
            if 'lr_scheduler_gamma' in optim_args.keys():
                gamma = optim_args['lr_scheduler_gamma'] = 0.5
            else:
                gamma = 0.1
            if 'lr_scheduler_step' in optim_args.keys():
                step_size = optim_args['lr_scheduler_step']
            else:
                step_size = configs['#epochs'] // 3
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        case _ if lr_schedule.startswith('step_'):
            step_size = int(lr_schedule.split('_')[1])
            if 'lr_scheduler_gamma' in optim_args.keys():
                gamma = optim_args['lr_scheduler_gamma'] = 0.5
            else:
                gamma = 0.1
            if 'lr_scheduler_step' in optim_args.keys():
                step_size = optim_args['lr_scheduler_step']
            else:
                step_size = configs['#epochs'] // 3
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        case _:
            raise NotImplementedError(f'{lr_schedule} LR scheduling is not yet implemented!')

    # Load checkpoint (if any)
    if checkpoint != {}:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    return lr_scheduler
