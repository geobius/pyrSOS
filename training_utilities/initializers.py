from pathlib import Path
from datetime import datetime
import pickle
from numpy import average
import torch
import torch.nn as nn
from torch.utils import checkpoint
from torchmetrics import MetricCollection
from itertools import chain
import rasterio as rio
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassJaccardIndex, MulticlassPrecision, MulticlassRecall

from config_parsers import font_colors

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



def save_checkpoint(checkpoint_path, loss, model, optimizer, lr_scheduler):

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'loss': loss},
               checkpoint_path)

    return checkpoint_path

def load_checkpoint(checkpoint_path):
    if checkpoint_path.exists():
        state_dictionaries = torch.load(checkpoint_path)
        return state_dictionaries
    else:
        return {}

def parse_epoch_from_checkpoint_filename(checkpoint_path):
    epoch = 0
    if checkpoint_path.exists():
        epoch = int(checkpoint_path.stem.split('epoch =')[1]) + 1
    return epoch

def name_checkpoint(parent_folder, epoch):
    path = parent_folder / f'checkpoint_epoch={epoch}.pt'
    return path



def init_new_checkpoints_folder(configs):
    checkpoints_path = None
    if not (configs['resume_training_from_checkpoint']) and configs['learning_stage'] == 'test':
        run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoints_path = configs['results_path'] / configs['model'] / run_timestamp / 'checkpoints'
        checkpoints_path.mkdir(exist_ok=True, parents=True)

    return checkpoints_path

def init_epoch(configs):
    epoch = 0
    if



def init_model(configs, model_configs, checkpoint, inp_channels, device):
    '''initiate the appropriate model, send it to the device and load a state dictionary'''
    model = None
    match configs['model']:
        case 'fc_ef_conc':
            model = FC_EF_conc(input_nbr=inp_channels, label_nbr=2)
        case 'fc_ef_diff':
            model = FC_EF_diff(input_nbr=inp_channels, label_nbr=2)
        case 'unet':
            model = Unet(input_nbr=inp_channels, label_nbr=2)
        case 'adhr_cdnet':
            model = ADHR(in_channels=inp_channels, num_classes=2)
        case 'snunet':
            model = SNUNet_ECAM(inp_channels, 2, base_channel=model_configs['base_channel'])
        case 'hfanet':
            model = HFANet(input_channel=inp_channels, input_size=configs['patch_width'], num_classes=2)
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
                img_dim=configs['patch_width'],
                in_channels=inp_channels,
                out_channels=model_configs['out_channels'],
                head_num=model_configs['head_num'],
                mlp_dim=model_configs['mlp_dim'],
                block_num=model_configs['block_num'],
                patch_dim=model_configs['patch_dim'],
                class_num=2,
                siamese=model_configs['siamese'])

    model = model.module.to(device) if isinstance(model, nn.DataParallel) else model.to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


def compute_class_weights(configs):
    '''
    Computes the number of pixels per class (burnt/unburnt), then computes the weights of each class
    based on these counts and returns the weights.
    '''
    if configs['weighted_loss']:
        patch_width = configs['patch_width']
        patch_height = configs['patch_height']

        burnt = {'train': 0, 'val': 0, 'test': 0}
        unburnt = {'train': 0, 'val': 0, 'test': 0}

        for mode in ['train', 'val', 'test']:
            pickle_path = Path(configs['dataset_path']) /configs[mode],
            pickle_file = pickle.load(open(pickle_path, 'rb'))
            all_patches_all_areas = chain(pickle_file.values)

            for patch in all_patches_all_areas:
                if 'positive' in patch['label']:
                    with rio.open(patch['label']):
                        mask_band = rio.read(1).flatten()
                        unburnt[mode] += sum(mask_band == 0) #count zeros
                        burnt[mode] += patch_width * patch_height - unburnt #the amount of remaining pixels

                else:
                    unburnt[mode] += (patch_width * patch_height)

        return {
            'train': ((burnt['train'] + unburnt['train']) / (2 * unburnt['train']),
                      (burnt['train'] + unburnt['train']) / (2 * burnt['train'])),
            'val': ((burnt['val'] + unburnt['val']) / (2 * unburnt['val']),
                    (burnt['val'] + unburnt['val']) / (2 * burnt['val'])),
            'test': ((burnt['test'] + unburnt['test']) / (2 * unburnt['test']),
                     (burnt['test'] + unburnt['test']) / (2 * burnt['test']))
        }
    else:
        return {
            'train': (1, 1),
            'val': (1, 1),
            'test': (1, 1)
        }


def init_metrics():

    pyrsos_metrics = MetricCollection({
        "accuracy": MulticlassAccuracy(num_classes=3, ignore_index=2, average=None),
        "precision": MulticlassPrecision(num_classes=3, ignore_index=2, average=None),
        "recall": MulticlassRecall(num_classes=3, ignore_index=2, average=None),
        "f1": MulticlassF1Score(num_classes=3, ignore_index=2, average=None),
        "iou": MulticlassJaccardIndex(num_classes=3, ignore_index=2, average=None)
    })

    return pyrsos_metrics




def init_loss(configs, mode, device, class_weights, model_configs):

    match configs['loss_function']:
        case 'cross_entropy':
            return nn.CrossEntropyLoss(weight=torch.Tensor(class_weights[mode]), ignore_index=2).to(device)
        case 'focal':
            return torch.hub.load(
                'adeelh/pytorch-multi-class-focal-loss',
                model='FocalLoss',
                alpha=torch.Tensor(class_weights[mode]),
                gamma=2,
                reduction='mean',
                force_reload=False,
                ignore_index=2).to(device)
        case 'dice':
            if ('activation' in model_configs.keys()) and (model_configs['activation'] is not None):
                use_softmax = False
            else:
                use_softmax = True
            return DiceLoss(ignore_index=2, use_softmax=use_softmax).to(device)

        case 'dice+ce':
            if ('activation' in model_configs.keys()) and (model_configs['activation'] is not None):
                use_softmax = False
            else:
                use_softmax = True
            return BCEandDiceLoss(weights=torch.Tensor(class_weights[mode]), ignore_index=2, use_softmax=use_softmax).to(device)

        case _:
            raise NotImplementedError(f'Loss {configs["train"]["loss_function"]} is not implemented!')


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
    if checkpoint is not None:
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
    if checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    return lr_scheduler

