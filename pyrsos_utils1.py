from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pyjson5
import matplotlib.pyplot as plt
import pickle
import torch
import pdb
import torch.nn as nn
from torchmetrics import ConfusionMatrix, JaccardIndex
from itertools import chain
import rasterio as rio
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


class font_colors:
    '''
    Colors for printing messages to stdout.
    '''
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


class NotSupportedError(Exception):
    def __init__(self, method_name, argument, message=""):
        self.message = f'"{argument}" is not yet supported for {method_name}!'
        super().__init__(self.message)


def read_learning_configs(configs_path):
    '''
    Reads the configs from the path into a dictionary
    and performs basic checks on the validity of the configs file.
    '''
    configs = pyjson5.load(open(configs_path,'r'))
  
    valid_models = ["unet", "fc_ef_diff", "fc_ef_conc", "snunet", "bam_cd", "changeformer", "hfanet", "adhr_cdnet", "transunet_cd", "bit_cd"]
    assert configs['model'] in valid_models, \
        f"""{font_colors.RED}Error: No valid model is provided.{font_colors.ENDC}
        A valid model must be one of the following {font_colors.BLUE}{valid_models}{font_colors.BLUE}"""

    valid_learning_stages = ["train", "eval"]
    assert configs['learning_stage'] in valid_learning_stages, \
        f"""{font_colors.RED}Error: No valid learning_stage is provided.{font_colors.ENDC}
        A valid learning_stage must be one of the following {font_colors.BLUE}{valid_learning_stages}{font_colors.BLUE}"""

    assert isinstance(configs['distributed?'], bool), \
        f"""{font_colors.RED}Error: No valid distributed? is provided.{font_colors.ENDC}
        A valid distributed value must be a boolean"""

    assert isinstance(configs['#workers'], int), \
        f"""{font_colors.RED}Error: #workers does not have a valid value.{font_colors.ENDC}
        A valid value must be an integer"""

    assert isinstance(configs['seed'], int), \
        f"""{font_colors.RED}Error: seed does not have a valid value.{font_colors.ENDC}
        A valid seed must be an integer"""

    valid_devices = ['cpu', 'cuda:0']
    assert configs['device'] in valid_devices, \
        f"""{font_colors.RED}Error: No valid device is provided.{font_colors.ENDC}
        A valid seed must be an integer"""
    if configs['device'] == 'cuda:0' and not torch.cuda.is_available():
        raise Exception("cuda is not supported on this machine. You must change the value of 'device' to 'cpu' inside the configuration file.")

    assert isinstance(configs['#epochs'], int), \
        f"""{font_colors.RED}Error: #epochs does not have a valid value.{font_colors.ENDC}
        A valid #epoch value must be an integer"""

    assert isinstance(configs['#training_repetitions'], int), \
        f"""{font_colors.RED}Error: #training_repetitions does not have a valid value.{font_colors.ENDC}
        A valid #training_repetitions value must be an integer"""

    assert isinstance(configs['#validation_frequency'], int), \
        f"""{font_colors.RED}Error: #validation_frequency does not have a valid value.{font_colors.ENDC}
        A valid #validation_frequency value must be an integer"""

    fr = configs['save_checkpoint_frequency']
    valid_fr = False
    if isinstance(fr, int):
        valid_fr = True
    elif (isinstance(fr, tuple) and len(fr) == 2 and all(isinstance(i, int) for i in fr) and fr[0] < fr[1]):
        valid_fr = True
    assert valid_fr, \
        f"""{font_colors.RED}Error: save_checkpoint_frequency does not have a valid value.{font_colors.ENDC}
        A valid save_checkpoint_frequency value must be an integer or a tuple of two integers,
        where the first is less than the second"""

    assert isinstance(configs['print_frequency'], int), \
        f"""{font_colors.RED}Error: print_frequency does not have a valid value.{font_colors.ENDC}
        A valid print_frequency value must be an integer"""

    assert isinstance(configs['mixed_precision?'], bool), \
        f"""{font_colors.RED}Error: print_frequency does not have a valid value.{font_colors.ENDC}
        A valid print_frequency value must be an integer"""

    valid_loss_functions = ["cross_entropy", "focal", "dice", "dice+ce"]
    assert configs['loss_function'] in valid_loss_functions, \
        f"""{font_colors.RED}Error: No valid loss_function is provided.{font_colors.ENDC}
        A valid loss_function must be one of the following {font_colors.BLUE}{valid_loss_functions}{font_colors.BLUE}"""

    assert isinstance(configs['weighted_loss?'], bool), \
        f"""{font_colors.RED}Error: weighted_loss? does not have a valid value.{font_colors.ENDC}
        A valid weighted_loss? value must be a bool"""

    assert isinstance(configs['resume_training_from_checkpoint?'], bool), \
        f"""{font_colors.RED}Error: resume_training_from_checkpoint? does not have a valid value.{font_colors.ENDC}
        A valid resume_training_from_checkpoint? value must be a boolean"""

    assert isinstance(configs['wandb_project'], str), \
        f"""{font_colors.RED}Error: wandb_project does not have a valid value.{font_colors.ENDC}
        A valid wandb_project value must be a string"""

    assert isinstance(configs['wandb_entity'], str), \
        f"""{font_colors.RED}Error: wandb_entity does not have a valid value.{font_colors.ENDC}
        A valid wandb_entity value must be a string"""

    assert isinstance(configs['wandb_id'], str), \
        f"""{font_colors.RED}Error: wandb_id does not have a valid value.{font_colors.ENDC}
        A valid wandb_id value must be a string"""

    assert isinstance(configs['wandb_resume?'], bool), \
        f"""{font_colors.RED}Error: wandb_resume? does not have a valid value.{font_colors.ENDC}
        A valid wandb_resume? value must be a boolean"""

    assert isinstance(configs['wandb_activate?'], bool), \
        f"""{font_colors.RED}Error: wandb_activate? does not have a valid value.{font_colors.ENDC}
        A valid wandb_activate? value must be a boolean"""

    assert Path(configs['dataset_path']).exists(), \
        f'{font_colors.RED}{font_colors.BOLD}The dataset path ({configs["dataset_path"]}) does not exist!{font_colors.ENDC}'

    assert Path(configs['results_path']).exists(), \
        f'{font_colors.RED}{font_colors.BOLD}The results path ({configs["results_path"]}) does not exist!{font_colors.ENDC}'

    if configs['load_state_path'] is not None:
        assert Path(configs['load_state_path']).exists(), \
        f"""{font_colors.RED}{font_colors.BOLD}The checkpoint path ({configs['load_state_path']}) does not exist!{font_colors.ENDC}"""


    ds_path = Path(configs['dataset_path'])
    assert (ds_path/configs['train']).exists(), \
        f"""cannot find the training pickle file in {ds_path}"""

    assert (ds_path/configs['val']).exists(), \
        f"""cannot find the validation pickle file in {ds_path}"""
   
    assert (ds_path/configs['test']).exists(), \
        f"""cannot find the testing pickle file in {ds_path}"""

    assert isinstance(configs['patch_width'], int), \
        f"""{font_colors.RED}Error: patch_width does not have a valid value.{font_colors.ENDC}
        A valid patch width value must be an integer"""

    assert isinstance(configs['patch_height'], int), \
        f"""{font_colors.RED}Error: patch_height does not have a valid value.{font_colors.ENDC}
        A valid patch height value must be an integer"""

    valid_pre_data_sources = ["sen2"]
    assert configs['pre_data_source'] in valid_pre_data_sources, \
        f"""{font_colors.RED}Error: No valid pre data source is provided.{font_colors.ENDC}
        A valid pre data source must be one of the following {font_colors.BLUE}{valid_pre_data_sources}{font_colors.BLUE}"""

    assert isinstance(configs['pre_normalize?'], bool), \
        f"""{font_colors.RED}Error: pre_normalize? does not have a valid value.{font_colors.ENDC}
        A valid pre_normalize? value must be a boolean"""

    assert isinstance(configs['pre_gsd_meters'], float), \
        f"""{font_colors.RED}Error: pre_gsd_meters does not have a valid value.{font_colors.ENDC}
        A valid pre_gsd_meters value must be a float"""

    sel0 = configs['pre_selected_bands']
    assert isinstance(sel0, list) and all(isinstance(i, str) for i in sel0), \
        f"""{font_colors.RED}Error: pre_selected_bands does not have a valid value.{font_colors.ENDC}
        A valid pre_selected_bands value must be a list of strings. Where each string represents the name of the desired band in the pre image."""

    valid_post_data_sources = ["sen2", "lma"]
    assert configs['post_data_source'] in valid_post_data_sources,  \
        f"""{font_colors.RED}Error: No valid post_data_source is provided.{font_colors.ENDC}
        A valid post_data_source must be one of the following {font_colors.BLUE}{valid_models}{font_colors.BLUE}"""

    assert isinstance(configs['post_normalize?'], bool), \
        f"""{font_colors.RED}Error: post_normalize? does not have a valid value.{font_colors.ENDC}
        A valid post_normalize? value must be a boolean"""

    assert isinstance(configs['post_gsd_meters'], float), \
        f"""{font_colors.RED}Error: post_gsd_meters does not have a valid value.{font_colors.ENDC}
        A valid post_gsd_meters value must be a float"""

    sel1 = configs['post_selected_bands']
    assert isinstance(sel1, list) and all(isinstance(i, str) for i in sel1), \
        f"""{font_colors.RED}Error: post_selected_bands does not have a valid value.{font_colors.ENDC}
        A valid post_selected_bands value must be a list of strings. Where each string represents the name of the desired band in the post image."""

    return configs

def init_model_log_path(configs):
    '''
    Initializes the path to save results for the given model.

    The general form of a model's path is:
        "results/<task_name>/<model_name>/<timestamp>/"
    and its checkpoints are saved in:
        "results/<task_name>/<model_name>/<timestamp>/checkpoints/<ckpt_name>"
    '''

    if configs['learning_stage'] == 'train':
        if configs['resume_training_from_checkpoint?']:
            # Use an existing path to resume training
            assert Path(configs['load_state_path']).exists(), \
                print(f'{font_colors.RED}Error: path {configs["load_state_path"]} does not exist!.{font_colors.ENDC}')

            resume_path = Path(configs['load_state_path'])
            results_path = Path(*resume_path.parts[:-3])
        else:
            # Create a new path
            run_ts = datetime.now().strftime("%Y%m%d%H%M%S")
            results_path = Path(configs['results_path']) / configs['model'] / run_ts
            results_path.mkdir(exist_ok=True, parents=True)

            ckpt_path = results_path / 'checkpoints'
            ckpt_path.mkdir(exist_ok=True, parents=True)
    else:
        # Use an existing path for testing
        assert Path(configs['load_state_path']).exists(), \
            print(f'{font_colors.RED}Error: path {configs["load_state_path"]} does not exist!.{font_colors.ENDC}')

        test_path = Path(configs['load_state_path'])
        results_path = Path(*test_path.parts[:-3])

    return results_path


def resume_or_restart(configs):
    '''
    Checks whether training must resume or start from scratch and returns
    the appropriate training parameters for each case.
    configs and models_configs and model_configs are both dictionaries
    the function returns 2 paths and an integer
     '''
    run_path = init_model_log_path(configs)
    checkpoint_path = None
    init_epoch = 0

    if configs['learning_stage'] == 'train' and configs['resume_training_from_checkpoint?']:
        checkpoint_path = Path(configs['load_state_path'])
        init_epoch = int(checkpoint_path.stem.split('epoch=')[1]) + 1

    return run_path, checkpoint_path, init_epoch


def init_model(configs, model_configs, checkpoint, inp_channels, device, run_path=None, init_epoch=None):
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


class MyConfusionMatrix():
    def __init__(self, num_classes, ignore_index=None, device=None):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.device = device

        if ignore_index is not None:
            self.tp = torch.zeros(num_classes-1).to(torch.long).to(self.device)
            self.fp = torch.zeros(num_classes-1).to(torch.long).to(self.device)
            self.tn = torch.zeros(num_classes-1).to(torch.long).to(self.device)
            self.fn = torch.zeros(num_classes-1).to(torch.long).to(self.device)
        else:
            self.tp = torch.zeros(num_classes).to(torch.long).to(self.device)
            self.fp = torch.zeros(num_classes).to(torch.long).to(self.device)
            self.tn = torch.zeros(num_classes).to(torch.long).to(self.device)
            self.fn = torch.zeros(num_classes).to(torch.long).to(self.device)

        self.cm = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(self.device)
        self._accuracy = 0.
        self._precision = 0.
        self._recall = 0.
        self._f1_score = 0

    def compute(self, preds, target):
        new_cm = self.cm(preds, target)

        if self.ignore_index is not None:
            # Drop row and column corresponding to ignored index
            idx = list(set(range(new_cm.shape[0])) - set([self.ignore_index]))
            new_cm = new_cm[idx, :][:, idx]

        tp = torch.diag(new_cm)
        fp = torch.sum(new_cm, dim=0) - torch.diag(new_cm)
        fn = torch.sum(new_cm, dim=1) - torch.diag(new_cm)

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += torch.sum(new_cm) - (fp + fn + tp)

        self._accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)
        self._precision = self.tp / (self.tp + self.fp)
        self._recall = self.tp / (self.tp + self.fn)
        self._f1_score = (2 * self._precision * self._recall) / (self._precision + self._recall)

        return new_cm

    def accuracy(self):
        return self._accuracy

    def precision(self):
        return self._precision

    def recall(self):
        return self._recall

    def f1_score(self):
        return self._f1_score

    def reset(self):
        if self.ignore_index is not None:
            self.tp = torch.zeros(self.num_classes-1).to(torch.long)
            self.fp = torch.zeros(self.num_classes-1).to(torch.long)
            self.tn = torch.zeros(self.num_classes-1).to(torch.long)
            self.fn = torch.zeros(self.num_classes-1).to(torch.long)
        else:
            self.tp = torch.zeros(self.num_classes).to(torch.long)
            self.fp = torch.zeros(self.num_classes).to(torch.long)
            self.tn = torch.zeros(self.num_classes).to(torch.long)
            self.fn = torch.zeros(self.num_classes).to(torch.long)

        self.cm = ConfusionMatrix(num_classes=self.num_classes).to(self.device)
        self._accuracy = 0.
        self._precision = 0.
        self._recall = 0.
        self._f1_score = 0

def initialize_metrics(configs, device):

    cm = MyConfusionMatrix(num_classes=3, ignore_index=2, device=device)
    iou = JaccardIndex(task='multiclass', num_classes=3, average='none', ignore_index=2).to(device)

    return cm, iou


def create_loss(configs, mode, device, class_weights, model_configs):

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


def get_sample_index_in_batch(batch_size, idx):
    '''
    Takes as input the index of an individual sample (as it is mapped by the Dataset object) and
    calculates the batch index it is contained into, as well as its index inside the batch.
    '''
    batch_idx = (idx // batch_size)
    idx_in_batch = (idx % batch_size)

    return batch_idx, idx_in_batch
