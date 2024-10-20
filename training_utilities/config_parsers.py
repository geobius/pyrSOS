#!/usr/bin/python3

from pathlib import Path
import pyjson5
import torch


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

    assert isinstance(configs['save_every_n_epochs'], int), \
        f"""{font_colors.RED}Error: save_every_n_epochs does not have a valid value.{font_colors.ENDC}
        A valid save_every_n_epochs value must be an integer"""

    assert isinstance(configs['save_last_epoch'], int), \
        f"""{font_colors.RED}Error: save_last_epoch does not have a valid value.{font_colors.ENDC}
        A valid save_last_epoch value must be an integer"""

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

    valid_pre_data_sources = ["sen2_pre"]
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
    assert isinstance(sel0, list) and all(isinstance(i, int) for i in sel0), \
        f"""{font_colors.RED}Error: pre_selected_bands does not have a valid value.{font_colors.ENDC}
        A valid pre_selected_bands value must be a list of integers. Where each integer represents the index of the desired band in the pre image.
        Indexing starts from 1"""

    valid_post_data_sources = ["sen2_post", "lma"]
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
    assert isinstance(sel1, list) and all(isinstance(i, int) for i in sel1), \
        f"""{font_colors.RED}Error: post_selected_bands does not have a valid value.{font_colors.ENDC}
        A valid post_selected_bands value must be a list of integers. Where each integer represents the index of the desired band in the post image.
        Indexing starts from 1"""

    return configs
