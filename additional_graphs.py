#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from torchviz import make_dot
import torch
import pyjson5
from pathlib import Path

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
    init_loss,
    init_wandb,
)
from training_utilities.dataloaders import Pyrsos_Dataset, Burned_Area_Sampler
from training_utilities.learning_loops import (
    train1epoch,
    eval1epoch,
    wandb_log_metrics
)

from training_utilities.prediction_visualization import convolutional_classifier_visualizer






def plot_logit():
    measurements = np.array([
        [-10, 0],
        [-8, 0],
        [-5, 0],
        [1, 0],
        [2, 0],
        [6, 1],
        [7, 1],
        [7.5, 1],
        [8.0, 1]
    ])

    model = LogisticRegression()
    model.fit(measurements[:, 0].reshape(-1, 1), measurements[:, 1])

    x = np.arange(-11, 11, 0.1).reshape(-1, 1)
    y_hat = model.predict_proba(x)[:, 1]

    plt.scatter(measurements[:, 0], measurements[:, 1])
    plt.plot(x, y_hat )
    plt.show()

    return

def plot_svm():
    x0 = np.random.normal(loc=100, scale=3, size=100)
    y0 = np.random.normal(loc=100, scale=3, size=100)
    labels0 = np.full(100, 0)

    x1 = np.random.normal(loc=120, scale=5, size=100)
    y1 = np.random.normal(loc=80, scale=2, size=100)
    labels1 = np.full(100, 1)

    cluster0 = np.stack((x0, y0), axis=1)
    cluster1 = np.stack((x1, y1), axis=1)

    features = np.concatenate((cluster0, cluster1), axis=0)
    all_labels = np.concatenate((labels0, labels1), axis=0)

    clf = SVC(kernel='linear')
    clf.fit(features, all_labels)

    w = clf.coef_[0]
    b = clf.intercept_[0]

    #Calculate the slope and intercept for the decision boundary line
    slope = -w[0] / w[1]
    intercept = -b / w[1]

    x_vals = np.linspace(70, 140)
    y_vals = slope * x_vals + intercept

    plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')
    plt.scatter(x0, y0, color='red', label='class0')
    plt.scatter(x1, y1, color='blue', label='class1')
    plt.legend()
    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    return

"""
def plot_torchmodel(config_folder):
    general_configs_path = config_folder/'general_config.json'
    configs = read_learning_configs(general_configs_path)

    model_configs_path = config_folder/'method'/ (configs['model'] + '.json')
    model_configs = pyjson5.load(open(model_configs_path,'r'))

    checkpoints_folder, state_dictionaries, starting_epoch = reset_or_continue(configs)
    model_name = configs['model']
    patch_width = configs['patch_width']
    number_of_channels = len(configs['pre_selected_bands'])

    dummy_pre = torch.rand(1, 4, 128, 128)
    dummy_post = torch.rand(1, 4, 128, 128)
    model = init_model(model_name, model_configs, state_dictionaries, patch_width, number_of_channels)

    output = model(dummy_pre, dummy_post)
    params_no_grad = {name: param.clone().detach() for name, param in model.named_parameters()}
    dot = make_dot(output, params=params_no_grad)
    dot.render("model_visualization", format="png", cleanup=True)  # Save as PNG
    return


config_folder = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs')
plot_torchmodel(config_folder)
"""
