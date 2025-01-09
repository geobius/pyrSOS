#!/usr/bin/python3
from pathlib import Path
import pickle
from random import shuffle
from matplotlib.colors import ListedColormap
import pyjson5
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from itertools import product


from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import plot_tree


from training_utilities.dataloaders import load_dataset_as_table
from training_utilities.prediction_visualization import pixel_visualizer


def make_balanced_subset(x, y, size, seed):
    np.random.seed(seed)

    all_burnt_row_indices = np.where(y == 1)[0]
    some_burnt_row_indices = np.random.choice(all_burnt_row_indices, size // 2, replace=False)
    subset_x_burnt = x[some_burnt_row_indices]
    subset_y_burnt = y[some_burnt_row_indices]

    all_unburnt_row_indices = np.where(y == 0)[0]
    some_unburnt_row_indices = np.random.choice(all_unburnt_row_indices, size // 2, replace=False)
    subset_x_unburnt = x[some_unburnt_row_indices]
    subset_y_unburnt = y[some_unburnt_row_indices]

    subset_x_balanced = np.concatenate([subset_x_burnt, subset_x_unburnt], 0)
    subset_y_balanced = np.concatenate([subset_y_burnt, subset_y_unburnt], 0)

    return subset_x_balanced, subset_y_balanced


pixel_configs_path = Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs/pixel_config.json')

x_train, y_train = load_dataset_as_table('training set', pixel_configs_path)
x_val, y_val = load_dataset_as_table('validation set', pixel_configs_path)
x_test, y_test = load_dataset_as_table('testing set', pixel_configs_path)

actual_x_train, actual_y_train = make_balanced_subset(x_train, y_train, 300000, 29)


svc_hyperparameters = {
    'kernel': 'linear',
    'gamma': 1.0
}

tree_hyperparameters = {
    'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 2,
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'monotonic_cst': None,
    'random_state': None,
    'splitter': 'best'}


forest_hyperparameters = {
    'n_estimators': 3,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 8,
    'random_state': 5385}




xgboost_hyperparameters = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'base_score': 0.5,
    'tree_method': 'hist',
    'early_stopping_rounds': 10,
    'verbose': 1,
    'n_estimators': 30,
    'max_depth': 8,
    'learning_rate': 0.3,
    'scale_pos_weight': 1
}

kappa = 2


class BinaryRectangleClassifier():
    def __init__(self, kappa=1.0):
        self.kappa = kappa
        self.means = []
        self.deviations = []

    def fit(self, x, y_true):
        burnt_pixels = x[y_true == 1]
        self.means = np.mean(burnt_pixels, 0)
        self.deviations = np.std(burnt_pixels, 0)

    def predict(self, x):
        lows = self.means - self.kappa * self.deviations
        highs = self.means + self.kappa * self.deviations
        comparisons = np.logical_and(x >= lows, x <= highs)
        y_estimated = np.all(comparisons, 1).astype(int)
        return y_estimated


def pick_model(name):
    model = []
    match (name):
        case 'log':
            model = LogisticRegression()
            model.fit(actual_x_train, actual_y_train)
        case 'svm':
            #model = SGDClassifier(loss='hinge', max_iter=80, verbose=1, shuffle=True, random_state=549)
            #model = SVC(**svc_hyperparameters)
            model = LinearSVC(random_state=0)
            model.fit(actual_x_train, actual_y_train)
        case 'rectangle':
            model = BinaryRectangleClassifier(kappa)
            model.fit(actual_x_train, actual_y_train)
        case 'tree':
            model = DecisionTreeClassifier(**tree_hyperparameters)
            model.fit(actual_x_train, actual_y_train)
        case 'forest':
            model = RandomForestClassifier(**forest_hyperparameters)
            model.fit(actual_x_train, actual_y_train)
        case 'xgb':
            model = XGBClassifier(**xgboost_hyperparameters)
            model.fit(actual_x_train, actual_y_train,
                      eval_set=[(actual_x_train, actual_y_train), (x_val, y_val)])
        case 'mlp':
            model = MLPClassifier(solver='sgd',
                                  activation='relu',
                                  batch_size=10000,
                                  max_iter=25,
                                  verbose=True,
                                  hidden_layer_sizes=(3, 2))
            model.fit(actual_x_train, actual_y_train)

    return model



def run(model_name):
    trained_model = pick_model(model_name)
    train_predictions = trained_model.predict(actual_x_train)
    val_predictions = trained_model.predict(x_val)
    test_predictions = trained_model.predict(x_test)

    prec = [0, 0, 0]
    rec = [0, 0, 0]
    jc = [0, 0, 0]

    for i, (estimated, manual) in enumerate(zip([train_predictions, val_predictions, test_predictions],
                                  [actual_y_train, y_val, y_test])):

        prec[i] = precision_score(manual, estimated)*100
        rec[i] = recall_score(manual, estimated)*100
        jc[i] = jaccard_score(manual, estimated)*100

    print('precision recall IOU ')
    for i in range(3):
        print(f"{prec[i]:.2f} & {rec[i]:.2f} & {jc[i]:.2f} &")


    return trained_model


#trained_log = run('log')
#trained_svm = run('svm')
#trained_tree = run('tree')
#trained_forest = run('forest')
#trained_xgb = run('xgb')
trained_rect = run('rectangle')

vis = pixel_visualizer('training set', pixel_configs_path, trained_rect) #use only the testing set

check = np.zeros((1, 4))
