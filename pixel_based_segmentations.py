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


#from sklearnex import patch_sklearn
#patch_sklearn()

from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
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


pixel_configs = pyjson5.load(open(Path('/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/python_scripts/configs/pixel_config_lma.json')))

x_train, y_train = load_dataset_as_table('training_set', pixel_configs)
x_val, y_val = load_dataset_as_table('validation_set', pixel_configs)
x_test, y_test = load_dataset_as_table('testing_set', pixel_configs)

actual_x_train, actual_y_train = make_balanced_subset(x_train, y_train, 1000000, 29)

svc_hyperparameters = {
    'gamma': 1.0,
    'kernel': 'linear'
}


tree_hyperparameters = {
    'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 16,
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
    'n_estimators': 101,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 16,
    'random_state': 5385}




xgboost_hyperparameters = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'base_score': 0.5,
    'tree_method': 'hist',
    'verbose': 1,
    'n_estimators': 50,
    'max_depth': 16,
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
        y_estimated = np.all(comparisons, 1).astype(np.uint8)
        return y_estimated


class AllburntClassifier():
    def __init__(self):
        pass
    def fit(self, x, y_true):
        pass
    def predict(self, x):
        y_estimated = np.ones(x.shape[0]).astype(np.uint8)
        return y_estimated

    
def pick_model(name,actual_x_train, actual_y_train):
    model = []
    match (name):
        case 'log':
            model = LogisticRegression()
        case 'svm':
            #model = SGDClassifier(loss='hinge', max_iter=80, verbose=1, shuffle=True, random_state=549)
            #model = SVC(**svc_hyperparameters)
            model = LinearSVC(random_state=0)
        case 'rectangle':
            model = BinaryRectangleClassifier(kappa)
        case 'tree':
            model = DecisionTreeClassifier(**tree_hyperparameters)
        case 'forest':
            model = RandomForestClassifier(**forest_hyperparameters)
        case 'xgb':
            model = XGBClassifier(**xgboost_hyperparameters)
        case 'allburnt':
            model = AllburntClassifier()
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
    trained_model = pick_model(model_name, actual_x_train, actual_y_train)
    train_predictions = trained_model.predict(actual_x_train)
    val_predictions = trained_model.predict(x_val)
    test_predictions = trained_model.predict(x_test)

    prec = [0, 0, 0]
    rec = [0, 0, 0]
    f1 = [0, 0, 0]
    jc = [0, 0, 0]

    for i, (estimated, manual) in enumerate(zip([train_predictions, val_predictions, test_predictions],
                                  [actual_y_train, y_val, y_test])):

        prec[i] = precision_score(manual, estimated)*100
        rec[i] = recall_score(manual, estimated)*100
        jc[i] = jaccard_score(manual, estimated)*100
        f1[i] = f1_score(manual, estimated)*100

    print('precision recall f1 IOU ')
    for i in range(3):
        print(f"{prec[i]:.2f} & {rec[i]:.2f} & {f1[i]:.2f} & {jc[i]:.2f} &")


    return trained_model


#trained_model = run('log')
#trained_model = run('svm')
#trained_model = run('tree')
#trained_model = run('forest')
trained_model = run('xgb')
#trained_model = run('allburnt')
#trained_model = run('rectangle')

vis = pixel_visualizer(trained_model, pixel_configs)

#check = np.zeros((1, 4))
