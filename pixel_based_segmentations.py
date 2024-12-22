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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import plot_tree


from training_utilities.dataloaders import load_lma_as_pixels
from training_utilities.prediction_visualization import lma_pixel_classifier_visualizer





train_pickle = '/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination/TrainEvents_v1.pkl'
val_pickle = '/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination/ValidationEvents_v1.pkl'
test_pickle = '/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination/TestingEvents_v1.pkl'

x_train, y_train = load_lma_as_pixels(train_pickle)
x_val, y_val = load_lma_as_pixels(val_pickle)
x_test, y_test = load_lma_as_pixels(test_pickle)



tree_hyperparameters = {
    'ccp_alpha': 0.0,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 10,
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
    'n_estimators': 20,
    'class_weight': None,
    'criterion': 'gini',
    'max_depth': 10,
    'random_state': 5385}




xgboost_hyperparameters = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'base_score': 0.5,
    'tree_method': 'hist',
    'early_stopping_rounds': 10,
    'verbose': 1,
    'n_estimators': 6,
    'max_depth': 30,
    'learning_rate': 0.3,
    'scale_pos_weight': 1
}



def pick_model(name):
    model = []
    match (name):
        case 'log':
            model = LogisticRegression()
            model.fit(x_train, y_train)
        case 'svm':
            model = SGDClassifier(loss='hinge', max_iter=80, verbose=1, shuffle=True, random_state=549)
            model.fit(x_train, y_train)
        case 'tree':
            model = DecisionTreeClassifier(**tree_hyperparameters)
            model.fit(x_train, y_train)
        case 'forest':
            model = RandomForestClassifier(**forest_hyperparameters)
            model.fit(x_train, y_train)
        case 'xgb':
            model = XGBClassifier(**xgboost_hyperparameters)
            model.fit(x_train, y_train,
                      eval_set=[(x_train, y_train), (x_val, y_val)])
        case 'mlp':
            model = MLPClassifier(solver='sgd',
                                  activation='relu',
                                  batch_size=10000,
                                  max_iter=25,
                                  verbose=True,
                                  hidden_layer_sizes=(3, 2))
            model.fit(x_train, y_train)

    return model



def run(model_name):
    trained_model = pick_model(model_name)
    train_predictions = trained_model.predict(x_train)
    val_predictions = trained_model.predict(x_val)
    test_predictions = trained_model.predict(x_test)

    prec = [0, 0, 0]
    rec = [0, 0, 0]
    jc = [0, 0, 0]

    for i, (estimated, manual) in enumerate(zip([train_predictions, val_predictions, test_predictions],
                                  [y_train, y_val, y_test])):

        prec[i] = precision_score(estimated, manual)*100
        rec[i] = recall_score(estimated, manual)*100
        jc[i] = jaccard_score(estimated, manual)*100

    print('precision recall IOU ')
    for i in range(3):
        print(f"{prec[i]:.2f} & {rec[i]:.2f} & {jc[i]:.2f} &")


    return trained_model


#trained_log = run('log')
#trained_svm = run('svm')
#trained_tree = run('tree')
#trained_forest = run('forest')
trained_xgb = run('xgb')

#vis = lma_pixel_classifier_visualizer(trained_tree, val_pickle)
