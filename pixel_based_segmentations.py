#!/usr/bin/python3
from pathlib import Path
import pickle
from random import shuffle
from matplotlib.colors import ListedColormap
import pyjson5
import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, jaccard_score, accuracy_score, recall_score
from itertools import product


from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from training_utilities.dataloaders import load_lma_as_pixels
from training_utilities.prediction_visualization import lma_pixel_classifier_visualizer


def evaluate_model(fixed_hyperparameters, criterion, x_train, y_train, x_val, y_val):
    eval_set = [(x_train, y_train), (x_val, y_val)]

    model = XGBClassifier(**fixed_hyperparameters)
    model.fit(x_train, y_train,
              eval_set=eval_set)
    val_outputs = model.predict(x_val)
    val_score = criterion(y_val, val_outputs)

    return val_score


def GridSearch(dictionary_of_possible_hyperparameters, criterion, x_train, y_train, x_val, y_val):
    best_score = 0
    best_hyperparameters = {}

    for params in product(*dictionary_of_possible_hyperparameters.values()):
        param_dict = dict(zip(dictionary_of_possible_hyperparameters.keys(), params))
        val_score = evaluate_model(param_dict, criterion, x_train, y_train, x_val, y_val)

        if val_score > best_score:
            best_score = val_score
            best_hyperparameters = param_dict

    return best_score, best_hyperparameters






"""
hyperparameters = {
    'objective': ['binary:logistic'],
    'eval_metric': ['aucpr'],
    'base_score': [0.5],
    'tree_method': ['hist'],
    'early_stopping_rounds': [3],
    'verbose': [1],
    'n_estimators': [5],
    'max_depth': [4],
    'learning_rate': [0.3]
}

"""


train_pickle = '/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination/TrainEvents_v1.pkl'
val_pickle = '/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination/ValidationEvents_v1.pkl'
test_pickle = '/mnt/7EBA48EEBA48A48D/examhno10/ptyhiakh/pyrsos/destination/TestingEvents_v1.pkl'

x_train, y_train = load_lma_as_pixels(train_pickle)
x_val, y_val = load_lma_as_pixels(val_pickle)
x_test, y_test = load_lma_as_pixels(test_pickle)

#x_train_reduced, y_train_reduced, pca = load_lma_as_pixels(train_pickle, True)
#x_val_reduced, y_val_reduced, pca = load_lma_as_pixels(train_pickle, True)



#best_score, best_hyperparameters = GridSearch(best_hyperparameters, f1_score, x_train, y_train, x_val, y_val)
best_hyperparameters = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'base_score': 0.5,
    'tree_method': 'hist',
    'early_stopping_rounds': 10,
    'verbose': 1,
    'n_estimators': 75,
    'max_depth': 6,
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
            model = SGDClassifier(loss='hinge', max_iter=50, verbose=1, shuffle=True)
            model.fit(x_train, y_train)
        case 'tree':
            model = DecisionTreeClassifier()
            model.fit(x_train, y_train)
        case 'forest':
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
        case 'xgb':
            model = XGBClassifier(**best_hyperparameters)
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

trained_model = pick_model('forest')

train_predictions = trained_model.predict(x_train)
val_predictions = trained_model.predict(x_val)
test_predictions = trained_model.predict(x_test)


jc = [0, 0, 0]
acc = [0, 0, 0]
rec = [0, 0, 0]
for i, prediction in enumerate(train_predictions, val_predictions, test_predictions):
    jc[i] = jaccard_score(val_predictions, y_val)
    acc[i] = accuracy_score(val_predictions, y_val)
    rec[i] = recall_score(val_predictions, y_val)

print('train', 'val', 'test')
print(jc)
print(acc)
print(rec)

#vis = lma_pixel_classifier_visualizer(trained_model, train_pickle)
