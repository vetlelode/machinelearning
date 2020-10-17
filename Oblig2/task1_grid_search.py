
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import itertools

X_tot = datasets.load_digits().data
Y_tot = datasets.load_digits().target
X, X_test, Y, Y_test = train_test_split(X_tot, Y_tot, test_size=0.2)

parameters_3layers = {'hidden_layer_sizes': [x for x in itertools.product((50, 100, 200, 300, 400, 500), repeat=3)], 'solver': [
    "lbfgs"], 'activation': ["tanh"], 'max_iter': [1000], 'verbose': [True]}
parameters_2layers = {'hidden_layer_sizes': [x for x in itertools.product((50, 100, 200, 300, 400, 500), repeat=2)], 'solver': [
    "lbfgs"], 'activation': ["tanh"], 'max_iter': [1000], 'verbose': [True]}

clf_3_layers = GridSearchCV(
    MLPClassifier(), parameters_3layers, n_jobs=-1, iid=True).fit(X, Y)
score3 = clf_3_layers.score(X_test, Y_test)

clf_2_layers = GridSearchCV(
    MLPClassifier(), parameters_2layers, n_jobs=-1, iid=True).fit(X, Y)
score2 = clf_2_layers.score(X_test, Y_test)

print("The highest score achived with 2 layers: {}, with params: {} \nThe highest score achived with 3 layers: {}, with params: {}".format(
    round(score2, 3), clf_2_layers.best_params_, round(score3, 3), clf_3_layers.best_params_))
