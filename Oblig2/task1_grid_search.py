
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import itertools

X_tot = datasets.load_digits().data
Y_tot = datasets.load_digits().target
X, X_test, Y, Y_test = train_test_split(X_tot, Y_tot, test_size=0.2)

parameters = {'hidden_layer_sizes': [x for x in itertools.product(np.arange(100, 600, 50), repeat=2)], 'solver': [
    "lbfgs"], 'activation': ["tanh"], 'max_iter': [1000], 'verbose': [True]}

clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, iid=True).fit(X, Y)
print(clf.score(X_test, Y_test))
print(clf.best_params_)
