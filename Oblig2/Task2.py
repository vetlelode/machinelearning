import numpy as np
import matplotlib as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

X_tot = datasets.load_digits().data
Y_tot = datasets.load_digits().target
X, X_test, Y, Y_test = train_test_split(X_tot, Y_tot, test_size=0.2)
# Scale here is the value that would normally equal gamma = 'scale'
scale = 1/(X.shape[1]*X.var())
parameters = {'C': np.arange(1, 10.5, 0.5), 'gamma': np.arange(
    scale/2, scale*2, scale/10)}

clf = GridSearchCV(SVC(), parameters, n_jobs=-1).fit(X, Y)
score = clf.score(X_test, Y_test)
C = clf.best_estimator_.C
Gamma = clf.best_estimator_.gamma
print("Best result gained with C={} and gamma={}\nGiving the accuracy: p={}".format(
    C, Gamma, score))
