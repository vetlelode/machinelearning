# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/ see this link later on
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from scipy.spatial import distance
from numpy.linalg import eig
import pandas as pd
import numpy as np
import sys
# how to execute this code: in the command line :  python knn_Example.py  data_toy_example.csv


df = pd.read_csv("data_file_toy.csv", index_col=None, header=None, )
X = df.loc[:, 1:].values
Y = df[0].values


def KNN(x, y):
    # built-in function to use K-nn
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x, y)
    return model


if __name__ == '__main__':

    # we call the builtin function here by invoking our user defined function KNN()
    model = KNN(X, Y)
    testsample = X[0]
    dst = float("+inf")
    distances = list()
    num_neighbors = 3

    for i in range(len(X)):
        # Compute Eucledian distance between tessample and each of the training sample feature vector in X[]
        dst_temp = float(distance.euclidean(X[i], testsample))
        distances.append((i, dst_temp))
        # compare current distance with previously obtained lowest distance
        if float(dst_temp) < float(dst):
            dst = dst_temp
            # we are assigning the class id of the ith element to testsample
            predicted_classid = Y[i]
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    print(" The neighbours based on built in function model\n")
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    print(neighbors)  # show the row numbers , it starts from 0

    # we can say testsample , since both are same
    predicted_class = model.predict(X)
    # when k=3, when using the built in function  KNeighborsClassifier()### how to execute this code: in the command line :  python PCA_Example.py  data_toy_example.csv
    print("Predicted Class label using builtin function with K=3 is ", predicted_class)
    print("Predicted Class Label after computing eucledian distance is ", predicted_classid,
          " because we are considering only the nearest neighbour, that is k=1")  # when k=1, we calculated the distance on our own
