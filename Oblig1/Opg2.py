import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
# This is functionally the same as reading the data from the CSV file in the assignment
iris = datasets.load_iris()
X_tot = iris.data
y_tot = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_tot, y_tot, test_size=0.25)


accX = []
accY = []
# Test out 20 different variations of n_neighbours and find the most accurate one
for i in range(20):
    kfold = model_selection.KFold(n_splits=5, random_state="seed")
    model = KNeighborsClassifier(n_neighbors=i+1).fit(X_train, y_train)
    result = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
    est = model
    # run over the entirety of the test data and report the accuracy level
    predictions = est.predict(X_test)
    correct = 0
    for pred, actual in zip(predictions, y_test):
        if pred == actual:
            correct += 1
    print("The prediction Algorithm with n_neighbors = ", i, " is:",
          (correct/len(predictions))*100, "% correct (", correct, "/", len(predictions), ")")
    print(result.mean())

    accY.append(result.mean())
    accX.append(i+1)
# Graph out the accuracy level of the varying levels of n_neighbours
fig, ax = plt.subplots()
ax.bar(accX, accY)
ax.set_xlabel('n_neighbours')
plt.xlim(0.5, 20.5)
ax.set_ylabel('Mean accuracy')
plt.show()

#distances = list()
#dst = float("+inf")
# for i in range(len(X_train)):
# Compute Eucledian distance between tessample and each of the training sample feature vector in X[]
#    dst_temp = float(distance.euclidean(X_train[i], X_train[0]))
#    distances.append((i, dst_temp))
# compare current distance with previously obtained lowest distance
#    if float(dst_temp) < float(dst):
#        dst = dst_temp
# we are assigning the class id of the ith element to testsample
#        predicted_classid = y_train[i]
#distances.sort(key=lambda tup: tup[1])


# print(predicted_classid)
#neighbors = list()
# for i in range(15):
#   neighbors.append(distances[i][0])
# print(neighbors)
