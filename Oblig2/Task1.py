import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import csv

# Import the dataset split into training and testing, etc..
X_tot = datasets.load_digits().data
Y_tot = datasets.load_digits().target
X, X_test, Y, Y_test = train_test_split(X_tot, Y_tot, test_size=0.2)


# Plotting related code
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = []
ys = []
zs = []
ax.set_xlabel('Hidden layers:')
ax.set_ylabel('Neurons undearneath layer:')
ax.set_zlabel('Accuracy')
plt.style.use('bmh')
cm = plt.get_cmap("RdYlGn")

# Find the best scoring alternative
best = [0, 0, 0]
for i in range(100, 400):
    # Since this takes a lot of time log the progress every now and then.
    if i % 10 == 0:
        print(i)
    # Uncomment the two lines below to run the simulation alot faster, but with a lot less simulations
    # else:
    #    continue

    # Only used for preventing duplicate sublayers
    neurons = ()
    # Generate 5 random choices for the amount of neurons in the hidden layer between i/10 and i/2
    for n in range(5):
        neuron = random.randint(math.ceil(i/10), math.ceil(i/2))
        # Make sure there are no duplicate neurons
        while isinstance(neuron, neurons):
            neuron = random.randint(math.ceil(i/10), math.ceil(i/2))
        clf = MLPClassifier(hidden_layer_sizes=(i, neuron),
                            random_state=1, max_iter=1000).fit(X, Y)
        # Score the algorithm and save the result to the plotting dataset
        score = clf.score(X_test, Y_test)
        xs.append(i)
        ys.append(neuron)
        zs.append(score)
        # If this is the most accurate yet save as the array best
        if score > best[2]:
            best = [i, neuron, score]


ax.scatter3D(xs, ys, zs, c=zs, cmap='Accent_r')
plt.tight_layout()
plt.show()

print("Most accurate configuration was:", best)
# log the best result in a CSV file
with open('best.csv', mode="a") as results:
    writer = csv.writer(results, delimiter=",")
    writer.writerow(best)
