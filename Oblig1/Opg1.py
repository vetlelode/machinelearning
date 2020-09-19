import numpy as np
import matplotlib.pyplot as plt
import statistics
import pandas as pd
from sklearn.model_selection import train_test_split


data_tot = pd.read_csv('ex1data1.csv')
# Explore the shape of the data
print(data_tot.shape)

# first 80% of the dataset
data = data_tot[:77]
# last 20% of the dataset
test = data_tot[-20:]


X_test = test.iloc[:, 0]
Y_test = test.iloc[:, 1]
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

m, c = 0
n = float(len(X))
L = 0.0001

for i in range(2000):
    Y_hat = m*X+c
    D_m = (-2/n) * sum(X*(Y-Y_hat))
    D_c = (-2/n) * sum(Y - Y_hat)
    m = m - L * D_m
    c = c - L * D_c

Y_Pred = m*X + c
# Plot the training data as the blue dots
plt.scatter(X, Y, color="b")
# Plot the test data as the green dots
plt.scatter(X_test, Y_test, color="g")
# Plot the regression line along with the scatter plots
plt.plot([min(X), max(X)], [min(Y_Pred), max(Y_Pred)], color="r")

plt.show()
