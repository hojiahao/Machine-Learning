import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression

data = pd.read_csv('../data/microchips-tests.csv')
# Catagory labels
validities = [0, 1]
x_axis = 'param_1'
y_axis = 'param_2'

for validity in validities:
    plt.scatter(data[x_axis][data['validity'] == validity], data[x_axis][data['validity'] == validity], label = validity)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.title('Microchips Tests')
plt.legend(loc='best')
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, -1))
y_train = data['validity'].values.reshape((num_examples, 1))

max_iterations = 100000
regularization_param = 0
polynomial_degree = 5
sinusoid_degree = 0

logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

(thetas, costs) = logistic_regression.train(max_iterations)

columns = []
for theta_index in range(0, thetas.shape[1]):
    columns.append('Theta' + str(theta_index))

labels = logistic_regression.unique_labels

plt.plot(range(len(costs[0])), costs[0], label=labels[0])
plt.plot(range(len(costs[1])), costs[1], label=labels[1])
plt.xlabel('Gradient Steps')
plt.ylabel('Cost')
plt.legend(loc='best')
plt.show()

y_pred = logistic_regression.predict(x_train)
precision = np.sum(y_pred == y_train) / num_examples * 100
print('Precision: {:5.4f}%'.format(precision))

num_examples = x_train.shape[0]
samples = 150
x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])
X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)
Z = np.zeros((samples, samples))

for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data2 = np.array([x, y]).reshape(-1, 2)
        Z[x_index][y_index] = logistic_regression.predict(data2)[0][0]
positives = (y_train == 1).flatten()
negatives = (y_train == 0).flatten()

plt.scatter(x_train[negatives, 0], x_train[negatives, 1], label='Negatives')
plt.scatter(x_train[positives, 0], x_train[positives, 1], label='Positives')
plt.contour(X, Y, Z)
plt.show()