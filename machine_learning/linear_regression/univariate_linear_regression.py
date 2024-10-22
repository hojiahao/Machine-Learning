import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

data = pd.read_csv('../data/world-happiness-report-2017.csv')
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

X_train, y_train = train_data[input_param_name].values, train_data[output_param_name].values
X_test, y_test = test_data[input_param_name].values, test_data[output_param_name].values

plt.scatter(X_train, y_train, label='Train data')
plt.scatter(X_test, y_test, label='Test data')
plt.legend(loc='best')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.show()

num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(X_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate)

print('Initial cost:', cost_history[0])
print('Final cost:', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.title('Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

predictions_num = 100
X_predictions = np.linspace(min(X_train), max(X_train), predictions_num).reshape(predictions_num, 1)
y_predictions = linear_regression.predict(X_predictions)

print('Theta values:', theta)
print('Predictions:', y_predictions[:10])

plt.scatter(X_train, y_train, label='Train data')
plt.scatter(X_test, y_test, label='Test data')
plt.plot(X_predictions, y_predictions, 'r', label='Predictions')
plt.legend(loc='best')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.show()
