import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from k_means import KMeans

data = pd.read_csv('../data/iris.csv')
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']

x_axis = 'petal_length'
y_axis = 'petal_width'

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type],label=iris_type)
plt.title('known category')
plt.legend(loc='best')
plt.subplot(1, 2, 2)
plt.scatter(data[x_axis][:], data[y_axis][:])
plt.title('unknown category')
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape(num_examples, 2)

# 指定训练参数
num_clusters = 3
max_iterations = 100

kmeans = KMeans(data=x_train, num_clusters=num_clusters)
centroids, closest_centroids_ids = kmeans.train(max_iterations=max_iterations)

# 对比结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type], data[y_axis][data['class'] == iris_type],label = iris_type)
plt.title('known category')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
for centroid_id, centroid in enumerate(centroids):
    current_examples_index = (closest_centroids_ids == centroid_id).flatten()
    plt.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], label = centroid_id)
for centroid_id, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], c = 'black', marker = '+')
plt.legend(loc='best')
plt.title('kmeans')
plt.show()