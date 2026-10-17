"""
Basic Unsupervised Learning example using K-Means clustering.

This version goes one step further by:
- checking how the number of clusters affects the model
- predicting cluster assignment for new points

Unsupervised learning means the model works with unlabeled data.
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = np.array([
    [1, 2], [1, 4], [1, 5],
    [2, 1], [2, 3], [2, 5],
    [8, 8], [8, 9], [9, 7],
    [9, 9], [10, 8], [10, 10]
])

model = KMeans(n_clusters=2, random_state=0)
model.fit(X)

print("Cluster centers:")
print(model.cluster_centers_)

print("\nCluster labels for each point:")
print(model.labels_)

print("\nInertia for different K values:")
for k in range(1, 6):
    temp_model = KMeans(n_clusters=k, random_state=0)
    temp_model.fit(X)
    print(f"k={k} -> inertia={temp_model.inertia_:.2f}")

new_points = np.array([
    [2, 2],
    [9, 8],
    [5, 5]
])

predicted_clusters = model.predict(new_points)
print("\nPredicted clusters for new points:")
for point, cluster in zip(new_points, predicted_clusters):
    print(f"Point {point} -> Cluster {cluster}")

plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis', s=80)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroid')

for point in new_points:
    plt.scatter(point[0], point[1], s=100, c='black', marker='o')

plt.title('K-Means Clustering Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

