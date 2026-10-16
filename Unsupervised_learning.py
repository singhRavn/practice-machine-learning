"""
Basic Unsupervised Learning example using K-Means clustering.

Unsupervised learning means the model works with data that has no labels.
The algorithm learns patterns or groups in the data by itself.

In this example, we create a small dataset with 2 clusters and use K-Means
to group points into clusters.
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1) Create a simple dataset with 2 clusters
# ------------------------------------------------------------
X = np.array([
    [1, 2], [1, 4], [1, 5],
    [2, 1], [2, 3], [2, 5],
    [8, 8], [8, 9], [9, 7],
    [9, 9], [10, 8], [10, 10]
])

# ------------------------------------------------------------
# 2) Train K-Means model
# ------------------------------------------------------------
model = KMeans(n_clusters=2, random_state=0)
model.fit(X)

# ------------------------------------------------------------
# 3) Show cluster centers and labels
# ------------------------------------------------------------
print("Cluster centers:")
print(model.cluster_centers_)

print("\nCluster labels for each point:")
print(model.labels_)

# ------------------------------------------------------------
# 4) Plot the clusters
# ------------------------------------------------------------
plt.figure(figsize=(6, 6))

# Plot all data points
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap='viridis', s=80)

# Plot cluster centers
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            s=200, c='red', marker='X', label='Centroid')

plt.title('K-Means Clustering Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# ------------------------------------------------------------
# 5) Explanation
# ------------------------------------------------------------
# K-Means finds groups (clusters) in the data by repeatedly:
# - assigning points to the nearest center
# - updating the center based on the points in each cluster
#
# Unlike supervised learning, there are no labels here.
# The model tries to discover the structure on its own.
