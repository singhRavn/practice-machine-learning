import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets._samples_generator as make_blobs
import sklearn.cluster as cluster
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot')

centers = [[1,1],[5,7],[8,3]]   
X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=1.5)

ms = cluster.MeanShift()
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Number of estimated clusters:", n_clusters_) 

colors = 10*["g","r","c","b","k"]   
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  
