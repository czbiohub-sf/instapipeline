from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


coords = np.asarray([1,1,2,3,15,16,17,18])

# treshold kmeans
km = KMeans(n_clusters = 2).fit(coords.reshape(-1,1))
cluster_centers = km.cluster_centers_
print(cluster_centers)

# yay this works

