import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import k_means, Birch

n = 1000
mi = np.array([[0,0],[4,0]])
sigma = np.diag([1,100])
points = np.random.normal(loc=mi.repeat(n/2,0), size=(n,2))
points = sigma.dot(points.T).T

points_normalized = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
points_standarized = (points - points.mean(axis=0)) / points.std(axis=0)


_, label, _ = k_means(points,2)
_, label_normalized, _ = k_means(points_normalized,2)
_, label_standarized, _ = k_means(points_standarized,2)


fig, axes = plt.subplots(3,3, figsize=(20,20))

axes[0,0].scatter(points[:,0], points[:,1], c=label)
axes[0,1].scatter(points[:,0], points[:,1], c=label_normalized)
axes[0,2].scatter(points[:,0], points[:,1], c=label_standarized)

birch = Birch(threshold=0.01, branching_factor=30, n_clusters=2)


birch.fit(points)
axes[1,0].scatter(points[:,0], points[:,1], c=birch.labels_)
birch.fit(points_normalized)
axes[1,1].scatter(points[:,0], points[:,1], c=birch.labels_)
birch.fit(points_standarized)
axes[1,2].scatter(points[:,0], points[:,1], c=birch.labels_)



#DBScan 

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=2.5)


dbscan.fit(points)
axes[2,0].scatter(points[:,0], points[:,1], c=dbscan.labels_)

#dbscan = DBSCAN(eps=0.05)

dbscan.fit(points_normalized)

axes[2,1].scatter(points[:,0], points[:,1], c=dbscan.labels_)

dbscan.fit(points_standarized)
axes[2,2].scatter(points[:,0], points[:,1], c=dbscan.labels_)

plt.show()




