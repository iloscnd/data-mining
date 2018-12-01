import numpy as np
import matplotlib.pyplot as plt


from sklearn import datasets



centers_ = [[1, 1], [3, 3], [5, 1]]
X, labels = datasets.make_blobs(n_samples=3000, n_features=2, centers=centers_, cluster_std=0.5)


#plt.scatter(X[:,0], X[:,1], c=labels)

#plt.show()


from sklearn.cluster import Birch

birch = Birch(threshold=0.25)
birch.fit(X)

plt.scatter(X[:,0], X[:,1], c=birch.labels_)

plt.scatter(birch.root_.centroids_[:,0], birch.root_.centroids_[:,1], c="r", marker='x')



from itertools import cycle
colors = cycle(["g", "blue", "y", "black"])

if not birch.root_.is_leaf:
    for sub in birch.root_.subclusters_:
        subNode = sub.child_      
        print(subNode.is_leaf)
        color = next(colors)
        plt.scatter(subNode.centroids_[0], subNode.centroids_[1], c=color, marker='*', s=120)
        
plt.show()



birch = Birch(threshold=0.25, n_clusters=None)
birch.fit(X)

plt.scatter(X[:,0], X[:,1], c=birch.labels_)

plt.scatter(birch.root_.centroids_[:,0], birch.root_.centroids_[:,1], c="r", marker='x')

if not birch.root_.is_leaf:
    for sub in birch.root_.subclusters_:
        subNode = sub.child_      
        print(subNode.is_leaf)

        color = next(colors)
        plt.scatter(subNode.centroids_[0], subNode.centroids_[1], c=color, marker='*', s=120)
        

plt.show()

