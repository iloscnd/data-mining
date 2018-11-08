import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

from kmeans import kmeans

iris = datasets.load_iris()


fig, axes = plt.subplots(4, 3, figsize=(15,15))


for k in range(3,6):
    C, assignment = kmeans(iris.data, k)

    for i in range(3):
        axes[k-3, i].scatter(iris.data[:, i], iris.data[:, i+1], c=assignment)
        axes[k-3, i].scatter(C[:,i], C[:, i+1], c="r")

for i in range(3):
    axes[3, i].scatter(iris.data[:, i], iris.data[:, i+1], c=iris.target)

plt.show()


