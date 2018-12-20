import numpy as np
from scipy.stats import mode


def knn(X, Y, labels, k):
    
    dists =  - 2*Y.dot(X.T) + np.sum(Y**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1) 

## za duzo

    closest = np.argsort(dists, axis=1)[:, :k]
#    uni, counts = np.unique(labels[closest], return_counts=True) # does not realy work with 2d arrays
    #print(labels[closest])
    return mode(labels[closest], axis=1)[0].ravel()









