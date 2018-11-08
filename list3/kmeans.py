import numpy as np
from scipy.spatial.distance import cdist


def kmeans(X, k): # X in dxN, k \in int
    ns, dim = X.shape

    C = np.random.choice(np.arange(ns), k)
    C = X[C]
    
    last_assignment=None
    print(X.shape, C.shape)
    while True:
        assignment = np.argmin(cdist(X, C, metric="euclidean"), axis=1)

        if np.array_equal(assignment, last_assignment):
            break
        
        for cluster in range(k):
            choice = (assignment == cluster)
            nChoice = np.sum(choice)
            if nChoice == 0:
                continue
            C[cluster] = np.sum(X * choice[:, np.newaxis], axis=0)/nChoice
        
        last_assignment = assignment

    return C, assignment


