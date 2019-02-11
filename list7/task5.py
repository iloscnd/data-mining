import numpy as np
from scipy.stats import mode
import scipy.io as sio

import sys

def fit(X, X_test, n_components=100):
    mi = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_0 = (X-mi) / sigma
    S = np.cov(X_0, rowvar=False)
 
    mi_test = np.mean(X_test, axis=0)
    sigma_test = np.mean(X_test, axis=0)
    X_0_test = (X_test - mi_test)/sigma_test
 
    D, V = np.linalg.eig(S)
    D = D.real
    V = V.real
    Y = V.T.dot(X_0.T).T/np.sqrt(np.maximum(D,1e-17))
    Y = Y[:, :n_components]

    Y_test = V.T.dot(X_0_test.T).T/np.sqrt(np.maximum(D,1e-17))
    Y_test = Y_test[:, :n_components]
    return Y, Y_test

def knn(X, Y, labels, k):
    dists =  - 2*Y.dot(X.T) + np.sum(Y**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1) 
    closest = np.argsort(dists, axis=1)[:, :k]
    return mode(labels[closest], axis=1)[0].ravel()



X = sio.loadmat('list7/ReducedImagesForTraining.mat')['images'].T
X_prime = sio.loadmat('list7/ReducedImagesForTesting.mat')['images'].T
target = np.arange(250) // 5
target_priem = np.arange(100) // 2

X = np.concatenate((X, X_prime))
target = np.concatenate((target, target_priem), axis=0)

print("loaded", file=sys.stderr)

num_samples = X.shape[0]
perm = np.arange(num_samples)
np.random.shuffle(perm)
X = X[perm]
target = target[perm]

batch_sz = 50



for k in range(1,12,2):
    errs = 0
    cnt = 0
    print("k = {}:".format(k))
    for batch in range(0, num_samples - batch_sz + 1, batch_sz):
        cnt +=1
        
        batch_train_X = np.concatenate( [X[:batch],   X[(batch+batch_sz):]] )
        batch_train_Y = np.concatenate( [target[:batch], target[(batch + batch_sz):]] )

        batch_test_X = X[batch:batch+batch_sz]
        batch_test_Y = target[batch:batch+batch_sz]
        

        fitted_train_X, fitted_test_X = fit(batch_train_X, batch_test_X, 250)
        predictions = knn(fitted_train_X, fitted_test_X, batch_train_Y, k )

        step_errs = np.sum(predictions == batch_test_Y)

        print("batch err:", step_errs)

        errs += step_errs

    print("Total accuraccy: ", 1-errs/350)
    print()





