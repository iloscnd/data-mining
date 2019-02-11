import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


reduction_dim = 100

train = sio.loadmat('list7/ReducedImagesForTraining.mat')['images'].T
print(train.shape)
print("loaded")

mi = np.mean(train, axis=0)
sigma = np.std(train, axis=0)
train_0 = (train-mi) / sigma
S = np.cov(train_0, rowvar=False)

D, V = np.linalg.eig(S)
D = D.real
V = V.real


train_red = V.T.dot(train_0.T).T/np.sqrt(np.maximum(D,1e-17))
train_red = train_red[:, :reduction_dim]

test = sio.loadmat('list7/ReducedImagesForTesting.mat')['images'].T
mi_test = np.mean(test, axis=0)
sigma_test = np.std(test, axis=0)
test_0 = (test-mi_test) / sigma_test

print(test_0.shape)


test_red = V.T.dot(test_0.T).T/np.sqrt(np.maximum(D,1e-17))
test_red = test_red[:,:reduction_dim]

print(test_red.shape)

def find(X, Y, max_dist=10e9):

    
    dists =  (-2*X.dot(Y.T) + np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1)).T

    min_dists = np.amin(dists, axis=1)
    assignment = np.argmin(dists, axis=1) //5
    correct = np.arange(100) //2 

    print(min_dists)
    print(assignment)
    print(correct)

    return np.sum(assignment == correct)/100


res = find(train_red,test_red)

print("Accuracy: {}".format(res))