import scipy.io as sio
import numpy as np

train = sio.loadmat('list7/ReducedImagesForTraining.mat')['images'].T
test = sio.loadmat('list7/ReducedImagesForTesting.mat')['images'].T
print("loaded")
print(train.shape)



def find(X, Y, max_dist=10e9):

    
    dists =  (-2*X.dot(Y.T) + np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(Y**2, axis=1)).T

    min_dists = np.amin(dists, axis=1)
    assignment = np.argmin(dists, axis=1) //5
    correct = np.arange(100) //2 

    print(min_dists)
    print(assignment)
    print(correct)

    return np.sum(assignment == correct)/100


res = find(train,test)

print("Accuracy: {}".format(res))