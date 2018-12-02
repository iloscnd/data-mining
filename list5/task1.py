import numpy as np
from knn import knn

from sklearn import datasets

import matplotlib.pyplot as plt

def cross_validation(data, target, k):
    num_samples = data.shape[0]
    perm = np.arange(num_samples)
    np.random.shuffle(perm)
    data = data[perm]
    target = target[perm]

    batch_sz = num_samples//10
    
    errs = 0
    for batch in range(0, num_samples, batch_sz):

        batch_train_X = np.concatenate( (data[batch:],   data[:(batch+batch_sz)]) )
        batch_train_Y = np.concatenate( (target[batch:], target[:batch + batch_sz]) )

        batch_test_X = data[batch:batch+batch_sz]
        batch_test_Y = target[batch:batch+batch_sz]

        pred = knn(batch_train_X, batch_test_X, batch_train_Y, k)

        #print(pred)
        #print(batch_test_Y)
        errs += np.sum(pred != batch_test_Y)

    return errs/num_samples



iris = datasets.load_iris()


test_count = 100

perm = np.arange(iris['data'].shape[0])

print("IRIS")

for k in range(1,20,2):
    errs = 0
    for test in range(test_count):
        np.random.shuffle(perm)
        
        data = iris['data'][perm]
        target = iris['target'][perm]

        prediction = knn(data[:100], data[100:], target[:100], k)
        #print(prediction)
        errs += np.sum(prediction != target[100:])
    
    print("for k={} error rate is {}".format(k, errs/(50*test_count)))
    print("for k={} cross validation is {}".format(k, cross_validation(iris['data'], iris['target'], k)))

print("DIGITS")

import pandas

data = pandas.read_csv("list5/optdigits.tes").values

num_samples = data.shape[0]
num_features = data.shape[1] - 1
data, target = data[:,:num_features], data[:, -1].ravel()

perm = np.arange(num_samples)

train_size = num_samples*2//3
test_size = num_samples - train_size

test_count = 10
for k in range(1,20,2):
    errs = 0
    for test in range(test_count):
        np.random.shuffle(perm)
        
        data = data[perm]
        target = target[perm]

        prediction = knn(data[:train_size], data[train_size:], target[:train_size], k)
        errs += np.sum(prediction != target[train_size:])

    print("for k={} error rate is {}".format(k, errs/(test_size*test_count)))
    print("for k={} cross validation is {}".format(k, cross_validation(data, target, k)))

    










