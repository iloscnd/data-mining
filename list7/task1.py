import numpy as np
import matplotlib.pyplot as plt
import random

def fun(X):
    X_0 = (X-np.mean(X, axis=0)) / np.std(X, axis=0)
    S = np.cov(X_0, rowvar=False)
    D, V = np.linalg.eig(S)

    print(V, D)

    Y = V.T.dot(X_0.T).T/np.sqrt(D)
    return Y, X_0


def stats(X):
    #średnią, wariancję, macierz kowariancji i macierz korelacji
    print("Średnia ", np.mean(X, axis=0))
    print("Wariancja", np.var(X, axis=0))
    print("Macierz Kowariancji")
    print(np.cov(X, rowvar=False))
    print("Macierz korelacji")
    print(np.corrcoef(X, rowvar=False))
    return

def stats_three(X,X_0,Y):
    print("Dane oryginalne")
    stats(X)
    print("Dane po standaryzacji")
    stats(X_0)
    print("Dane po PCA")
    stats(Y)

print("Podpunkt a")

X = np.random.standard_normal(2000).reshape(1000,2)
sigma = np.linalg.cholesky([[12, 3], [3, 1]])
X = sigma.dot(X.T).T
X = X + [3,5]

Y, X_0 = fun(X)

stats_three(X,X_0, Y)

fig, axes = plt.subplots(ncols=3,figsize=(15,5))
axes[0].scatter(X[:,0], X[:,1])
axes[1].scatter(X_0[:,0], X_0[:,1])
axes[2].scatter(Y[:,0], Y[:,1])
plt.show()


print("Podpunkt b")

X = np.random.standard_normal(2000).reshape(1000,2)
sigma = np.linalg.cholesky([ [12, 3], [3, 1]] )
X = sigma.dot(X.T).T

mis =  [[-21, -2], [3, 5], [27, 12]]
for i in range(1000):
    r = random.random()

    if r < 1/3:
        X[i] += mis[0]
    elif r < 2/3:
        X[i] += mis[1]
    else:
        X[i] += mis[2]

Y, X_0 = fun(X)

stats_three(X,X_0, Y)

fig, axes = plt.subplots(ncols=3,figsize=(15,5))
axes[0].scatter(X[:,0], X[:,1])
axes[1].scatter(X_0[:,0], X_0[:,1])
axes[2].scatter(Y[:,0], Y[:,1])
plt.show()
