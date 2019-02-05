import numpy as np
import matplotlib.pyplot as plt
import random



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


from sklearn import datasets

X, target = datasets.load_iris(True)



mi = np.mean(X, axis=0)
sigma = np.std(X, axis=0)

X_0 = (X-mi) / np.std(X, axis=0)
S = np.cov(X_0, rowvar=False)
D, V = np.linalg.eig(S)

V_less = V[:,:2]

Y = V.T.dot(X_0.T).T/np.sqrt(D)

Y = Y[:,:2]

#tats_three(X,X_0, Y)



X_rec = Y.dot(V_less.T)*sigma + mi


print(X[2])
print(X_rec[2])

print(np.sum(np.sum((X - X_rec)**2,axis=1)))

fig, axes = plt.subplots(ncols=3,figsize=(15,5))
axes[0].scatter(X[:,0], X[:,1], c=target)
axes[1].scatter(X_rec[:,0], X_rec[:,1], c=target)
axes[2].scatter(Y[:,0], Y[:,1], c=target)
plt.show()




## I -> wartości własne mówią jak bardzo korzystamy z tej podprzestrzeni