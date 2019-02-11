import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def picture_print(pic, ax):
    p = pic.reshape(60,82)
    ax.imshow(p, cmap="gray")

X = sio.loadmat('list7/ReducedImagesForTraining.mat')['images'].T
print(X.shape)
print("loaded")

mi = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_0 = (X-np.mean(X, axis=0)) / np.std(X, axis=0)
S = np.cov(X_0, rowvar=False)

D, V = np.linalg.eig(S)
D = D.real
V = V.real


#print(D,V)
print(S.shape)
#print(V.shape)

eigenvals = np.sum(D)

n = 0
while D[n] >= eigenvals/10000:
    print(D[n], end=" ")
    n += 1
print("")
print(n)




fig, axes = plt.subplots(3,5)
for i in range(15):
    x = i // 3
    y = i % 3
    picture_print(V[:,i], axes[y,x])
 
plt.show()

Y = V.T.dot(X_0.T).T/np.sqrt(np.maximum(D,1e-17))

Y = Y[:, :250]
V_less = V[:,:250]

X_rec = Y.dot(V_less.T)*sigma + mi

fig, axes = plt.subplots(4,2)

for i in range(4):
    picture_print(X[i], axes[i,0])
    picture_print(X_rec[i], axes[i,1])

plt.show()