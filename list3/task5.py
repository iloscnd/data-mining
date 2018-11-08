import numpy as np
from kmeans import kmeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


tsne = TSNE(n_components=2)


Z = np.zeros(41271)

for line in open("list3/kosarak.dat"):
    for num in line.split():
        Z[int(num)] += 1


T = 1000

Z = np.argsort(-Z)[:1000]# te sa ok
m = np.zeros(41271, dtype=np.integer)
m[Z] = 1
id = np.cumsum(m) - 1


p = np.zeros((T,T))
data = []

for line in open("list3/kosarak.dat"):
    l = [int(x) for x in line.split()]
    nl = []
    for n1 in l:
        if m[n1]:
            nl.append(n1)
    if nl:
        data.append(nl)


for line in data:
    n = len(line)
    for i in range(n):
        for j in range(i, n):
            p[id[line[i]], id[line[j]]] += 1
            p[id[line[j]], id[line[i]]] += 1

X = tsne.fit_transform(p)

fig, axes = plt.subplots(3,3)
fig.set_size_inches(25,25)

for k in range(3, 11):
    C, ass = kmeans(p.T, k)
    axes[(k-3)%3][(k-3)//3].scatter(X[:,0], X[:,1], c=ass)

plt.show()
