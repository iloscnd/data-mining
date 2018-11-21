import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans 


def get_centers(K, d, q):
    # return K points with distances bigger then 
    return (np.sqrt(q) + 0.01) * np.diag(np.ones(max(K,d)))[:K, :d]

def get_points(N, d, p, mi):
    # p:: K to prawdopodobieństwa pkt 
    # mi: Kxd
    # wynik: NxD
    
    K = len(p)
    p = np.cumsum(p)
    choice = np.random.uniform(0,1,size=(N,1))
    choice = np.digitize(choice, p)
    Ns = np.arange(K, dtype=np.float).reshape(1,K)
    Ns = np.sum(np.equal(Ns,choice), axis = 0)


    points = np.random.normal(np.zeros( (N,d) ), np.ones( (N,d) ))

    points = points + mi.repeat(Ns, axis=0)
    
    return points


N = 100000

d = 1000
K = 1000
q = 10

#find q points in 


p = np.ones(K, dtype=np.float)/K
mi = get_centers(K,d,q)

points = get_points(N, d, p, mi)

print("done")

c, ass = kmeans(points, K)

print("kmeans done")

plt.scatter(points[:, 0], points[:,1])
plt.show()
