import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans


def get_points(N, d, p, mi, sigma):
    # p:: K to prawdopodobieństwa pkt 
    # mi: Kxd, sigma Kxdxd
    # wynik: NxD
    
    K = len(p)
    p = np.cumsum(p)
    choice = np.random.uniform(0,1,size=(N,1))
    choice = np.digitize(choice, p)
    Ns = np.arange(K, dtype=np.float).reshape(1,K)
    Ns = np.sum(np.equal(Ns,choice), axis = 0)

    points = np.random.normal(np.zeros( (N,d) ), np.ones( (N,d)  ))

    sigma = np.linalg.cholesky(sigma)
    points = np.squeeze(np.matmul(sigma.repeat(Ns,axis=0),points[:,:,np.newaxis])) + mi.repeat(Ns, axis=0)
    
    #print(points.shape)
    return points




N = 5000

plt.figure(figsize=(20,20))

# Case a)


d = 2
K = 5
p = np.ones(K, dtype=np.float)/K
mi = 3*(np.arange(K)+1).reshape(K,1) * np.ones((d))
sigma = np.array([ np.eye(d) for _ in range(K) ])

points = get_points(N, d, p, mi, sigma)
_, ass = kmeans(points, K)

plt.scatter(points[:, 0], points[:,1], c=ass)
plt.show()

#Case b)

sigma[2] = np.array([[3., 0.], [0., 1.]])

points = get_points(N, d, p, mi, sigma)
_, ass = kmeans(points, K)

plt.figure(figsize=(20,20))
plt.scatter(points[:, 0], points[:,1], c=ass)
plt.show()


#Case c)

sigma[0] = np.array([[3.,1.],[1.,1.]])

points = get_points(N, d, p, mi, sigma)
_, ass = kmeans(points, K)

plt.figure(figsize=(20,20))
plt.scatter(points[:, 0], points[:,1], c=ass)
plt.show()

#Case d)

p = np.array([0.2, 0.1, 0.3, 0.1, 0.3])

points = get_points(N, d, p, mi, sigma)
_, ass = kmeans(points, K)

plt.figure(figsize=(20,20))
plt.scatter(points[:, 0], points[:,1], c=ass)
plt.show()

#Case e)

d = 3
K = 5
p = np.ones(K, dtype=np.float)/K
mi = 3*(np.arange(K)+1).reshape(K,1) * np.ones((d))
sigma = np.array([ np.eye(d) for _ in range(K) ])

sigma[2] = np.array( [[3, 1, 0], [1, 1, 0], [0, 0, 1]] )

points = get_points(N, d, p, mi, sigma)
_, ass = kmeans(points, K)

plt.figure(figsize=(20,20))
plt.scatter(points[:, 0], points[:,1], c=ass)
plt.show()

#Case f)


d = 100
K = 10
p = np.ones(K, dtype=np.float)/K
mi = 3*(np.arange(K)+1).reshape(K,1) * np.ones((d))
sigma = np.array([ np.eye(d) for _ in range(K) ])

points = get_points(N, d, p, mi, sigma)
_, ass = kmeans(points, K)

plt.figure(figsize=(20,20))
plt.scatter(points[:, 0], points[:,1], c=ass)
plt.show()

