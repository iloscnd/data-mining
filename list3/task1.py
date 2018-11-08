import numpy as np



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
    
    #Ns [i] to liczba punktów w rozkładzie itym

    points = np.random.normal(np.zeros( (N,d) ), np.ones( (N,d)  ))

    
    print(points.shape)



N = 1000
d = 2
K = 5
p = np.ones(5, dtype=np.float)/K
mi = np.zeros((K,d))
sigma = np.broadcast_to(np.eye(d)[np.newaxis, :, :], (K, d, d))
print(get_points(N, d, p, mi, sigma))

    




