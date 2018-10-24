import numpy as np



a = np.arange(1,101)
b = np.arange(1,100,2)
c = np.linspace(-1., 1, 201)*np.pi
d = np.append(np.linspace(-1, 0, 100, endpoint=False), np.linspace(0.01, 1, 100))*np.pi
e = np.maximum(np.sin(a), 0)

print(a, b, c, d, e, sep='\n')

A = a.reshape(10,10)
B = np.diag(a) + np.diag(a[:-1], k=1) + np.diag(a[:-1], k=-1)
C = np.triu(np.ones((10,10)))
D = np.append( np.cumsum(a), np.cumprod(a, dtype=np.float64) ).reshape(2,-1)
E = (np.mod(np.arange(0,100), np.arange(0,100).reshape(100,1)) == 0) * 1 # 0-indexed

print(A, B, C, D, E, sep='\n')

print(E[1 ,2], E[10,20], E[20,10])

