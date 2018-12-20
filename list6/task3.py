from apriori import Apriori
#import pandas as pd



#T = pd.read_csv("list6/task2Data/retail.dat", header=None)
#print(T)

X = set()
T = []

for line in open("list6/task3Data/kosarak.dat"):
    t = frozenset( [int(x) for x in line.split()] )
    X |= t
    T.append(t)



assoc = Apriori(X, T, alpha=0.01, debug=True)
