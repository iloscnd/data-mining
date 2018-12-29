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


best_rules = []

for a,b in assoc.rules:
    conf, lift, suppA, suppB, suppAB = assoc._get_stats(a,b)
    best_rules.append((lift,conf,len(a)+len(b),suppAB, a,b ))

i = 0
for lift, conf, _, suppAB, a, b in sorted(best_rules, reverse=True):
    print("{} => {} \t lift: {}, conf: {}, supp: {}".format(tuple(a), tuple(b), lift, conf, suppAB))




