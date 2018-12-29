
from apriori import Apriori
import pandas as pd



T = pd.read_csv("list6/task4Data/D12", sep=";", dtype=str, encoding='iso-8859-1', names=["date", "CustID", "Age", "Residance", "Prod Class", "ProdId", "Amount", "Asset", "Price" ])
T = T.drop(0)
T = T.drop("date", axis=1)
#T["CustId"] = T["CustId"].map( (lambda x: "c"+str(x)))
T["ProdId"] = T["ProdId"].map( (lambda x: "p"+str(x)))
T["Prod Class"] = T["Prod Class"].map( (lambda x: "pc"+str(x)))
T["Amount"] = T["Amount"].map( (lambda x: "n"+str(x)))
T["Asset"] = T["Asset"].map( (lambda x: "a"+str(x)))
T["Price"] = T["Price"].map( (lambda x: str(x) + "$"))
T["Age"] = T["Age"].replace(  {"A ": "<25", "B ": "25-29", "C ": "30-34", "D ": "35-39", "E ": "40-44", "F ": "45-49", "G ": "50-54", "H ": "55-59", "I ": "60-64", "J ": ">65"} )
print(T.head(), file=sys.stderr)


X = set()
Ts = []

for i in range(len(T)):
    t = frozenset(T.loc[i+1])
    X |= t
    Ts.append(t)



assoc = Apriori(X, Ts, alpha=0.003, debug=True)


best_rules = []

for a,b in assoc.rules:
    conf, lift, suppA, suppB, suppAB = assoc._get_stats(a,b)
    best_rules.append((lift,conf,len(a)+len(b),suppAB, a,b ))

i = 0
for lift, conf, _, suppAB, a, b in sorted(best_rules, reverse=True):
    print("{} => {} \t lift: {}, conf: {}, supp: {}".format(tuple(a), tuple(b), lift, conf, suppAB))








