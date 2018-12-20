import numpy as np
from sklearn import tree
import pandas as pd

bank = pd.read_csv('list5/bank.csv', sep=";")

n = len(bank)
bank_y = pd.Categorical(bank['y']).codes

for col in ['job', 'marital',  'education', 'default', 'housing', 'loan']:
    bank[col] = pd.Categorical(bank[col]).codes

bank_x = bank.as_matrix()[:, :8]
print(bank_x)

t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=6)

t.fit(bank_x, bank_y)
print(t.score(bank_x, bank_y))

shuffler = np.random.permutation(n)
test_size = int(n * 2/3)

t.fit(bank_x[shuffler][:test_size], bank_y[shuffler][:test_size])

print("Error rate {}".format(t.score(bank_x[shuffler][test_size:], bank_y[shuffler][test_size:])))

with open("list5/bank.dot", "w") as f:
    tree.export_graphviz(t, out_file=f, feature_names=['age', 'job', 'marital',  'education', 'default',  'balance', 'housing', 'loan'])



