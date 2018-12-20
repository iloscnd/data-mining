from sklearn import tree
import numpy as np
import pandas as pd


mushrooms = pd.read_csv('list5/agaricus-lepiota.data', header=None)


for col in range(23):
    mushrooms[col] = pd.Categorical(mushrooms[col]).codes


mushrooms = mushrooms.drop(11, axis = 1) ### It is a row with n/a

mushrooms_x = mushrooms.drop(0, axis=1).as_matrix()
mushrooms_y = mushrooms[0].as_matrix()

n = len(mushrooms)


t = tree.DecisionTreeClassifier(criterion='entropy')

test_size = int(1/2 * n)

shuffle = np.random.permutation(n)

t.fit(mushrooms_x[shuffle][:test_size], mushrooms_y[shuffle][:test_size])

#print(t.score(mushrooms_x[shuffle][test_size:], mushrooms_y[shuffle][test_size:]))
predictions = t.predict(mushrooms_x[shuffle][test_size:])

print("correct {}/{}".format(np.sum(predictions == mushrooms_y[shuffle][test_size:]),(n-test_size)))

with open("list5/mushroom.dot", "w") as f:
    tree.export_graphviz(t, out_file=f)



