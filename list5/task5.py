import numpy as np
from sklearn import tree
import pandas as pd


cars = pd.read_csv("list5/car.data", header=None)

print(pd.Categorical(cars[6]))

for i in range(7):
    cars[i] = pd.Categorical(cars[i]).codes

cars_x = cars.drop(6, axis=1).as_matrix()
cars_y = cars[6].as_matrix()


t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=8)


n = len(cars)
print(n)

test_size = int(n * 2/3)

shuffle = np.random.permutation(n)



t.fit(cars_x[shuffle][:test_size], cars_y[shuffle][:test_size])

# acc = 0, good = 1, unacc=2, vgood =3 #

prediction = t.predict(cars_x[shuffle][test_size:])


acc_index = (cars_y[shuffle][test_size:] == 0)
good_index = (cars_y[shuffle][test_size:] == 1)
unacc_index = (cars_y[shuffle][test_size:] == 2)
vgood_index = (cars_y[shuffle][test_size:] == 3)

acc_predict_index = (prediction == 0)
good_predict_index = (prediction == 1)
unacc_predict_index = (prediction == 2)
vgood_predict_index = (prediction == 3)



print("Correctly guessed as acc {}/{}".format(np.sum(np.logical_and(acc_index, acc_predict_index)), np.sum(acc_index)))
print("Correctly guessed as good {}/{}".format(np.sum(np.logical_and(good_index, good_predict_index)), np.sum(good_index)))
print("Correctly guessed as unacc {}/{}".format(np.sum(np.logical_and(unacc_index, unacc_predict_index)), np.sum(unacc_index)))
print("Correctly guessed as vgood {}/{}".format(np.sum(np.logical_and(vgood_index, vgood_predict_index)), np.sum(vgood_index)))

print("Mislabeled as acc {}/{}".format(np.sum(np.logical_and(np.logical_not(acc_index), acc_predict_index)), np.sum(acc_predict_index)))
print("Mislabeled as good {}/{}".format(np.sum(np.logical_and(np.logical_not(good_index), good_predict_index)), np.sum(good_predict_index)))
print("Mislabeled as unacc {}/{}".format(np.sum(np.logical_and(np.logical_not(unacc_index), unacc_predict_index)), np.sum(unacc_predict_index)))
print("Mislabeled as vgood {}/{}".format(np.sum(np.logical_and(np.logical_not(vgood_index), vgood_predict_index)), np.sum(vgood_predict_index)))


with open("list5/cars.dot", "w") as f:
    tree.export_graphviz(t, out_file=f)


