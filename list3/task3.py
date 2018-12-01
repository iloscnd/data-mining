import numpy as np
import matplotlib.pyplot as plt

from kmeans import kmeans

from sklearn.manifold import TSNE

# iris 
#   1. sepal length in cm
#   2. sepal width in cm
#   3. petal length in cm
#   4. petal width in cm
#   5. class: 
#      -- Iris Setosa
#      -- Iris Versicolour
#      -- Iris Virginica

class_to_number = {
    "Iris-setosa":0,
    "Iris-versicolor":1,
    "Iris-virginica":2
}

data = []
target = []

for line in open("task3resources/iris/iris.data"):
    if len(line) < 10:
        continue
    sl, sw, pl, pw, cl = line.split(',')
    data.append(np.array([float(x) for x in [sl, sw, pl, pw]], dtype=np.float))
    target.append(class_to_number[cl[:-1]])

data = np.array(data)
target = np.array(target)

fig, axes = plt.subplots(2, 3, figsize=(15,15))

for i in range(3):
    axes[0, i].scatter(data[:, i], data[:, i+1], c=target)

C, assignment = kmeans(data, 3)

for i in range(3):
    axes[1, i].scatter(data[:, i], data[:, i+1], c=assignment)
    axes[1, i].scatter(C[:,i], C[:, i+1], c="r", marker='x')

plt.show()



#wine

# 	1) Alcohol - Class
# 	2) Malic acid
# 	3) Ash
#	4) Alcalinity of ash  
# 	5) Magnesium
#	6) Total phenols
# 	7) Flavanoids
# 	8) Nonflavanoid phenols
# 	9) Proanthocyanins
#	10)Color intensity
# 	11)Hue
# 	12)OD280/OD315 of diluted wines
# 	13)Proline



data = []
target = []

for line in open("task3resources/wine/wine.data"):
    li = line.split(",")
    target.append(int(li[0]))
    data.append(np.array([float(x) for x in li[1:]]))

data = np.array(data)
target = np.array(target)


data_normalized = data / np.max(data, axis= 0)


_, assignment = kmeans(data_normalized, 3)


fig, axes = plt.subplots(1, 2, figsize=(15,20))

print(data.shape)
data_showable =  TSNE(n_components=2).fit_transform(data)
print(data_showable.shape)
axes[0].scatter(data_showable[:, 0], data_showable[:, 1], c=target)
axes[1].scatter(data_showable[:, 0], data_showable[:, 1], c=assignment)


plt.show()


#car
#| attributes
#
#buying:   vhigh, high, med, low.
#maint:    vhigh, high, med, low.
#doors:    2, 3, 4, 5more.
#persons:  2, 4, more.
#lug_boot: small, med, big.
#safety:   low, med, high.
#| class values
#
#unacc, acc, good, vgood
#


class_to_number = {
    "vhigh":4,
    "high":3,
    "med":2,
    "low":1,
    "2":2,
    "3":3,
    "4":4,
    "5more":5,
    "more":5,
    "small":1,
    "big":3,
    "unacc":0,
    "acc":1,
    "good":2,
    "vgood":3

}


data = []
target = []
for line in open("task3resources/car/car.data"):
    li = [class_to_number[x] for x in line[:-1].split(',')]
    data.append(np.array(li[:-1]))
    target.append(li[-1])

data = np.array(data)
target = np.array(target)

_, assignment = kmeans(data, 4)


#print how many in classes




