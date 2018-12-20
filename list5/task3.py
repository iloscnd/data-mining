from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import datasets
import numpy as np

###
# Random Forest - draws samples (with replacement), split is taken via random subset of features
# result is avarage of probabilites on trees in forest
#
# Extremely -||- - as previous trees but treshold is not best but best from some randomly chosen subset
#
###


forest = RandomForestClassifier(100, max_features='log2')
extra = ExtraTreesClassifier(100,max_features='log2')

iris_X, iris_Y = datasets.load_iris(return_X_y=True)

shuffler = np.random.permutation(len(iris_X))

forest.fit(iris_X[shuffler][:100], iris_Y[shuffler][:100])
extra.fit(iris_X[shuffler][:100], iris_Y[shuffler][:100])

prediction = forest.predict(iris_X[shuffler][100:])
prediction_extra = extra.predict(iris_X[shuffler][100:])

print("Random Trees error rate: {}".format(1 - np.sum(prediction == iris_Y[shuffler][100:])/50))
print("Extra Random Trees error rate: {}".format(1 - np.sum(prediction_extra == iris_Y[shuffler][100:])/50))




#### TITANIC ######

import pandas as pd



titanic = pd.read_csv("list5/train.csv")
titanic = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)
titanic = titanic.dropna()
titanic['Sex'] = pd.Categorical(titanic['Sex']).codes
titanic['Embarked'] = pd.Categorical(titanic['Embarked']).codes


titanic_Y = titanic['Survived'].as_matrix()
titanic_X = titanic.drop("Survived", axis=1).as_matrix()


forest = RandomForestClassifier(100, max_features='log2')
extra = ExtraTreesClassifier(100, max_features='log2')

def cross_validation(data, target, model):
    num_samples = data.shape[0]
    perm = np.arange(num_samples)
    np.random.shuffle(perm)
    data = data[perm]
    target = target[perm]

    batch_sz = num_samples//10
    
    errs = 0
    cnt = 0
    for batch in range(0, num_samples - batch_sz + 1, batch_sz):
        cnt +=1
        
        batch_train_X = np.concatenate( [data[:batch],   data[(batch+batch_sz):]] )
        batch_train_Y = np.concatenate( [target[:batch], target[(batch + batch_sz):]] )

        batch_test_X = data[batch:batch+batch_sz]
        batch_test_Y = target[batch:batch+batch_sz]
        model = model.fit(batch_train_X,batch_train_Y)

        #print(pred)
        #print(t.score(batch_test_X, batch_test_Y))
        errs += model.score(batch_test_X, batch_test_Y)

    return errs/cnt






print("Random Trees Titanic cross validation: {}".format(cross_validation(titanic_X, titanic_Y, forest)))
print("Extra Random Trees Titanic cross validation: {}".format(cross_validation(titanic_X, titanic_Y, extra)))


### no dane są dyskretne to normalizacja trochę słabo



