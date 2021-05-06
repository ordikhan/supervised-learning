import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

random.seed(2002)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)

KNNmodel = KNeighborsClassifier(n_neighbors=5)
KNNmodel = KNNmodel.fit(Xtrain, Ytrain)

pred = KNNmodel.predict(Xtest)
acc = accuracy_score(Ytest, pred)
print(acc)
