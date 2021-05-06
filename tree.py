import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import random
import pickle

random.seed(2002)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)

 
tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=15, min_samples_leaf=5)
tree_model = tree_model.fit(Xtrain, Ytrain)

pred = tree_model.predict(Xtest)
acc = accuracy_score(Ytest, pred)
print(acc)

