import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import random

random.seed(2002)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)

 
ANN = MLPClassifier(hidden_layer_sizes=(20), activation='logistic', learning_rate_init=0.05,
                        learning_rate='invscaling', momentum=0.8, max_iter=200)
ANN = ANN.fit(Xtrain, Ytrain)

pred = ANN.predict(Xtest)
acc = accuracy_score(Ytest, pred)
print(acc)
