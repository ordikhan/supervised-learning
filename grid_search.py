import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import random
from sklearn.model_selection import GridSearchCV

random.seed(2002)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

param = {'learning_rate_init': [0.01,0.05,0.1,0.15,0.2,0.25,0.3],
           'momentum':[0.4,0.5,0.6,0.7,0.8,0.9]
           }
net = MLPClassifier(hidden_layer_sizes=(10), learning_rate='invscaling', max_iter=1000)

gs = GridSearchCV(net, param)
gs.fit(X,Y)

pred = gs.predict(X)

print(accuracy_score(Y, pred))
print(gs.best_params_)