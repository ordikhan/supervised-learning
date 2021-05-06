import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
Y = iris.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)
GNB = GaussianNB().fit(Xtrain, Ytrain)
pred = GNB.predict(Xtest)
acc = accuracy_score(Ytest, pred)
print(acc)
#print(GNB.sigma_)


from sklearn.naive_bayes import BernoulliNB
BNB = BernoulliNB(binarize=1.8).fit(Xtrain, Ytrain)
pred = BNB.predict(Xtest)
acc = accuracy_score(Ytest, pred)
print(acc)

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB().fit(Xtrain, Ytrain)
pred = MNB.predict(Xtest)
acc = accuracy_score(Ytest, pred)
print(acc)