import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import random
from sklearn.ensemble import VotingClassifier


random.seed(2002)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

tree = DecisionTreeClassifier()
GNB = GaussianNB()
BNB = BernoulliNB()

vote = VotingClassifier(estimators=[('tree',tree),('Gnb', GNB),('Bnb', BNB)], weights=[2,1,1])
vote.fit(X,Y)
pred = vote.predict(X)

print(accuracy_score(Y, pred))


