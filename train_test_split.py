import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()

X = iris.data
Y = iris.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

print(Xtrain.shape, Xtest.shape)
