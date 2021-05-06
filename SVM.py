import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
import pickle

random.seed(2002)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)

 
SVM = SVC(kernel='rbf', gamma=0.2, C=1.8) # sigmoid rbf poly linear
SVM = SVM.fit(Xtrain, Ytrain)

pred = SVM.predict(Xtest)
acc = accuracy_score(Ytest, pred)
print(acc)


print(SVM.support_vectors_)
print(SVM.support_)
print(SVM.n_support_)

save_classifier = open('C:/Users/e.almaee/Desktop/Dataset/svm.pickle', 'wb')
pickle.dump(SVM, save_classifier)
save_classifier.close()


