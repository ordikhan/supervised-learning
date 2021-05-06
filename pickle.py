import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import random
import pickle

 

temp = open('C:/Users/e.almaee/Desktop/Dataset/svm.pickle', 'rb')
model = pickle.load(temp)

pred = model.predict([[0.3, 0.5, 1.4, 4], [1.3, 2.5, 0.4, 1], [1.2, 2.5, 3.4, 1]])
#pred2 = model.predict([1.3, 2.5, 0.4, 4])

print(pred)