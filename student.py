import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('C:/Users/e.almaee/Desktop/Dataset/train.csv', sep=',')
#print(df.head())
#print(df.dtypes)

df = pd.get_dummies(df, drop_first=True)

dfx = df.ix[:,df.columns != 'G3']
dfy = df['G3']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(dfx, dfy, test_size=0.25)

model = LinearRegression()
model = model.fit(Xtrain, Ytrain)
pred = model.predict(Xtest)
mse = mean_squared_error(Ytest, pred)
rmse = np.sqrt(mse)
print(rmse)


# answer test file
test = pd.read_csv('C:/Users/e.almaee/Desktop/Dataset/test.csv', sep=',')

test = pd.get_dummies(test, drop_first=True)
pred_test = model.predict(test)

z = np.zeros(len(test))
pred_test = np.maximum(z,pred_test)

np.savetxt('C:/Users/e.almaee/Desktop/Dataset/result.csv', pred_test, delimiter=',')

