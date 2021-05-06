import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('C:/Users/e.almaee/Desktop/Dataset/cars.xls')
#print(df.head())

X = df[['Mileage', 'Liter', 'Doors']]
Y = df['Price']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3)

model = sm.OLS(Ytrain, Xtrain).fit()
pred = model.predict(Xtest)
mse = mean_squared_error(Ytest, pred)
rmse = sqrt(mse)
print(mse, rmse)

