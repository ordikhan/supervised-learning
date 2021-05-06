from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error

X = np.arange(1,10,0.1)
Y = 2*X*X*X - 4.5*X + 10.25
noise = np.random.normal(0,0.01,len(X))
Y = Y + noise

p2 = np.poly1d(np.polyfit(X,Y,2))
pred2 = p2(X)
p3 = np.poly1d(np.polyfit(X,Y,3))
pred3 = p3(X)

MSE2 = mean_squared_error(Y, pred2)
MSE3 = mean_squared_error(Y, pred3)

print(MSE2, MSE3)
