from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error

X = np.arange(1,10,0.1)
Y = X*2 - 4.5
noise = np.random.normal(0,0.01,len(X))
Y = Y+noise

slope , intercept, r, p, s = stats.linregress(X, Y)

def predict(samples):
    return samples*slope+intercept


Ypred = predict(X)
#print(Y)
#print(Ypred)

MSE = mean_squared_error(Y, Ypred)
print(MSE)