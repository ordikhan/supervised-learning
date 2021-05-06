import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

url = 'http://dataqueez.ir/files/questions/train(1).csv'
df = pd.read_csv(url)

#print(df.head())

dfx = df.iloc[:,2:24]
dfy = df.iloc[:,1]

lblenc = LabelEncoder()
for c in dfx.columns:
    dfx[c] = lblenc.fit_transform(dfx[c])


#print(dfx.head())

model = DecisionTreeClassifier()
scores = cross_val_score(model, dfx, dfy, cv=10)
print(np.mean(scores))


pred = cross_val_predict(model, dfx, dfy, cv=10)
cf = confusion_matrix(dfy, pred)
print(cf)


url_t = 'http://dataqueez.ir/files/questions/test(1).csv'
test = pd.read_csv(url_t)

testx = test.iloc[:,1:23]
testid = test.iloc[:,0]

lblenc = LabelEncoder()
for c in testx.columns:
    testx[c] = lblenc.fit_transform(testx[c])


model.fit(dfx, dfy)
pred = model.predict(testx)

df_pred = pd.DataFrame(pred)
df_pred.to_csv('C:/Users/e.almaee/Desktop/data mining python 4/result.csv', header=None, index=False)


