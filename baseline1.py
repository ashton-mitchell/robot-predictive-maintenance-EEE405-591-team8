# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 23:08:21 2025

@author: karth
"""

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
df = pd.read_csv("./results/train_snapshot.csv")
# ps = MinMaxScaler()
df = df.dropna()
X = df.loc[:,'crack length (arbitary unit)']
X_1 = X.values
X_1 = X_1.reshape(-1, 1)
X = X_1
# ps.fit(X)
# X = ps.transform(X)
y1 = df.loc[:,'ttf_months'] 
y2 = df.loc[:,'fail_in_4m']


predicted_failure = Ridge(alpha=0)

predicted_failure.fit(X,y1)
w = predicted_failure.coef_
yP = predicted_failure.predict(X)
mse = mean_squared_error(y1, yP)
r_squared = predicted_failure.score(X, y1)

print("Predictions", yP)

print("Error in Predicted Failure Months:", mse)
print("R-Squared:", r_squared)
print("Weights: ", w)
log = LogisticRegression(class_weight='balanced', C=10000)
log.fit(X,y2)
y2P = log.predict(X)
mse2 = log_loss(y2, y2P)
print("Logistic Regression:", y2P)
print("Error in Logistic Regression:", mse2)

plt.plot(X,yP)
plt.show()
plt.show()
plt.plot(yP, y1)

