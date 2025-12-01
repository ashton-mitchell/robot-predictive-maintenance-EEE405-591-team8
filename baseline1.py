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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss

df = pd.read_csv("./results/train_snapshot.csv")
ps = MinMaxScaler()
df = df.dropna()
X = df.loc[:,'time (months)':'crack length (arbitary unit)']
ps.fit(X)
X = ps.transform(X)
y1 = df.loc[:,'ttf_months'] 
y2 = df.loc[:,'fail_in_4m']


predicted_failure = Ridge(alpha=0.0)

predicted_failure.fit(X,y1)
yP = predicted_failure.predict(X)
mse = mean_squared_error(y1, yP)
print("Predictions", yP)

print("Error in Predicted Failure Months:", mse)

log = LogisticRegression(class_weight='balanced', solver = 'saga')
log.fit(X,y2)
y2P = log.predict(X)
mse2 = log_loss(y2, y2P)
print("Logistic Regression:", y2P)
print("Error in Logistic Regression:", mse2)




