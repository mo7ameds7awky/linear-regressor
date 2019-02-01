# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 16:18:42 2018

@author: Abdallah Tarek& Mohamed Shawky
"""

# Imports
import pandas as pd
import numpy as np

# Get data
train = pd.read_csv('dataset/reg_train.csv')

# Drop un-useful columns
train =train.drop(['dteday','casual','registered'], axis=1)
x_train = train.loc[:,'instant':'windspeed']
y_train = train.loc[:,'cnt']

# Create vector of ones
one = np.ones([13903,1])

# Convert the vector to pandas dataframe
one = pd.DataFrame(one,dtype=int)

# Concatinate the ones column to the data
x=pd.concat([one, x_train],axis=1)

# Converting data to numpy matrix for the sake of calculation on the data
x=np.matrix(x)
y = np.matrix(y_train)

# Reshaping the data
y = y.reshape(13903,1)
x_t=x.reshape(14,13903)

# Building the mathematical model
xTx = np.dot(x_t,x)
xTx_Inv = np.linalg.inv(xTx)
xTx_Inv_xT = np.dot(xTx_Inv,x_t)
Beta = np.dot(xTx_Inv_xT,y)

# Starting the testing phase

# Get test data
test =pd.read_csv('daatset/reg_test.csv')
inst = test.loc[:,'instant']

# Drop the un-useful data
test =test.drop(['dteday'], axis=1)

# Create the ones column and concatinate it with the data
one = np.ones([3476,1])
one = pd.DataFrame(one,dtype=int)
test=pd.concat([one, test],axis=1)

# Converting data to numpy matrix for the sake of calculation on the data
test=np.matrix(test)

# Predict
y_hat = np.dot(test,Beta)
y_hat = pd.DataFrame(y_hat)
y_hat = pd.concat([inst , y_hat] , axis = 1)
