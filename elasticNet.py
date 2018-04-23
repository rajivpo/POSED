#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 02:30:09 2018

@author: rajivpatel-oconnor
"""
import numpy as np
import math
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

scalr = StandardScaler();

#load five folds
fold1 = np.loadtxt('../projData/ObesityCVFold_1.csv', delimiter=',', skiprows =1)
fold2 = np.loadtxt('../projData/ObesityCVFold_2.csv', delimiter=',', skiprows =1)
fold3 = np.loadtxt('../projData/ObesityCVFold_3.csv', delimiter=',', skiprows =1)
fold4 = np.loadtxt('../projData/ObesityCVFold_4.csv', delimiter=',', skiprows =1)
fold5 = np.loadtxt('../projData/ObesityCVFold_5.csv', delimiter=',', skiprows =1)


#define parameters
l1_ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
alpha = [.1, .5, .7, .9, .95, .99, 1]
weights = np.zeros([len(l1_ratio), len(alpha), 5, 123])
bias = np.zeros([len(l1_ratio), len(alpha), 5])
RMSE = np.zeros([len(l1_ratio), len(alpha), 5])


for f in range(0, 5):
    for l in range(len(l1_ratio)):
        for a in range(len(alpha)):
            if f == 0:
                train = np.concatenate((fold1, fold2, fold3, fold4))
                test = fold5
            elif f == 1:
                train = np.concatenate((fold2, fold3, fold4, fold5))
                test = fold1
            elif f == 2:
                train = np.concatenate((fold3, fold4, fold5, fold1))
                test = fold2
            elif f == 3:
                train = np.concatenate((fold4, fold5, fold1, fold2))
                test = fold3
            else:
                train = np.concatenate((fold5, fold1, fold2, fold3))
                test = fold4
                
            #train algo and get weights    
            [r, c] = np.shape(test)
            X = train[:, 0:c-1]
            y = train[:, c-1]
            #standardize X
            X = scalr.fit_transform(X)
            EN = ElasticNet(alpha=alpha[a], l1_ratio=l1_ratio[l], max_iter=3000, fit_intercept=True)
            EN.fit(X, y)
            weights[l, a, f, :] = EN.coef_
            bias[l, a, f] = EN.intercept_
            
            #calculate RMSE
            Xtest = test[:, 0:c-1]
            Xtest = scalr.fit_transform(Xtest)
            yHat = EN.predict(Xtest)
            RMSE[l, a, f] = math.sqrt(np.sum(np.power(yHat - test[:, c-1], 2))/r)
            
#determine ideal combo of alpha + l1_ratio from crossval
RMSE = np.mean(RMSE,2)
bias = np.mean(bias, 2)
weights = np.mean(weights, 2)

#get best values from 
[rmin, cmin] = np.unravel_index(RMSE.argmin(), np.shape(RMSE))
RMSEmin = RMSE[rmin, cmin]
biasMin = bias[rmin, cmin]
weightsMin = weights[rmin, cmin, :]
alphaMin = alpha[cmin]
l1_ratioMin = l1_ratio[rmin]


#load testData and test with appropriate vals
testData = np.loadtxt('../projData/ObesityTestSet.csv', delimiter=',', skiprows = 1)
[r, c] = testData.shape
Xtest = testData[:, 0:c - 1]
Xtest = scalr.fit_transform(Xtest)
yHatTest = np.matmul(Xtest, weightsMin) + biasMin
RMSEtest = math.sqrt(np.sum(np.power(yHatTest - testData[:, c-1], 2))/r)

#save weights as csv
np.savetxt('enetW.csv', weightsMin, delimiter=',')