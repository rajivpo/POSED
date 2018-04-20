#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 21:34:44 2018

@author: rajivpatel-oconnor
"""

import numpy as np
import math
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler

scalr = StandardScaler();
epsilon = np.linspace(0, .9, num=10)
C = np.logspace(-4, 2, num=7)

#load five folds
fold1 = np.loadtxt('../projData/ObesityCVFold_1.csv', delimiter=',', skiprows =1)
fold2 = np.loadtxt('../projData/ObesityCVFold_2.csv', delimiter=',', skiprows =1)
fold3 = np.loadtxt('../projData/ObesityCVFold_3.csv', delimiter=',', skiprows =1)
fold4 = np.loadtxt('../projData/ObesityCVFold_4.csv', delimiter=',', skiprows =1)
fold5 = np.loadtxt('../projData/ObesityCVFold_5.csv', delimiter=',', skiprows =1)


#define parameters
weights = np.zeros([len(epsilon), len(C), 5, 123])
bias = np.zeros([len(epsilon), len(C), 5])
err = np.zeros([len(epsilon), len(C), 5])


for f in range(0, 5):
    for e in range(len(epsilon)):
        for c in range(len(C)):
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
             
             #standardize X
            [r, col] = np.shape(test)
            X = train[:, 0:col-1]
            y = train[:, col-1]
            X = scalr.fit_transform(X)
            
            #train algo and get weights  
            mdl = LinearSVR(C=C[c], epsilon=epsilon[e])
            mdl.fit(X, y)
            weights[e, c, f, :] = mdl.coef_
            bias[e, c, f] = mdl.intercept_
            
            
            #calculate RMSE
            Xtest = test[:, 0:col-1]
            Xtest = scalr.fit_transform(Xtest)
            yHat = mdl.predict(Xtest)
            err[e, c, f] = math.sqrt(np.sum(np.power(yHat - test[:, col-1], 2))/r)
            
#determine ideal combo of alpha + l1_ratio from crossval
err = np.mean(err,2)
bias = np.mean(bias, 2)
weights = np.mean(weights, 2)

#get best values from 
[rmin, cmin] = np.unravel_index(err.argmin(), np.shape(err))
errmin = err[rmin, cmin]
biasMin = bias[rmin, cmin]
weightsMin = weights[rmin, cmin, :]
Cmin = C[cmin]
epsilonmin = epsilon[rmin]


#load testData and test with appropriate vals
testData = np.loadtxt('../projData/ObesityTestSet.csv', delimiter=',', skiprows = 1)
[r, col] = testData.shape
Xtest = testData[:, 0:col - 1]
Xtest = scalr.fit_transform(Xtest)
yHatTest = np.matmul(Xtest, weightsMin) + biasMin
errTest = math.sqrt(np.sum(np.power(yHatTest - testData[:, col-1], 2))/r)


