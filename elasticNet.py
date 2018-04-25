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
import time

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
            #standardize y
            mu = np.mean(y)
            std = np.std(y)
            y = (y - mu)/std
            EN = ElasticNet(alpha=alpha[a], l1_ratio=l1_ratio[l], max_iter=3000, fit_intercept=True)
            EN.fit(X, y)
            weights[l, a, f, :] = EN.coef_
            bias[l, a, f] = EN.intercept_
            
            #calculate RMSE
            Xtest = test[:, 0:c-1]
            Xtest = scalr.fit_transform(Xtest)
            yHat = EN.predict(Xtest)
            yHat = yHat*std + mu;
            RMSE[l, a, f] = math.sqrt(np.sum(np.power(yHat - test[:, c-1], 2))/r)
            
#determine ideal combo of alpha + l1_ratio from crossval
RMSE = np.mean(RMSE,2)

#get best values from 
[rmin, cmin] = np.unravel_index(RMSE.argmin(), np.shape(RMSE))
RMSEmin = RMSE[rmin, cmin]
alphaMin = alpha[cmin]
l1_ratioMin = l1_ratio[rmin]

#load trainData
trainData = np.loadtxt('../projData/ObesityFullTrainSet.csv', delimiter=',', skiprows = 1)
[r, c] = trainData.shape
Xtrain = trainData[:,0:c-1]
yTrain = trainData[:,c-1]
Xtrain = scalr.fit_transform(Xtrain)
std = np.std(yTrain)
mu = np.mean(yTrain)
yTrain = (yTrain-mu)/std

#Train model with parameters learned from cross-val
EN = ElasticNet(alpha = alphaMin, l1_ratio=l1_ratioMin, max_iter=3000, fit_intercept=True)
tot = 0;
for u in range(0, 100):
    starttime = time.time()
    EN.fit(Xtrain, yTrain)
    tot_time = time.time() - starttime
    tot += tot_time
tot/=100
EN.fit(Xtrain, yTrain)
trainW = EN.coef_
trainB = EN.intercept_
yHatTrain = np.matmul(Xtrain, trainW) + trainB
yHatTrain = yHatTrain*std + mu
trainError = math.sqrt(np.sum(np.power(yHatTrain - trainData[:, c-1], 2))/r)

#load testData and test with appropriate vals
testData = np.loadtxt('../projData/ObesityTestSet.csv', delimiter=',', skiprows = 1)
[r, c] = testData.shape
Xtest = testData[:, 0:c - 1]
Xtest = scalr.fit_transform(Xtest)
tot2 = 0
for v in range(0,100):
    starttime2 = time.time()
    yHatTest = np.matmul(Xtest, trainW) + trainB
    tot_time2 = time.time() - starttime2
    tot2 += tot_time2
tot2/=100
yHatTest = np.matmul(Xtest, trainW) + trainB
yHatTest = yHatTest*std + mu
RMSEtest = math.sqrt(np.sum(np.power(yHatTest - testData[:, c-1], 2))/r)
#save weights as csv
np.savetxt('enetW.csv', trainW, delimiter=',')