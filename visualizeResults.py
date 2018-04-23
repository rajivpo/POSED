#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 13:26:05 2018

@author: rajivpatel-oconnor
"""
import numpy as np
import matplotlib.pyplot as plt

#change each time script is run 
filepath = './SVR/svrWeights.csv'; #filepath to csv containing Weights
model = 'SVR' #change this every time to the model in question
shortmodel = ''.join(model.split())

#load weights
weights = np.loadtxt(filepath, delimiter=',')

#Get num of features
d = len(weights)
x = np.arange(1, d+1) #start call first feature '1' instead of '0'

#Visualize results
plt.bar(x, weights) #makes bar plot
plt.xlabel('Feature Index') #x-axis label
plt.ylabel('Magnitude') #y-axis label
plt.xlim((1, d+1)) #range to visualize x-axis
plt.title(model + ' ' + 'Weight Visualization') # DO NOT MODIFY THIS LINE
plt.savefig(shortmodel + '_weight_visualization.png', dpi=1000)