#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:38:42 2017

@author: abk
"""
#%%
import numpy as np
import math
import random
import matplotlib.pyplot as plt

#%%
#def sigmoid(x):
#    res = np.zeros(np.size(x))
#    for i in range(np.size(x)):
#        res[i] = 1/(1+math.exp(-x[i]))
#    return res
    
def sigmoid(x):
    return 1/(1+math.exp(-x))
    
def get_gaussian(m1,m2,N):
    #generate gaussian distribution
    cov = [[1,0],[0,1]]
    c1 = np.random.multivariate_normal(m1, cov, N)
    c2 = np.random.multivariate_normal(m2, cov, N)
    data = np.append(c1,c2,axis=0)
    label = np.append(np.zeros(N),np.ones(N),axis=0)
    return data,label

#def hypo(x,theta):
#    x = list(x)
#    x.insert(0,1)
#    x = np.array(x)
#    return sum(x*theta)
    
def plus_one(data):
    #this function add a 1 to each sample
    num = np.size(data[:,0])
    new_data = []
    for i in range(num):
        x = list(data[i])
        x.append(1)
        x = np.array(x)
        new_data.append(x)
    new_data = np.array(new_data)
    return new_data
 
def logistic(data,label):
    data = plus_one(data)   #data dimension + 1
    dim = np.size(data[0])
    num = np.size(label)
    theta = np.ones(dim)    #theta
    alpha = 0.01       #learning rate
#    k = 10000        #round of iteration
#    for i in range(k):
    while True:
        dif = np.zeros(dim)
        for j in range(num):
            dif = dif + (sigmoid(sum(data[j] * theta))-label[j]) * data[j]
        print(dif)
        if sum(abs(dif))/np.size(dif) < 0.01:    #break when gradient vector is small
            break
        theta = theta - alpha * dif
        print("theta:",theta)
    return theta
#%% generate gaussian data
N = 10
data,label = get_gaussian([5,5],[6,7],N)
#%% train classifier
theta = logistic(data,label)
#%% plot 
plt.scatter(data[0:N:,0],data[0:N:,1],c='r')
plt.scatter(data[N:2*N:,0],data[N:2*N:,1],c='b')
x1 = 0
x2 = 10
plt.plot([x1,x2],[-(x1 * theta[0]/theta[1])-(theta[2]/theta[1]),-(x2 * theta[0]/theta[1])-(theta[2]/theta[1])],'k-')