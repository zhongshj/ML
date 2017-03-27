#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:18:31 2017

@author: abk
"""


#%%
import numpy as np
import matplotlib.pyplot as plt
import math
import random
#%%

def get_gaussian(m1,m2,N):
    #generate gaussian distribution
    cov = [[1,0],[0,1]]
    c1 = np.random.multivariate_normal(m1, cov, N)
    c2 = np.random.multivariate_normal(m2, cov, N)
    data = np.append(c1,c2,axis=0)
    label = np.append(np.zeros(N)-1,np.ones(N),axis=0)
    return data,label

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
    
def perceptron(data,label):
    dim = np.size(data,1)
    num = np.size(data,0)
    w = np.ones(dim)
    alpha = 0.01
    mis = True
    while mis:
        li = np.arange(num) #create a random list
        random.shuffle(li)
        mis = False
        for j in li:    #check miclassified labels randomly
            if label[j] * sum(data[j] * w) < 0:
                w = w + alpha * label[j] * data[j]
                print("w:",w)
                mis = True
                break
    
    return w
    
#%%
N = 10
m1 = [1,3]
m2 = [3,1]
data, label = get_gaussian(m1,m2,N)
data = plus_one(data)
plt.scatter(data[0:N:,0],data[0:N:,1],c='r')
plt.scatter(data[N:2*N:,0],data[N:2*N:,1],c='b')
#%%
w = perceptron(data,label)
plt.scatter(data[0:N:,0],data[0:N:,1],c='r')
plt.scatter(data[N:2*N:,0],data[N:2*N:,1],c='b')
x1 = 0
x2 = 5
plt.plot([x1,x2],[-(x1 * w[0]/w[1])-(w[2]/w[1]),-(x2 * w[0]/w[1])-(w[2]/w[1])],'k-')
    