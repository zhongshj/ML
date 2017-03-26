#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 10:07:42 2017

@author: abk
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
PI = 3.14159265359

#%%
def get_gaussian(m1,m2,N):
    #generate gaussian distribution
    cov = [[1,0],[0,1]]
    c1 = np.random.multivariate_normal(m1, cov, N)
    c2 = np.random.multivariate_normal(m2, cov, N)
    data = np.append(c1,c2,axis=0)
    label = np.append(np.zeros(N)-1,np.ones(N),axis=0)
    return data,label

def gaussian(x,u,cov):
    #calculate gaussian density
    dim = np.size(x)
    x = np.mat(x)
    u = np.mat(u)
    cov = np.mat(cov) 
    a = ((2*PI)**dim)*np.linalg.det(cov+0.000000001)
    b = (x-u)*(cov+0.000000001*np.mat(np.identity(dim))).I*(x-u).T
    gaussian = (1/math.sqrt(a))*np.exp(-0.5*b[0,0])
    return gaussian
    
def em_gaussian(data,semi_data,soft_label):
    #em for gaussian
    #not for 1-dim
    dim = np.size(data,1)
    num = np.size(data,0)
    u = np.zeros(dim)
    cov = np.mat(np.zeros([dim,dim]))
    for i in data:
        u = u + i
    for i in range(np.size(semi_data,0)):
        u = u + semi_data[i] * soft_label[i]
    u = u / (num + sum(soft_label))
    for i in data:
        cov = cov + np.mat(i - u).T * np.mat(i - u)
    for i in range(np.size(semi_data,0)):
        cov = cov + soft_label[i] * np.mat(semi_data[i] - u).T * np.mat(semi_data[i] - u)
    cov = cov / (num + sum(soft_label))
    return u, cov
    
def em(data0,data1,semi_data):
    
    semi_size = np.size(semi_data,0)
    
    #initialize soft labels
    soft_label0 = np.ones(semi_size) / 2
    soft_label1 = np.ones(semi_size) / 2

    
    for i in range(10):
        #get label
        u0, cov0 = em_gaussian(data0,semi_data,soft_label0)
        u1, cov1 = em_gaussian(data1,semi_data,soft_label1)
        
        print("u0:",u0)
        print("u1:",u1)
        
        for i in range(semi_size):
            #get likelihood as weight
            soft_label0[i] = gaussian(semi_data[i],u0,cov0)
            soft_label1[i] = gaussian(semi_data[i],u1,cov1)
            #normalize
            nor = soft_label0[i] + soft_label1[i]
            soft_label0[i] = soft_label0[i] / nor
            soft_label1[i] = soft_label1[i] / nor
         
    return u0,cov0,u1,cov1
        
    
#%%
m1 = [0,5]
m2 = [5,0]
N = 5
train_data, train_label = get_gaussian(m1,m2,N)
data0 = train_data[0:N]
data1 = train_data[N:2*N]
semi_data, semi_label = get_gaussian(m1,m2,10*N)
#set semi dataset unlabeled
semi_label = np.zeros(np.size(semi_label))

#%%
u0,cov0,u1,cov1 = em(data0,data1,semi_data)
    
    