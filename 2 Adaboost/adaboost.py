#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def new_sign(x):
    #when cutting through a sample, it is still error
    if x >= 0:
        return 1
    else:
        return -1
        

def min_cut(cut,array,label,p):
    error1 = 0
    error2 = 0
    for i in list(range(np.size(array))):
        if label[i]*(cut - array[i]) >= 0:
            #make sure the label correspond well to their position
            error1 = error1 + p[i]
        
        if label[i]*(cut - array[i]) <= 0:
            error2 = error2 + p[i]           
            #implement 2 error variable in case making completely opposite classification
            #we take the smaller one to return the error
    #print(cut,error1,error2)
    return min(error1,error2)

def stump(data,label,p):
    #make sure it update at the first time
    min_error = 1000 
    step_length = 0.5
    for i in list(range(np.size(data.T[0]))):#i indicates the dimension(feature)
        #make sure we scan it thoroughly
        ceil = math.ceil(max(data[i]))+1
        floor = math.floor(min(data[i]))-1
        steps = np.arange(floor,ceil,step_length)#adjust the step length here
        
        for j in steps:#j goes through one feature of all samples
            error = min_cut(j,data[i],label,p)
            #get min error and corresponding dimension and cut position
            if error < min_error:
                min_error = error
                min_cut_dimension = i
                min_cut_position = j
            #print(i,j,error)
    return min_cut_dimension, min_cut_position, min_error
    
    
def get_err_index(array,label,cut):
    #calculate misclassified nodes. Initialize with 0 and mark mistaken index as 1
    err_index = np.zeros(np.size(array))
    for i in list(range(np.size(array))):
        if label[i]*(cut - array[i]) >= 0:
            err_index[i] = 1
    
    #if more than 1/2 samples are misclassified, we just change the labels
    if sum(err_index) > 0.5 * np.size(err_index):
        err_index = 1 - err_index
        
    return err_index
    

    
#%%
    
#data = np.array([[1,3,5,2,4,6],[1,1,1,1,1,1]])
#label = np.array([-1,-1,-1,1,1,1])
#weight = np.array([1,1,1,1,1,1])

#a,b = stump(data,label)

#%% example of gaussian data
#generate gaussian data
m1 = [0,0]
m2 = [2,0]
cov = [[1,0],[0,1]]
x1 = np.random.multivariate_normal(m1, cov, 50).T
x2 = np.random.multivariate_normal(m2, cov, 50).T
plt.scatter(x1[0],x1[1],c='r')
plt.scatter(x2[0],x2[1],c='b')

#set labels as -1 and 1
x = np.append(x1,x2,axis=1)
label1 = np.zeros(50)-1
label2 = np.ones(50)
label = np.append(label1,label2,axis=0)

weight = np.ones(50)
p = weight/sum(weight)
dim,cut,error = stump(x,label,p)
err_index = get_err_index(x[dim],label,cut)

#plot decision boundary
if dim == 0:
    plt.axvline(x=cut)
else:
    plt.axhline(y=cut)
    
#%% using handwritten digit data

data = []

for l in open("opt.txt"):
    row = [int(x) for x in l.split()]
    if len(row) > 0:
        data.append(row)
        
data0 = np.array(data[0:554])
data1 = np.array(data[554:1125])
train0 = data0[0:50].T
train1 = data1[0:50].T

data = np.append(train0,train1,axis=1)
label1 = np.zeros(50)-1
label2 = np.ones(50)
label = np.append(label1,label2,axis=0)
weight = np.ones(np.size(data[0]))/np.size(data[0])


#%% Adaboost
#data = x
#weight = np.ones(np.size(data[0]))/np.size(data[0])

data = np.array([[1,1,2,2,3,3],[1,2,2,1,1,2]])
label = np.array([-1,-1,-1,1,1,1])
weight = np.ones(np.size(data[0]))/np.size(data[0])
#%%
for i in list(range(100)):
    p = weight/sum(weight)
    dim,cut,error = stump(data,label,p)
    err_index = get_err_index(data[dim],label,cut)
    weight = weight * (error/(1-error))**(1-err_index)
    print(i,":  dim:",dim,"  cut:",cut,"  error:",error)
    
    
    
    
    