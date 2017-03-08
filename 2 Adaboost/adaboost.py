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
    if error1 <= error2:
        order = -1  #means small - -1, large - 1
        return error1, order
    else:
        order = 1
        return error2, order

def get_cut(data):
    ceil = math.ceil(max(data))+1
    floor = math.floor(min(data))-1
    steps = [ceil,floor]
    for i in range(0,len(data)):
        steps.append(data[i])
    steps.sort()
    cuts = []
    for i in range(0,len(steps)-1):
        cuts.append((steps[i]+steps[i+1])/2)
    return cuts
        
def stump(data,label,p):
    #make sure it update at the first time
    min_error = 100000
    step_length = 0.01
    for i in list(range(np.size(data.T[0]))):#i indicates the dimension(feature)
        #make sure we scan it thoroughly
        cuts = get_cut(data[i])
        #adjust the step length here
        
        for j in cuts:#j goes through one feature of all samples
            error, order = min_cut(j,data[i],label,p)
            #get min error and corresponding dimension and cut position
            if error < min_error:
                min_error = error
                min_cut_dimension = i
                min_cut_position = j
                min_cut_order = order
            
            #print(i,j,error)
    return min_cut_dimension, min_cut_position, min_error, min_cut_order
    
    
def get_err_index(array,label,cut,order):
    #calculate misclassified nodes. Initialize with 0 and mark mistaken index as 1
    err_index = np.zeros(np.size(array))
    for i in list(range(np.size(array))):
        if label[i]*order*(cut - array[i]) < 0:
            err_index[i] = 1
    
    #if more than 1/2 samples are misclassified, we just change the labels
    if sum(err_index) > 0.5 * np.size(err_index):
        err_index = 1 - err_index
        
    return err_index
   
def get_ada(data,label):
    
    plt.scatter(data0[0],data0[1],c='r')
    plt.scatter(data1[0],data1[1],c='b')

    weight = np.ones(np.size(data[0]))/np.size(data[0])
    list_dim = []
    list_cut = []
    list_beta = []
    list_order = []

    for i in list(range(10)):
        p = weight/sum(weight)
        dim,cut,error,order = stump(data,label,p)
        err_index = get_err_index(data[dim],label,cut,order)
        weight = weight * (error/(1-error))**(1-err_index)
        print(": dim:",dim," cut:",cut," error:",error," order:",order)
        #plot decision boundary
        if dim == 0:
            plt.axvline(x=cut)
        else:
            plt.axhline(y=cut)
    
        list_dim.append(dim)
        list_cut.append(cut)
        list_beta.append(error/(1-error))
        list_order.append(order)
        
    return list_cut
    
def ada_classifier(dim,cut,beta,order,observation):
    sum_decision = 0
    for i in list(range(np.size(dim))):
        sum_decision = sum_decision + math.log(1/beta[i],2)*order[i]*np.sign(cut[i]-observation[dim[i]])
        #print(i,"vote:",sum_decision," beta:",beta[i])
    return new_sign(sum_decision)
#%%
    
#data = np.array([[1,3,5,2,4,6],[1,1,1,1,1,1]])
#label = np.array([-1,-1,-1,1,1,1])
#weight = np.array([1,1,1,1,1,1])

#a,b = stump(data,label)

#%% gaussian data
#generate gaussian data
m1 = [0,0]
m2 = [2,2]
cov = [[1,0],[0,1]]
N = 10
data0 = np.random.multivariate_normal(m1, cov, N).T
data1 = np.random.multivariate_normal(m2, cov, N).T
plt.scatter(data0[0],data0[1],c='r')
plt.scatter(data1[0],data1[1],c='b')

#set labels as -1 and 1
data = np.append(data0,data1,axis=1)
label1 = np.zeros(N)-1
label2 = np.ones(N)
label = np.append(label1,label2,axis=0)



#plot decision boundary
#if dim == 0:
#    plt.axvline(x=cut)
#else:
#    plt.axhline(y=cut)
    
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


#%% simple data
#data = x
#weight = np.ones(np.size(data[0]))/np.size(data[0])
data0 = np.array([[1,1,2],[1,2,2]])
data1 = np.array([[2,3,3],[1,1,2]])
data = np.append(data0,data1,axis=1)
label = np.array([-1,-1,-1,1,1,1])
weight = np.ones(np.size(data[0]))/np.size(data[0])
#%% Adaboost
#
#plt.scatter(data0[0],data0[1],c='r')
#plt.scatter(data1[0],data1[1],c='b')
#
#weight = np.ones(np.size(data[0]))/np.size(data[0])
#list_dim = []
#list_cut = []
#list_beta = []
#list_order = []
#
#for i in list(range(10)):
#    p = weight/sum(weight)
#    dim,cut,error,order = stump(data,label,p)
#    err_index = get_err_index(data[dim],label,cut,order)
#    weight = weight * (error/(1-error))**(1-err_index)
#    print(": dim:",dim," cut:",cut," error:",error," order:",order)
#    #plot decision boundary
#    if dim == 0:
#        plt.axvline(x=cut)
#    else:
#        plt.axhline(y=cut)
#  
# #  list_dim.append(dim)
#    list_cut.append(cut)
#    list_beta.append(error/(1-error))
#    list_order.append(order)
#    
#%%    
    
return_label = np.ones(np.size(data[0]))
correct_count = 0
for i in list(range(np.size(data[0]))):
    return_label[i] = ada_classifier(list_dim,list_cut,list_beta,list_order,data.T[i])
    print("true: ",label[i]," test: ",return_label[i])
    if return_label[i] == label[i]:
        correct_count = correct_count + 1

print("correct rate: ",correct_count/np.size(data[0]))
    