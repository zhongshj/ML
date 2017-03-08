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
        
def cut_error(cut,array,label,p):
    #calculate the cut error
    #implement 2 error variable in case making completely opposite classification
    error1 = 0
    error2 = 0
    for i in list(range(np.size(array))):
        if label[i]*(cut - array[i]) >= 0:           
            error1 = error1 + p[i]            
        
        if label[i]*(cut - array[i]) <= 0:
            error2 = error2 + p[i]      
            
    #we take the smaller one to return the error
    if error1 <= error2:
        order = -1  #means small -> -1, large -> 1
        return error1, order
    else:
        order = 1   #small -> 1, large -> -1
        return error2, order

def get_cut(data):
    #this function returns middle position of every pair of neighbours
    #it is a good searching strategy for stump
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
    #make sure the error update at the first time
    min_error = 100000
    for i in list(range(np.size(data.T[0]))):#i indicates the dimension(feature)       
        cuts = get_cut(data[i])
        for j in cuts:#j goes through the cut set for one feature
            error, order = cut_error(j,data[i],label,p)
            #get error, cut, dimension and order for the best cut          
            if error < min_error:
                min_error = error
                min_cut_dimension = i
                min_cut_position = j
                min_cut_order = order
            
    return min_cut_dimension, min_cut_position, min_error, min_cut_order
     
def get_err_index(array,label,cut,order):
    #calculate misclassified nodes. Initialize with 0 and mark mistaken index as 1
    err_index = np.zeros(np.size(array))
    for i in list(range(np.size(array))):
        if label[i]*order*(cut - array[i]) < 0:
            err_index[i] = 1
        
    return err_index
   
def get_ada(data,label,rounds):
    #this function trains adaboost classifier with given training set
    
    #following part is just for visualizing sample points
    data0 = []
    data1 = []
    for i in range(0,np.size(data[0])):
        if label[i] < 0:
            data0.append(data[:,i])
        else:
            data1.append(data[:,i])
    data0 = np.array(data0).T
    data1 = np.array(data1).T
    #the visualization here is for 2d data
    plt.scatter(data0[0],data0[1],c='r')
    plt.scatter(data1[0],data1[1],c='b')
    
    #initialize weight
    #the parameters of weak-classifiers will be stored in these 4 lists
    weight = np.ones(np.size(data[0]))/np.size(data[0])
    list_dim = []
    list_cut = []
    list_beta = []
    list_order = []

    #start adaboost
    for i in list(range(rounds)):
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
    
        #store this weak-classifier in lists
        list_dim.append(dim)
        list_cut.append(cut)
        list_beta.append(error/(1-error))
        list_order.append(order)
        
    return list_dim, list_cut, list_beta, list_order
    
def ada_classifier(dim,cut,beta,order,observation):
    #this function is for predicting label for one sample
    #I used -1 and 1 as labels, so here we just judge >0 or <0 (different from the paper)
    sum_decision = 0
    for i in list(range(np.size(dim))):
        sum_decision = sum_decision + math.log(1/beta[i],2)*order[i]*np.sign(cut[i]-observation[dim[i]])
    return new_sign(sum_decision)
    
def test_ada(data, label, list_dim, list_cut, list_beta, list_order):
    #this function is for calculating accuracy for ada-classifiers
    return_label = np.ones(np.size(data[0]))
    correct_count = 0
    for i in list(range(np.size(data[0]))):
        return_label[i] = ada_classifier(list_dim,list_cut,list_beta,list_order,data.T[i])
        print("true: ",label[i]," test: ",return_label[i])
        if return_label[i] == label[i]:
            correct_count = correct_count + 1
    rate = correct_count/np.size(data[0])
    print("correct rate: ",rate)
    return rate

#%% gaussian data(here are 3 options for generating data)
#generate gaussian data
m1 = [0,0]
m2 = [2,2]
cov = [[1,0],[0,1]]
N = 20
train0 = np.random.multivariate_normal(m1, cov, N).T
train1 = np.random.multivariate_normal(m2, cov, N).T
test0 = np.random.multivariate_normal(m1, cov, N).T
test1 = np.random.multivariate_normal(m2, cov, N).T

plt.scatter(train0[0],train0[1],c='r')
plt.scatter(train1[0],train1[1],c='b')

#set labels as -1 and 1
train_data = np.append(train0,train1,axis=1)
test_data = np.append(test0,test1,axis=1)
label1 = np.zeros(N)-1
label2 = np.ones(N)
train_label = np.append(label1,label2,axis=0)
test_label = np.append(label1,label2,axis=0)
    
#%% using handwritten digit data
data = []

train_size = 30
test_size = 50

for l in open("opt.txt"):
    row = [int(x) for x in l.split()]
    if len(row) > 0:
        data.append(row)
        
data0 = np.array(data[0:554])
data1 = np.array(data[554:1125])

train0 = data0[0:train_size].T
train1 = data1[0:train_size].T
train_data = np.append(train0,train1,axis=1)
label1 = np.zeros(train_size)-1
label2 = np.ones(train_size)
train_label = np.append(label1,label2,axis=0)

test0 = data0[train_size:train_size+test_size].T
test1 = data1[train_size:train_size+test_size].T
test_data = np.append(test0,test1,axis=1)
label1 = np.zeros(test_size)-1
label2 = np.ones(test_size)
test_label = np.append(label1,label2,axis=0)

#%% simple data(6 samples)
data0 = np.array([[1,1,2],[1,2,2]])
data1 = np.array([[2,3,3],[1,1,2]])
data = np.append(data0,data1,axis=1)
label = np.array([-1,-1,-1,1,1,1])

#%% generate ada classifier
list_dim, list_cut, list_beta, list_order = get_ada(train_data,train_label,10)
#%% test
accuracy = test_ada(test_data,test_label,list_dim, list_cut, list_beta, list_order)
    

