#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:11:14 2017

@author: abk
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
PI = 3.14159265359

#%%

def get_gaussian(m1,m2,cov,N):
    #generate gaussian distribution
    #cov = [[1,0],[0,1]]
    c1 = np.random.multivariate_normal(m1, cov, N)
    c2 = np.random.multivariate_normal(m2, cov, N)
    data = np.append(c1,c2,axis=0)
    label = np.append(np.zeros(N)-1,np.ones(N),axis=0)
    return data,label

def multi_eucli(x,u):
    #calculate eucli distance
    eucli = []
    for xi in x:
        eucli.append(sum((xi-u)**2))
    return eucli
    
def multi_gaussian(x,u,cov):
    #calculate gaussian density
    gaussian = []
    for xi in x:
        dim = np.size(xi)
        xi = np.mat(xi)
        u = np.mat(u)
        cov = np.mat(cov) 
        a = ((2*PI)**dim)*np.linalg.det(cov)
        b = (xi-u)*np.linalg.inv(cov)*(xi-u).T
        g = (1/math.sqrt(a))*np.exp(-0.5*b[0,0])
        gaussian.append(g)
    return gaussian
    
def ml_gaussian(data):
    #maximum likelihood estimation for gaussian
    #not for 1-dim
    dim = np.size(data,1)
    num = np.size(data,0)
    u = np.zeros(dim)
    cov = np.mat(np.zeros([dim,dim]))
    for i in data:
        u = u + i
    u = u / num
    for i in data:
        cov = cov + np.mat(i - u).T * np.mat(i - u)
    cov = cov / num
    return u, cov
    
def lda(data1,data2):
    #generate linear discriminant classifier
    dim = np.size(data1,1)
    num1 = np.size(data1,0)
    num2 = np.size(data2,0)
    cov = np.mat(np.ones([dim,dim]))
    u1 = np.zeros(dim)
    u2 = np.zeros(dim)
    for i in data1:
        u1 = u1 + i
    u1 = u1 / num1
    for i in data2:
        u2 = u2 + i
    u2 = u2 / num2
    for i in data1:
        cov = cov + np.mat(i - u1).T * np.mat(i - u1)
    for i in data2:
        cov = cov + np.mat(i - u2).T * np.mat(i - u2)
    cov = cov / (num1 + num2)
    w = np.mat(u2-u1) * cov.I
    w0 = -(w * np.mat(u1 + u2).T)[0,0] / 2
    print(u1,u2)
    return np.array(w)[0,:], w0
    #return u1, u2, cov
    
def semi_nearest(train_data,train_label,test_data,test_label):
    data = list(train_data)
    data.extend(list(semi_data))
    label = list(train_label)
    label.extend(list(semi_label))
    
    data_p = [] #positive class
    data_n = [] #negative class
    data_s = [] #unlabeled
    for i in range(len(label)):
        if label[i] == -1:
            data_n.append(data[i])
        elif label[i] == 1:
            data_p.append(data[i])
        else:
            data_s.append(data[i])
    
#    data_n = list(data_n)
#    data_p = list(data_p)
#    data_s = list(data_s)

    #assign labels
    while len(data_s) > 0:
        u_n, cov_n = ml_gaussian(data_n)
        u_p, cov_p = ml_gaussian(data_p)
        euc_n = multi_eucli(data_s,u_n)
        euc_p = multi_eucli(data_s,u_p)
        if min(euc_n) < min(euc_p):
            idx = euc_n.index(min(euc_n))
            data_n.append(data_s[idx])
            #print("n",idx)
            data_s.pop(idx)
        else:
            idx = euc_p.index(min(euc_p))
            data_p.append(data_s[idx])
            #print("p",idx)
            data_s.pop(idx)
            
    return data_n, data_p

    
def semi_gaussian(train_data,train_label,test_data,test_label):
    data = list(train_data)
    data.extend(list(semi_data))
    label = list(train_label)
    label.extend(list(semi_label))
    
    data_p = []
    data_n = []
    data_s = []
    for i in range(len(label)):
        if label[i] == -1:
            data_n.append(data[i])
        elif label[i] == 1:
            data_p.append(data[i])
        else:
            data_s.append(data[i])
#    data_n = list(data_n)
#    data_p = list(data_p)
#    data_s = list(data_s)
    
    
    while len(data_s) > 0:
        u_n, cov_n = ml_gaussian(data_n)
        u_p, cov_p = ml_gaussian(data_p)
        gau_n = multi_gaussian(data_s,u_n,cov_n)
        gau_p = multi_gaussian(data_s,u_p,cov_n)
        if max(gau_n) > max(gau_p):
            idx = gau_n.index(max(gau_n))
            data_n.append(data_s[idx])
            #print("n",idx)
            data_s.pop(idx)
        else:
            idx = gau_p.index(max(gau_p))
            data_p.append(data_s[idx])
            #print("p",idx)
            data_s.pop(idx)
            
    return data_n, data_p
    
def divide_spam(spam_n, spam_p, semi):
    test_size = 100
    r_n = np.random.choice(len(spam_n), 75+semi+test_size, replace=False)
    r_p = np.random.choice(len(spam_p), 75+semi+test_size, replace=False)
    train_n = spam_n[r_n[0:75]]
    train_p = spam_p[r_p[0:75]]
    semi_n = spam_n[r_n[75:75+semi]]
    semi_p = spam_p[r_p[75:75+semi]]
    test_n = spam_n[r_n[75+semi:]]
    test_p = spam_p[r_p[75+semi:]]
    return train_n, train_p, semi_n, semi_p, test_n, test_p
    
def count_error(array_n,array_p):
    true_error_n = len([x for x in array_n if x < 0])/len(array_n)
    true_error_p = len([x for x in array_p if x < 0])/len(array_p)
    return (true_error_n + true_error_p) / 2
#%% simple dataset
m1 = [1,4]
m2 = [4,1]
N = 5
train_data, train_label = get_gaussian(m1,m2,N)
semi_data, semi_label = get_gaussian(m1,m2,2*N)
#set semi dataset unlabeled
semi_label = np.zeros(np.size(semi_label))

#%% example for exercise 4
m1 = [0,0]
m2 = [20,0]
m3 = [0,5]
m4 = [20,5]
cov = [[20,0],[0,1]]
N = 300
M = 75
train_data, train_label = get_gaussian(m1,m4,cov,N)
semi_data, semi_label = get_gaussian(m3,m2,cov,N)
semi_label = np.zeros(np.size(semi_label))
train_label[0:M] = -1
train_label[N:N+M] = 1

#%% nearest method
data_n, data_p = semi_nearest(train_data,train_label,semi_data,semi_label)

#%% gaussian method
data_n, data_p = semi_gaussian(train_data,train_label,semi_data,semi_label)

#%% plot for exercise 4(before)
plt.scatter(train_data[0:M:,0],train_data[0:M:,1],c='r')
plt.scatter(train_data[N:M+N:,0],train_data[N:M+N:,1],c='b')
plt.scatter(train_data[M:N:,0],train_data[M:N:,1],c='grey')
plt.scatter(train_data[M+N:2*N:,0],train_data[M+N:2*N:,1],c='grey')
plt.scatter(semi_data[:,0],semi_data[:,1],c='grey')
plt.axis([-20,40,-10,20])
plt.savefig("original.eps")
#%%
w, w0 = lda(train_data[0:M],train_data[N:N+M])
plt.scatter(train_data[0:M][:,0],train_data[0:M][:,1],c='r')
plt.scatter(train_data[N:N+M][:,0],train_data[N:N+M][:,1],c='b')
x1 = -10
x2 = 30
plt.plot([x1,x2],[-(x1 * w[0]/w[1])-(w0/w[1]),-(x2 * w[0]/w[1])-(w0/w[1])],'k-')
plt.axis([-20,40,-10,20])
plt.savefig("only.eps")
#%% plot for exercise 4(after)
plt.scatter(np.array(data_n)[:,0],np.array(data_n)[:,1],c='r')
plt.scatter(np.array(data_p)[:,0],np.array(data_p)[:,1],c='b')
x1 = -10
x2 = 30
plt.plot([x1,x2],[-(x1 * w[0]/w[1])-(w0/w[1]),-(x2 * w[0]/w[1])-(w0/w[1])],'k-')
plt.axis([-20,40,-10,20])
plt.savefig("gaussian.eps")
#%% draw decision boundary
cov = [[1,-0.5],[-0.5,1]]
data1 = np.random.multivariate_normal([1,0], cov, 100)
data2 = np.random.multivariate_normal([0,1], cov, 100)
w, w0 = lda(data1,data2)
plt.scatter(data1[:,0],data1[:,1],c='r')
plt.scatter(data2[:,0],data2[:,1],c='b')
x1 = -1
x2 = 3
plt.plot([x1,x2],[-(x1 * w[0]/w[1])-(w0/w[1]),-(x2 * w[0]/w[1])-(w0/w[1])],'k-')


#%% read data
spam = pd.read_csv("spambase.txt",header=None)
spam_n = np.array(spam[0:1813])
spam_p = np.array(spam[1813:])
spam_n = spam_n.T[0:-1]  #delete label
spam_p = spam_p.T[0:-1]
for i in range(len(spam_n)):
    spam_n[i] = spam_n[i] / max(spam_n[i])
for i in range(len(spam_p)):
    spam_p[i] = spam_p[i] / max(spam_p[i])
spam_n = spam_n.T
spam_p = spam_p.T
#%%
test_error_list = []
#%%
mean = []
std = []
#%%
llh_list = []

semi_size = 320
for i in range(10):
    train_n, train_p, semi_n, semi_p, test_n, test_p = divide_spam(spam_n,spam_p,semi_size)
    semi_all = np.append(semi_n,semi_p,axis=0)
    data0,data1 = semi_gaussian(train_n,train_p,semi_all)
    u1,u2,cov = lda(data0,data1)
    # for testin    
    result_ten = multi_gaussian(test_n,u1,cov) + multi_gaussian(test_n,u2,cov)
    result_tep = multi_gaussian(test_p,u1,cov) + multi_gaussian(test_p,u2,cov)
    #print(result_ten)
    llh = 0
    for i in result_ten:
        llh = llh + math.log(i+0.000000001)
    for i in result_tep:
        llh = llh + math.log(i+0.000000001)
    llh = llh / 200
    print(llh)
    llh_list.append(llh)
print("average:",sum(llh_list)/10)
    
#%%
l_mn, = plt.plot([0,1,2,3,4,5,6,7,8],mean_nearest,label = "nearest")
l_mg, = plt.plot([0,1,2,3,4,5,6],mean_gaussian[0:-1],label="gaussian")

l_sn, = plt.plot([0,1,2,3,4,5,6,7,8],std_nearest,label="std nearest")
l_sg, = plt.plot([0,1,2,3,4,5,6],std_gaussian,label="std gaussian")
plt.legend(handles=[l_mn, l_mg, l_sn, l_sg],loc="upper right")
plt.xlabel("Unlabeled data size")
plt.ylabel("error rate")
plt.savefig("curve.eps")
#%%
plt.plot([0,1,2,3,4,5,6,7,8],mean_nearest)
plt.plot([0,1,2,3,4,5,6,7],mean_gaussian)
#%%
l_n, = plt.plot([0,1,2,3,4,5,6,7,8],ml_n,label="nearest")
l_g, = plt.plot([0,1,2,3,4,5,6,],ml_g,label="gaussian")
plt.legend(handles=[l_n,l_g],loc="lower right")
plt.xlabel("Unlabeled data size")
plt.ylabel("log likelihood")
plt.savefig("llh.eps")