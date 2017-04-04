#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:15:49 2017

@author: abk
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import queue

#%%




class Node(object):
    
    def __init__(self,dim,cut):
        self.dim = dim
        self.cut = cut
        self.left = None
        self.right = None
        self.label = None
        
# for leaf nodes, left and right will be None(but have label)
# for non-leaf nodes, label will be None(but no left & right)
        
def rd(mean,var,N):
    return np.random.normal(mean, var, N)

def entropy(prob):
    h = 0
    for i in prob:
        if i == 0:
            continue
        h = h - i * math.log(i,2)
    return h
    
    
def bfs(head):
    q = queue.Queue()
    q.put(head)
    while q.empty() == False:
        node = q.get()
        
        if node.left != None:
            q.put(node.left)
        if node.right != None:
            q.put(node.right)
            
        print(node.dim,node.cut)
        
def split_data(data, dim, cut):
    li_left = []
    li_right = []
    for i in range(np.size(data,0)):
        if data[i,dim] <= cut:
            li_left.append(i)
        else:
            li_right.append(i)
    
    return np.array(li_left), np.array(li_right)
            

#def _cal_gain_rate(data0,label,h_d):
#    #define feature 0~n, label 0~n
#    h_d_ai = np.zeros(max(data0)+1)
#    count_a = np.zeros(max(data0)+1)
#    
#    # i denotes A(i) for this feature
#    for i in range(max(data0)+1):
#        li = np.zeros(max(label)+1)
#        for j in range(np.size(data0)):
#            if data0[j] == i:
#                li[label[j]] = li[label[j]] + 1
#        count_a[i] = sum(li)
#        li = li / count_a[i]
#        h_d_ai[i] = entropy(li)
#    
#    count_a = count_a / sum(count_a)
#    
#    return (h_d - sum(h_d_ai * count_a)) / entropy(count_a)


#def max_entropy(data,label):
#    
#    dim = np.size(data,1)
#    
#    #calculate H(D)
#    label_freq = np.zeros(max(label)+1)
#    for i in label:
#        label_freq[i] = label_freq[i] + 1
#    h_d = entropy(label_freq / sum(label_freq))
#    print("entropy:",h_d)
#    #calculate gR(D,A)
#    grda = np.zeros(dim)
#    for i in range(dim):
#        grda[i] = _cal_gain_rate(data[:,i],label,h_d)
#    
#    #return max gain rate and the corresponding feature
#    return grda
    
def _cal_gain_binary(data0,label,h_d):
    #define feature 0~n, label 0~n
    h_d_ai = np.zeros(2)
    count_a = np.zeros(2)
    
    # i denotes A(i) for this feature
    for i in range(2):
        li = np.zeros(max(label)+1)
        for j in range(np.size(data0)):
            if data0[j] == i:
                li[label[j]] = li[label[j]] + 1
        count_a[i] = sum(li)
        li = li / count_a[i]
        h_d_ai[i] = entropy(li)
    
    count_a = count_a / sum(count_a)
    
    return h_d - sum(h_d_ai * count_a)

def _find_cut(data0,label):

    #sort data and label
    sort_idx = np.argsort(data0)
    data0 = data0[sort_idx]
    label = label[sort_idx]
    #print(label)
    #create cut array(only choose cut that differs labels)
    cut = []
    for i in range(np.size(data0)-1):
        if label[i] != label[i+1]:
            cut.append((data0[i] + data0[i+1]) / 2)
    cut = np.array(cut)
    #print(cut)
    
    #calculate H(D)
    label_freq = np.zeros(max(label)+1)
    for i in label:
        label_freq[i] = label_freq[i] + 1
    h_d = entropy(label_freq / sum(label_freq))
    #print("entropy:",h_d)
    
    #find a best cut
    for i in cut:
        data_q = np.zeros(np.size(data0))
        max_gain = 0
        max_cut = 0
        for j in range(np.size(data0)):
            if data0[j] > i:
                data_q[j] = 1
        gain = _cal_gain_binary(data_q,label,h_d)
        print("gain",gain)
        if gain > max_gain:
            max_gain = gain
            max_cut = i
            
    return max_gain, max_cut
    
def _find_feature(data,label,prune):
    
    dim = np.size(data,1)
    max_dim = 0
    max_cut = 0
    max_gain = 0
    for i in range(dim):
        print("search dim:",i)
        gain, cut = _find_cut(data[:,i],label)
        if gain > max_gain:
            max_gain = gain
            max_dim = i
            max_cut = cut
            
    print("maxgain:",max_gain,"dim,cut",)
 
    #if the gain is too small, set dim with -1 to tell outside to stop spliting
    if max_gain < prune:
        max_dim = -1
    
    return max_dim, max_cut

def _unsep(label):
    #judge if only 1 class in label
    for i in range(np.size(label)-1):
        if label[i] != label[i+1]:
            return False
    return True
    
def _get_majority(label):
    print("label:",label)
    li = []
    for i in range(int(max(label)+1)):
        li.append(len([x for x in label if x == i]))
    return li.index(max(li))
    
    
def build_tree(data,label,prune):
    dim = np.size(data,1)
    num = np.size(data,0)
    
    if _unsep(label):
        # if there is only 1 class, no need for further spliting
        print("leaf node")
        node = Node(None,None)
        node.label = label[0]
        return node
        
    else:
    
        #get feature with highest entropy
        dim,cut = _find_feature(data,label,prune)
        
        if dim == -1:
            # pruning
            print("low gain")
            node = Node(None,None)
            
            #select majority as label if multiple labels
            if _unsep(label):
                node.label = label[0]
            else:
                node.label = _get_majority(label)
            return node
            
        else:    
            left, right = split_data(data,dim,cut)
            data_left = data[left]
            label_left = label[left]
            data_right = data[right]
            label_right = label[right]
            
            print("dim:",dim,"cut:",cut)
            print("left:",data_left)
            print("right:",data_right)
            
            node = Node(dim,cut)
            print("let's go left")
            node.left = build_tree(data_left,label_left,prune)
            print("let's go right")
            node.right = build_tree(data_right,label_right,prune)
    
    return node
    
def test_c45(node,test_data):
    
    if node.label != None:
        return node.label
    else:
        if test_data[node.dim] <= node.cut:
            return test_c45(node.left,test_data)
        else:
            return test_c45(node.right,test_data)
        
#%% 
N = 50
c21 = rd(1, 1, N)
c22 = rd(3, 1, N)
c1 = rd(2, 1, 2*N)
c31 = rd(1.5, 1, N)
c32 = rd(2.5, 1, N)
data = np.array([c1,np.append(c21,c22),np.append(c31,c32)])
data = data.T
label = np.append(np.zeros(N),np.ones(N))

#%%
head = build_tree(data,label,0.01)
#%%
bfs(head)
#%%
for i in data:
    a = test_c45(head,i)
    print(a)