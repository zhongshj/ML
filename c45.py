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
        for i in node.children:
            q.put(i)
        print(node.attr)
        
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
    
    #create cut array(only choose cut that differs labels)
    cut = []
    for i in range(np.size(data0)-1):
        if label[i] != label[i+1]:
            cut.append((data0[i] + data0[i+1]) / 2)
    cut = np.array(cut)
    
    #calculate H(D)
    label_freq = np.zeros(max(label)+1)
    for i in label:
        label_freq[i] = label_freq[i] + 1
    h_d = entropy(label_freq / sum(label_freq))
    print("entropy:",h_d)
    
    #find a best cut
    for i in cut:
        data_q = np.zeros(np.size(data0))
        max_gain = 0
        max_cut = 0
        for j in range(np.size(data0)):
            if data0[j] > i:
                data_q[j] = 1
        gain = _cal_gain_binary(data_q,label,h_d)
        if gain > max_gain:
            max_gain = gain
            max_cut = i
            
    return max_gain, max_cut
    
def find_feature(data,label):
    
    dim = np.size(data,1)
    max_dim = 0
    max_cut = 0
    max_gain = 0
    for i in range(dim):
        gain, cut = _find_cut(data[:,i],label)
        if gain > max_gain:
            max_dim = i
            max_cut = cut
            
    return max_dim, max_cut

def build_tree(data,label):
    dim = np.size(data,1)
    num = np.size(data,0)
    
    if num == 1:
        node = Node(None,None)
        node.label = label[0,0]
        return node
        
    else:
    
        #get feature with highest entropy
        dim,cut = find_feature(data,label)
        left, right = split_data(data,dim,cut)
        data_left = data[left]
        label_left = label[left]
        data_right = data[right]
        label_right = label[right]
        
        node = Node(dim,cut)
        node.left = build_tree(data_left,label_left)
        node.right = build_tree(data_right,label_right)
    
    return node
        
        