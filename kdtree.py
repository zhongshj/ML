#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:23:19 2017

@author: abk
"""
import numpy as np
import math
import matplotlib.pyplot as plt

#%%

class Node(object):
    
    def __init__(self,x,dim):
        #for each node, store sample and dimension
        self.x = x
        self.dim = dim
        self.parent = None
        self.left = None
        self.right = None
        
    def print_df(self):
        #depth first search
        print(self.x,self.dim)
        if self.left != None:
            print("left:")
            self.left.print_df()
        if self.right != None:
            print("right:")
            self.right.print_df()
 #%%
def get_cut(data):
    
    dim = np.size(data,1)
    num = np.size(data,0)
    #print(num)
    max_std = 0
    cut_dim = 0
    for i in range(dim):
        std = np.std(data[:,i])
        if std > max_std:
            max_std = std
            cut_dim = i
            
    sort_index = np.argsort(data[:,cut_dim])
    cut_index = sort_index[math.ceil((np.size(sort_index)-1)/2)]
    cut_x = data[cut_index]
    #data = np.append(data[:cut_index],data[cut_index:],axis=0)   
                                
    return cut_x, cut_dim, cut_index
         
def split_data(data):
    
    cut_x, cut_dim, cut_index = get_cut(data)
    
    data = np.append(data[:cut_index],data[cut_index+1:],axis=0)
    
    data0 = []
    data1 = []
    
    dim = np.size(data,1)
    num = np.size(data,0)
    
    for i in range(num):
        if data[i,cut_dim] <= cut_x[cut_dim]:
            data0.append(data[i])
        else:
            data1.append(data[i])
    
    return cut_x, cut_dim, np.array(data0), np.array(data1)
                 
def kdtree(data,parent=None):
    
    if np.size(data,0) == 1:
        print("end:",data[0])
        return Node(data[0],None)
    elif np.size(data,0) == 0:
        return None
    else:
        cut_x, cut_dim, data0, data1 = split_data(data)
        head = Node(cut_x,cut_dim)
        head.parent = parent
        head.left = kdtree(data0,head)
        head.right = kdtree(data1,head)
        
        return head
            
def dis(a, b):
    return sum((a - b)**2)

def update_li(li, p, x, k=3):
    # p: point you input
    # x: candidate point
    if len(li) < k:
        li.append(x)
    else:
        for i in range(len(li)):
            if dis(li[i], p) > dis(x, p):
                temp = li.pop(i)
                li.insert(i, x)   
                x = temp
    return li

#%%        
def kd_search(head, p, k, li=[]):
    
    if head.left == None and head.right == None:
        li = update_li(li, p, head.x, k)
        
    
    if head.x[head.dim] <= p[head.dim]:
        if head.left == None:
            
            kd_search(head.left, p, k, li)
    else:
        kd_search(head.right, p, k, li)
        
    
#%%

head = kdtree(data)