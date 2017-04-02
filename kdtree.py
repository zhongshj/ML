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
        self.visited = False
        
    def print_df(self):
        #depth first search
        print(self.x,self.dim)
        if self.left != None:
            print("left:")
            self.left.print_df()
        if self.right != None:
            print("right:")
            self.right.print_df()

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
        head = Node(data[0],None)
        head.parent = parent
        return head
    elif np.size(data,0) == 0:
        return None
    else:
        cut_x, cut_dim, data0, data1 = split_data(data)
        head = Node(cut_x,cut_dim)
        head.parent = parent
        head.left = kdtree(data0,head)
        head.right = kdtree(data1,head)
        
        return head
        
#%% 

            
def dis(a, b):
    if np.size(a) == 1:
        return (a-b)**2
    else:
        return sum((a - b)**2)
    
def update_li(li, p, x, k):
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

def dis_dim(p,li,node):
    p = p[node.dim]
    for i in range(len(li)):
        if dis(p, node.x) < dis(p, li[i][node.dim]):
            return True
    return False
        
def kd_search_in(node, p):
    
    # if leaf node, return
    if node.left == None and node.right == None:
        print("found leaf")
        node.visited = True
        return node
    
    # if not leaf, go deeper
    if p[node.dim] < node.x[node.dim]:
        print("go left")
        node = kd_search_in(node.left, p)
    else:
        print("go right")
        node = kd_search_in(node.right, p)
        
    return node
    

def kd_search(head, p, li=[]):
    k = 3
    #first, get leaf node
    leaf = kd_search_in(head,p)
    li = update_li(li, p, leaf.x, k)
    
    #then, get back and update list
    while leaf.parent != None:
        upper = leaf.parent
        print("go up")
        if upper.visited == False:
            upper.visited = True
            li = update_li(li, p, upper.x, k)
            if dis_dim(p,li,upper):
                if upper.left == leaf:
                    li = kd_search(upper.right, p, li)
                else:
                    li = kd_search(upper.left, p, li)                
    return li
    
#%%
data = np.array([[1,2],[1,4],[5,3],[8,7],[3,6],[6,6],[3,1]])

head = kdtree(data)

#%%

li = kd_search(head, np.array([1.1,2.1]))