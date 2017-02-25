#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:40:31 2017

@author: abk
"""
import numpy as np

def l1_norm(x):
    sum = 0
    for i in x:
        sum = sum + abs(i)
    return sum
    
def l2_norm(x):
    sum = 0
    for i in x:
        sum = sum + i*i
    return sum
    
def nmc_loss(xa,xb,ma,mb,l):
    sum = 0
    for i in xa:
        sum = sum + l2_norm(i-ma)
    for i in xb:
        sum = sum + l2_norm(i-mb)
    sum = sum + l*l1_norm(ma-mb)
    return sum

def grd_vec(xa,xb,ma,mb,lbd):
    sum1 = 0
    sum2 = 0
    for i in xa:
        sum1 = sum1 + ma - i
    sum1 = sum1 * 2
    sum1 = sum1 + lbd * np.sign(ma-mb)
    for i in xb:
        sum2 = sum2 + mb - i
    sum2 = sum2 * 2
    sum2 = sum2 - lbd * np.sign(ma-mb)
    return [sum1,sum2]
    
    
    
    
    