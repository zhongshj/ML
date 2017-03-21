#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:11:14 2017

@author: abk
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
PI = 3.14159265359

#%%

def multi_gaussian(x,u,cov):
    
    dim = np.size(x)
    x = np.mat(x)
    u = np.mat(u)
    cov = np.mat(cov)
    a = ((2*PI)**dim)*np.linalg.det(cov)
    b = (x-u)*np.linalg.inv(cov)*(x-u).T
    gaussian = (1/math.sqrt(a))*np.exp(-0.5*b[0,0])
    return gaussian
    