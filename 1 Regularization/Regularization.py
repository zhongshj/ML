#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:03:28 2017

@author: abk
"""
#%% 1.1
import matplotlib.pyplot as plt
import numpy as np
import pk.lib as pk
import random

#%%

x = np.arange(-5.0, 5.0, 0.01)
y0 = 2*x*x + 2 + 0*abs(1-x)
y2 = 2*x*x + 2 + 2*abs(1-x)
y4 = 2*x*x + 2 + 4*abs(1-x)
y6 = 2*x*x + 2 + 6*abs(1-x)
plt.figure(figsize=(6,6))
l1, = plt.plot(x,y0,'r-', lw=2, label="$\lambda =0$")
l2, = plt.plot(x,y2,'y-', lw=2,label="$\lambda =2$")
l3, = plt.plot(x,y4,'g-', lw=2,label="$\lambda =4$")
l4, = plt.plot(x,y6,'b-', lw=2,label="$\lambda =6$")
plt.legend(handles=[l1,l2,l3,l4])

plt.title('Loss function')
plt.xlabel('$m_{+}$')
plt.ylabel('$L(m_{+})$')
plt.grid(True)
plt.savefig('1.eps')

#%% 1.2
min0 = min(y0)
y0 = y0.tolist()
x0 = y0.index(min0)

min2 = min(y2)
y2 = y2.tolist()
x2 = y2.index(min2)

min4 = min(y4)
y4 = y4.tolist()
x4 = y4.index(min4)

min6 = min(y6)
y6 = y6.tolist()
x6 = y6.index(min6)

#%% 3.1
data = []

for l in open("opt.txt"):
    row = [int(x) for x in l.split()]
    if len(row) > 0:
        data.append(row)
        
data0 = np.array(data[0:554])
data1 = np.array(data[554:1125])

#%%

ma = np.zeros(64)
mb = np.zeros(64)
for i in list(range(64)):
    ma[i]=random.uniform(0,255)
    ma[i]=random.uniform(0,255)
   
step = 0.00001
lbd = 300000

while True:
    grad = pk.grd_vec(data0,data1,ma,mb,lbd)
    previous_loss = pk.nmc_loss(data0,data1,ma,mb,lbd)
    ma = ma - step*grad[0]
    mb = mb - step*grad[1]
    print(ma[36],mb[36],step)
    new_loss = pk.nmc_loss(data0,data1,ma,mb,lbd)
    if(abs(previous_loss-new_loss)<10000):
        break
    
#loss = pk.nmc_loss(data0,data1,ma,mb,lbd)
#print(loss)

#%%
ma = np.reshape(ma,(8,8)) 
mb = np.reshape(mb,(8,8))
plt.imshow(ma,cmap=plt.cm.gray,interpolation="nearest")    
plt.imshow(mb,cmap=plt.cm.gray,interpolation="nearest")