# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:09:09 2024

@author: nouri
"""

from matplotlib.pyplot import cm
import numpy as np
import random
from statistics import mean
from statistics import stdev
import matplotlib.pyplot as plt
name=input("Nom fichier :")


data=np.loadtxt("PVDE47/" + name + ".NL",skiprows=3)
a0=data[:,1]; a1=data[:,3]; a2=data[:,8]; a3=data[:,9]; a4=data[:,8];a5=data[:,11]
t=len(a1)
A0=[]
X0=np.zeros((int(t),4))
X1=np.zeros((int(t),4))
k=0
r=0
fig, ax1 = plt.subplots()
for i in range(t):
    #RESET conditions
    if a4[i]==6E-9 and a3[i]==20E-9  : #or a3[i]==12E-9 or a3[i]==14E-9 or a3[i]==16E-9 or a3[i]==18E-9 or a3[i]==20E-9 or a3[i]==22E-9 or a3[i]==24E-9 or a3[i]==25E-9:
        X0[i][1]=a1[i]
        X0[i][2]=a5[i] #Car offset de 0.5V
        X0[i][0]=i
        X0[i][3]=i
    #Set conditions
    else:
        if a4[i]!=0:
            X1[i][1]=a1[i]
            X1[i][2]=a5[i] #Car offset de 0.5V
            X1[i][0]=i
            X1[i][3]=i
    
#plt.grid()
color = 'tab:red'
ax1.set_xlabel('Nombre de pulse')
ax1.set_ylabel('Résistance en ohm', color=color)
ax1.plot(X0[:,0],X0[:,1],'r.',X1[:,0],X1[:,1],'k.')   
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')
ax1.set_ylim([100,100000])
ax1.grid(color='k',axis='y')

ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.set_ylabel('Tension de pulse en V', color=color) 
ax2.plot(X0[:,3],X0[:,2],'b.',X1[:,3],X1[:,2],'g.')
ax2.set_yscale('linear')
ax2.grid(color='b',axis='y')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
fig.set_dpi(1200)
    
    

#plt.yticks([1,1000,10000,100000])
#plt.xlim([68,117])
plt.title(name)