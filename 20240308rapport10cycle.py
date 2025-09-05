# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 18:16:06 2024

@author: nouri
"""

from matplotlib.pyplot import cm
import numpy as np
from math import pi
import random
from statistics import mean
from statistics import stdev
import matplotlib.pyplot as plt
name=input("Nom fichier :")
rp = float(input('Résistance pristine: '))



data=np.loadtxt("20240308/" + name + ".NL",skiprows=3)
a0=data[:,1]; a1=data[:,3]; a2=data[:,8]; a3=data[:,9]; a4=data[:,8];a5=data[:,11];a6=data[:,10]
t=len(a1)
A0=[]
X0=np.zeros((int(t),4))
X1=np.zeros((int(t),4))
k=0
r=0
fig, (ax1,ax2) = plt.subplots(2)
for i in range(t):
    #RESET conditions
    if a4[i]==6E-9 or a3[i]==30E-9: #or a3[i]==12E-9 or a3[i]==14E-9 or a3[i]==16E-9 or a3[i]==18E-9 or a3[i]==20E-9 or a3[i]==22E-9 or a3[i]==24E-9 or a3[i]==25E-9:
        X0[i][1]=a1[i]
        X0[i][2]=a5[i] 
        X0[i][0]=i
        X0[i][3]=i
    #Set conditions
    else:
        if a4[i]!=0:
            X1[i][1]=a1[i]
            X1[i][2]=a5[i]
            X1[i][0]=i
            X1[i][3]=i
            
Rp=[]            
for t in range(len(a5)):
    Rp.append(float(rp*1e+3)) # Mettre la valeur de R th dans la list        
         

#plt.grid()
#Plot pour la résistance
color = 'tab:red'
ax1.set_ylabel('Résistance en ohm', color=color)
ax1.plot(X0[:,0],X0[:,1],'r.',X1[:,0],X1[:,1],'k.',X0[:,0],Rp[:],'c-',X1[:,0],Rp[:],'c-')   
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')
ax1.set_ylim([0,40000])
ax1.set_xticks([])
ax1.grid(color='k',axis='y')
ax1.set_title(name)
#Plot pour la tension
#ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.set_ylabel('Tension de pulse en V', color=color) 
ax2.plot(X0[:,3],X0[:,2],'r.',X1[:,3],X1[:,2],'k.')
ax2.set_yscale('linear')
ax2.set_xlabel('Nombre de pulse')
ax2.grid(color='b',linestyle='--', linewidth=0.7,axis='y')
ax2.set_yticks([1,2,3,4,5,6,7,8])
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
fig.set_dpi(1200)
    
    

#plt.yticks([1,1000,10000,100000])
#plt.xlim([68,117])
