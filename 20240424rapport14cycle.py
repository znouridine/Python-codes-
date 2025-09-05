# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:01:39 2024

@author: nouri
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
name=input("Nom fichier :")
rp = float(input('Résistance pristine: '))
e = float(input('Epaisseur en nm: '))
data=np.loadtxt("20240424/" + name + ".NL" ,skiprows=3)
c2=data[:,1]; c4=data[:,3]; c9=data[:,8]; c10=data[:,9]
c9=data[:,8];c12=data[:,11];c11=data[:,10]
t=len(c4)
A0=[]
X0=np.zeros((int(t),4))
X1=np.zeros((int(t),4))
k=0
r=0
fig, (ax1,ax2) = plt.subplots(2)
for i in range(t):
    #RESET conditions
    if c9[i]==6E-9 :# and c10[i]==30E-9 and c11[i]==6E-9 : 
        X0[i][1]=c4[i]
        X0[i][2]=c12[i] 
        X0[i][0]=i
        X0[i][3]=i
    #Set conditions
    else:
        if c9[i]!=0:
            X1[i][1]=c4[i]
            X1[i][2]=c12[i]
            X1[i][0]=i
            X1[i][3]=i
            
Rp=[]
Rth=[] 
r_th=(35 * e*1E-7) / (pi * 0.6e-4*0.6e-4)           
for t in range(len(c12)):
    Rp.append(float(rp*1e+3)) # Mettre la valeur de
                                #R th dans la list
    Rth.append(float(r_th))
         
#plt.grid()
#Plot pour la résistance
color = 'tab:red'
ax1.set_ylabel('Résistance en ohm', color=color)
ax1.plot(X0[:,0],X0[:,1],'r.',X1[:,0],X1[:,1],'k.',X0[:,0],Rp[:],'c-',X0[:,0],Rth[:],'b-')
ax1.plot(X1[:,0],Rp[:],'c-',label='Rp='+str(int(rp))+'k')
ax1.plot(X1[:,0],Rth[:],'b-',label='Rth='+str(int(r_th*1e-3))+' k')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')
ax1.set_ylim([0,100000])
ax1.set_xlim([570,1100])
ax1.set_xticks([])
ax1.grid(color='k',axis='y')
ax1.set_title(name)
ax1.legend(loc='lower left',fontsize="5")
#Plot pour la tension
#ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.set_ylabel('Tension de pulse en V', color=color) 
ax2.plot(X0[:,3],X0[:,2],'r.',X1[:,3],X1[:,2],'k.')
ax2.set_yscale('linear')
ax2.set_xlabel('Nombre de pulse')
ax2.grid(color='b',linestyle='--', linewidth=0.7,axis='y')
ax2.set_yticks([1,2,3,4,5,6,7,8])
ax2.set_xlim([570,1100])
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
fig.set_dpi(1200)

#plt.yticks([1,1000,10000,100000])
#plt.xlim([68,117])
