# -*- coding: utf-8 -*-
"""
Created on Tues Nov  26 10:14:32 2024

@author: nouri
"""

import numpy as np
import matplotlib.pyplot as plt

dos = str(20241127)
dot_plot = 12  # taille point plot
labsize = 20 # Taille graduation axe
font = 26 # Taille titres des axes
fontLEG = 16 # taille legende
l=6 #largeur fig
L=7 #longueur fig
name=str(input('Nom du fichier: '))
v_prog=input('Voltage programmé: ')
data=np.loadtxt(""+dos+"/"+name+".osc",skiprows=1)
l=(data[:,0])*1e+9
k=data[:,1]
n=data[:,2]
K=max(n)
N=max(k)
plt.figure(figsize=(10, 9))
color = 'tab:red'
plt.plot(l,k,'g', label='Rload voltage: ' +str(N)+'V' ,  markersize=dot_plot)
plt.plot(l,n,'b', label='Applied V= '+str(K)+'V & Prog V= ' + str(v_prog) + 'V', markersize=dot_plot)
plt.ylabel('Voltage in V')
plt.xlabel('Time in ns')
plt.title(name, fontsize=font)
plt.legend(labelcolor='linecolor',fontsize=font)
plt.grid()
plt.xlim(-10e-9,100e-9)
plt.xticks(np.arange(-10, 100, step=10))
plt.ylim(-1, 7)
plt.tick_params(axis='x', labelcolor=color, labelsize=labsize)
plt.tick_params(axis='y', labelcolor=color, labelsize=labsize)
plt.savefig("" + dos + "/" + name + ".svg")
plt.show()