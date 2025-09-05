# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:48:32 2024

@author: nouri
"""
# -*- coding: utf-8 -*-
import numpy as np
from math import pi
import matplotlib.pyplot as plt
plt.ion()

name = input("Nom fichier :")
rp = float(input('Résistance pristine: '))
e = float(input('Epaisseur en nm: '))
data = np.loadtxt("20240621/" + name + ".NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1];
c4 = data[:, 3];
c9 = data[:, 8];
c10 = data[:, 9]
c12 = data[:, 11];
c11 = data[:, 10]
t = len(c4)
A0 = []
X0 = np.zeros((int(t), 4))
X1 = np.zeros((int(t), 4))
k = 0
r = 0
plt.figure(dpi=1200)
for i in range(t):
    # RESET conditions
    if c9[i] == 6E-9:  # and c10[i]==30E-9 and c11[i]==6E-9 :
        X0[i][1] = c4[i]
        X0[i][2] = c12[i]
        X0[i][0] = i
        X0[i][3] = i
    # Set conditions
    else:
        if c9[i] != 0:
            X1[i][1] = c4[i]
            X1[i][2] = c12[i]
            X1[i][0] = i
            X1[i][3] = i

Rp = []
Rth = []
r_th = (35 * e * 1E-7) / (pi * 0.6e-4 * 0.6e-4)
for t in range(len(c12)):
    Rp.append(float(rp * 1e+3))  # Mettre la valeur de
    # R th dans la list
    Rth.append(float(r_th))

# plt.grid()
# Plot pour la résistance
color = 'tab:red'
plt.ylabel('Resistance in ohm', color=color)
plt.label('# Pulses', color=color)
plt.plot(X0[:, 0], X0[:, 1], 'r.', X1[:, 0], X1[:, 1], 'k.', X0[:, 0], Rp[:], 'c-', X0[:, 0], Rth[:], 'b-')
plt.plot(X1[:, 0], Rp[:], 'c-', label='Rp=' + str(int(rp)) + 'k')
plt.plot(X1[:, 0], Rth[:], 'b-', label='Rth=' + str(int(r_th * 1e-3)) + ' k')
plt.tick_params(axis='y', labelcolor=color)
plt.yscale('log')
plt.ylim([500, 100000])
_a = 0;
_b = 600  # Limits of x axis
# ax1.set_xlim([_a,_b])
#plt.xticks([])
plt.grid(color='k', axis='y')
plt.title(name)
plt.legend(loc='upper right', fontsize="6")