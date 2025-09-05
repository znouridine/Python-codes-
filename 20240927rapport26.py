# -*- coding: utf-8 -*-
"""
Created on FRI Sept 27 15:37:32 2024

@author: nouri
"""
globals().clear()
import numpy as np
from math import pi
import matplotlib.pyplot as plt

plt.ioff()
name = input("Nom fichier :")
rp = float(input('Résistance pristine: '))
e = float(input('Epaisseur en nm: '))
data = np.loadtxt("20240927/" + name + ".NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1];
c4 = data[:, 3];
c9 = data[:, 8];
c10 = data[:, 9]
c12 = data[:, 11];
c11 = data[:, 10]
t = len(c4)
A0 = []
X0 = np.zeros((int(t), 5))
X1 = np.zeros((int(t), 5))
k = 0
r = 0
dot_plot = 12  # taille point plot
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 9), height_ratios=[3, 1, 1])
for i in range(t):
    # RESET conditions
    if c9[i] == 6E-9:  # and c10[i]==30E-9 and c11[i]==6E-9 :
        X0[i][1] = c4[i]
        X0[i][2] = c12[i]
        X0[i][4] = int((c9[i] + c10[i] + c11[i]) * 10 ** 9)
        X0[i][0] = i
        X0[i][3] = i
    # Set conditions
    else:
        if c9[i] != 0:
            X1[i][1] = c4[i]
            X1[i][2] = c12[i]
            X1[i][4] = int((c9[i] + c10[i] + c11[i]) * 10 ** 9)
            X1[i][0] = i
            X1[i][3] = i

Rp = []
Rth = []
r_th = (35 * e * 1E-7) / (pi * 0.6e-4 * 0.6e-4)  # La résistivité du 30% Cr est 35 Ohm.cm
for t in range(len(c12)):
    Rp.append(float(rp * 1e+3))  # Mettre la valeur de
    # R th dans la list
    Rth.append(float(r_th))

# plt.grid()
# Plot pour la résistance
color = 'tab:red'
ax1.set_ylabel('Resistance(ohm)', fontsize="18", color=color)
ax1.plot(X0[:, 0], X0[:, 1], 'r.', X1[:, 0], X1[:, 1], 'k.', X0[:, 0], Rp[:], 'c-', X0[:, 0], Rth[:], 'b-',
         markersize=dot_plot)
ax1.plot(X1[:, 0], Rp[:], 'c-', label='Rp=' + str(int(rp)) + 'k', markersize=dot_plot)  # Plot Rp
ax1.plot(X1[:, 0], Rth[:], 'b-', label='Rth=' + str(int(r_th * 1e-3)) + ' k', markersize=dot_plot)  # Plot R th
ax1.tick_params(axis='y', labelcolor=color, labelsize=15)
ax1.set_yscale('log')
ax1.set_ylim([100, 100000])
_a = 0;
_b = 1650  # Limits of x axis
# ax1.set_xlim([_a,_b])
ax1.set_xticks([])
ax1.grid(color='k', axis='y')
ax1.set_title(name, fontsize="14")
ax1.legend(loc='upper right', fontsize="12")
# Plot pour la tension
color = 'tab:blue'
ax2.set_ylabel("Voltage(V)", fontsize="18", color=color)
ax2.plot(X0[:, 3], X0[:, 2], 'r.', X1[:, 3], X1[:, 2], 'k.', markersize=dot_plot)  # Plot tension
ax2.set_xticks([])
ax2.set_yticks([2, 4, 6, 8, 10])
ax2.set_ylim([1, 10])
# ax2.set_xlim([_a,_b])
ax2.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
fig.tight_layout()
# Plot du temps
ax3.set_ylabel('Time(ns)', fontsize="18", color=color)
ax3.plot(X0[:, 3], X0[:, 4], 'r.', X1[:, 3], X1[:, 4], 'k.', markersize=dot_plot)  # Plot temps de pulse
ax3.set_yscale('linear')
ax3.set_yticks([50, 100, 150, 200, 250])
ax3.set_ylim([10, 250])
# ax3.set_xlim([_a,_b])
ax3.set_xlabel('Number of pulse', fontsize="20")
ax3.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax3.tick_params(axis='y', labelcolor=color, labelsize=15)
ax3.tick_params(axis='x', labelsize=15)
fig.tight_layout()
fig.savefig("20240927/" + name + ".svg")
plt.show()
