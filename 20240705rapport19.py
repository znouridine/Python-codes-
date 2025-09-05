# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:58:51 2024

@author: nouri
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
import csv
from matplotlib.pyplot import cm

name = input("Nom fichier :")
rp = float(input('Résistance pristine: '))
e = float(input('Epaisseur en nm: '))

plt.figure()
data = np.loadtxt("20240621/" + name + ".NL", skiprows=3,
                  usecols=range(0, 14))
c2 = data[:, 1];c4 = data[:, 3];c9 = data[:, 8];c10 = data[:, 9];c12 = data[:, 11];c11 = data[:, 10]
t = (len(c4) - 3)
A0 = []
X0 = np.zeros((int(t), 4))
X0_ = np.zeros((int(t), 4))
X1 = np.zeros((int(t), 4))
X1_ = np.zeros((int(t), 4))
k = 0
r = 0
step = 200
for i in range(t):
    # RESET conditions
    if c9[i] == 6E-9:
        X0[i][1] = c4[i]
        X0[i][2] = c12[i]
        X0[i][0] = i
        X0[i][3] = i
        """
        if abs(c4[i]-c4[i+1])>step and 
        abs(c4[i+1]-c4[i+2])>step and 
        abs(c4[i+2]-c4[i+3])>step:
                X0_[i][1] = X0[i][1]
                X0_[i][2] = X0[i][2]
                X0_[i][0] = i
                X0_[i][3] = i
                """
        # Set conditions
    else:
        if c9[i] != 0:
            X1[i][1] = c4[i]
            X1[i][2] = c12[i]
            X1[i][0] = i
            X1[i][3] = i
            """
            if abs(c4[i]-c4[i+1])>step and
            abs(c4[i+1]-c4[i+2])>step and 
            abs(c4[i+2]-c4[i+3])>step:
                    X1_[i][1] = X1[i][1]
                    X1_[i][2] = X1[i][2]
                    X1_[i][0] = i
                    X1_[i][3] = i
                    """

Rp = []
Rth = []
r_th = (35 * e * 1E-7) / (pi * 0.6e-4 * 0.6e-4)
for p in range(t):
    Rp.append(float(rp * 1e+3))  # Mettre la valeur de
    # R th dans la list
    Rth.append(float(r_th))

# plt.grid()
# Plot pour la résistance
color = 'tab:red'
plt.ylabel('Resistance in ohm', color=color)
plt.xlabel('# Pulses', color=color)
plt.plot(X0[:, 0], X0[:, 1], 'r.',
         X1[:, 0], X1[:, 1], 'k.',
         X0[:, 0], Rp[:], 'c-',
         X0[:, 0], Rth[:], 'b-')
plt.plot(X1[:, 0], Rp[:], 'c-',
         label='Rp=' + str(int(rp)) + 'k')
plt.plot(X1[:, 0], Rth[:], 'b-',
         label='Rth=' + str(int(r_th * 1e-3)) + ' k')
plt.tick_params(axis='y', labelcolor=color)
plt.yscale('log')
plt.ylim([500, 100000])
_a = 425;
_b = 1500  # Limits of x axis
plt.xlim([_a, _b])
plt.xticks([])
plt.grid(color='k', axis='y')
plt.title(name)
plt.legend(loc='upper right', fontsize="6")

# Range allocation
nb_bande = 8
bande = np.zeros((nb_bande + 1, 6))
gap_log = 0.06
min_bande1 = np.log10(800)
max_bande1 = np.log10(1050)
larg_range = max_bande1 - min_bande1
bande[0][0] = min_bande1
bande[0][1] = max_bande1
_x = max(X0[:, 0]);
_n = min(X0[:, 0])
color = iter(cm.rainbow(np.linspace(0, 1, nb_bande)))
for i in range(nb_bande):
    bande[i + 1][0] = bande[i][1] + gap_log
    bande[i + 1][1] = bande[i + 1][0] + larg_range
    bande[i][2] = (bande[i][0] + bande[i][1]) / 2
    bande[i][3] = 10 ** bande[i][0]
    bande[i][4] = 10 ** bande[i][1]
    bande[i][5] = (bande[i][3] + bande[i][4]) / 2
    c = next(color)
    plt.axhspan(10 ** bande[i][0], 10 ** bande[i][1],
                xmin=_n, xmax=_x,
                color=c, alpha=0.5)
DF = pd.DataFrame(bande)
DF.to_csv("20240621/" + name + ".csv")
plt.show()
plt.savefig("20240621/pics/" + name +".png", dpi=1200)
plt.ion()