# -*- coding: utf-8 -*-
"""
Created on MON Oct 30 16:37:32 2024
@author: nouri
"""
globals().clear()
import numpy as np
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
import csv
from matplotlib.pyplot import cm


####Chargement fichier de données
rp=47
Rp = []
Rth = []
e = 55
data = np.loadtxt("20241016/Ligne05_PVDE55_Rp27k.NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1]
c4 = data[:, 3]
c9 = data[:, 8]
c10 = data[:, 9]
c12 = data[:, 11]
c11 = data[:, 10]
t=int(len(c4))
###Tracé des résisitance
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
for p in range(t):
    Rp.append(float(rp * 1e+3))  # Mettre la valeur de
    # R th dans la list
    Rth.append(float(r_th))

# plt.grid()
# Plot pour la résistance
plt.figure(1)
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
#plt.xlim([_a, _b])
#plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Ligne06_PVDE55-Ligne06_PVDE55-B_Rp47k_20240903-4p3V')
plt.legend(loc='upper right', fontsize="6")
####Allocation bande de résistance
#Calcul largeur et espacement des bande
nb_bande = 4
bande = np.zeros((nb_bande + 1, 6))
gap_log = 0.35
min_bande1 = np.log10(900)
max_bande1 = np.log10(1100)
larg_range = max_bande1 - min_bande1
bande[0][0] = min_bande1
bande[0][1] = max_bande1
plt.figure(1)
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
                xmin=0, xmax=t,
                color=c, alpha=0.5)
DF = pd.DataFrame(bande)
DF.to_csv("20241016/bande.csv")
plt.savefig("20241016/bande.svg")
##Lecture largeur et espacement des bande


#traitment
b1_R=[]
b1_Pulse=[]
b2_R=[]
b2_Pulse=[]
b3_R=[]
b3_Pulse=[]
b4_R=[]
b4_Pulse=[]

for i in range (t):
    if bande[0][3]<=c4[i]<=bande[0][4] :
        b1_R.append(c4[i])
        b1_Pulse.append(i + 1)
        continue

    elif bande[1][3] <= c4[i] <= bande[1][4]:
        b2_R.append(c4[i])
        b2_Pulse.append(i + 1)

    elif bande[2][3] <= c4[i] <= bande[2][4]:
        b3_R.append(c4[i])
        b3_Pulse.append(i + 1)

    elif bande[3][3] <= c4[i] <= bande[3][4]:
        b4_R.append(c4[i])
        b4_Pulse.append(i + 1)
plt.figure(2)
color = 'tab:red'
plt.ylabel('Resistance in ohm', color=color)
plt.xlabel('# Pulses', color=color)
plt.plot(b1_Pulse[:],b1_R[:],'r.')
plt.plot(b2_Pulse[:],b2_R[:],'b.')
plt.plot(b3_Pulse[:],b3_R[:],'g.')
plt.plot(b4_Pulse[:],b4_R[:],'k.')
plt.tick_params(axis='y', labelcolor=color)
plt.yscale('log')
plt.ylim([500, 100000])
_a = 425
_b = 1500  # Limits of x axis
#plt.xlim([_a, _b])
plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées')
plt.savefig("20241016/Test.svg")
plt.show()