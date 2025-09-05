# -*- coding: utf-8 -*-

"""
Created on Tue Dec 10 15:37:32 2024
@author: nouri
"""

globals().clear()
import numpy as np
from math import pi
import matplotlib.pyplot as plt

dos = str(20250122)
dot_plot = 12  # taille point plot
labsize = 20 # Taille graduation axe
font = 26 # Taille titres des axes
fontLEG = 8 # taille legende
l=6 #largeur fig
L=7 #longueur fig
name = input("Nom fichier :")
rp = float(input('Résistance pristine: '))
ep = 50  # float(input('Epaisseur en nm: '))
Via = 1  # float(input('Taille du VIA en um: '))
Cr = 30  # float(input('% du Chrome: '))
cp = (rp * 1e+3)
data = np.loadtxt("" + dos + "/" + name + ".NL", skiprows=4, usecols=range(0, 14))
c4 = data[:, 3]
t_ = int(len(c4))
# Garder un même nombre de pulse pour la comparaison.
l_b =0
l_h =t_  ## t_ SI TOUT LE PLOT

# Limits axe des y
_c = 800
_d = 100000
# Limits axe des x
_a = 1560
_b = 2280

c2 = data[l_b:l_h, 1]
c4 = data[l_b:l_h, 3]
c9 = data[l_b:l_h, 8]
c10 = data[l_b:l_h, 9]
c12 = data[l_b:l_h, 11]
c11 = data[l_b:l_h, 10]
t = int(len(c4)) # Egalisation nb de pulse
A0 = []
X0 = np.zeros((int(t), 5))
X1 = np.zeros((int(t), 5))
k = 0
r = 0


fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 9), height_ratios=[3, 1, 1])
for i in range(t):
    # RESET conditions
    if c9[i] == 6E-9:
        X0[i][1] = c4[i]
        X0[i][2] = c12[i]
        X0[i][4] = int((c9[i] + c10[i] + c11[i]) * 10 ** 9)
        X0[i][0] = i + 1
        X0[i][3] = i + 1
    # Set conditions
    else:
        if c9[i] != 0:
            X1[i][1] = c4[i]
            X1[i][2] = c12[i]
            X1[i][4] = int((c9[i] + c10[i] + c11[i]) * 10 ** 9)
            X1[i][0] = i + 1
            X1[i][3] = i + 1

a_s = []; a_re = []
b_s = []; b_re = []
xa = []
Cp = []
cth = []
Rp = []
Rth = []
VIA = Via / 2
if Cr == 30:
    ro = 35
else:
    if Cr == 5:
        ro = 1.3
r_th = (ro * ep * 1E-7) / (pi * VIA * 1e-4 * VIA * 1e-4)  # La résistivité du 30% Cr est 35 Ohm.cm
c_th = (1 / (r_th)) * 1e+6
for ut in range(t):
    Rp.append(float(rp * 1e+3))  # Mettre la valeur de
    Cp.append(float(1 / (rp * 1e+3)) / float(1 / (rp * 1e+3)))  # Cp normalisée
    # R th dans la list
    Rth.append(float(r_th))
    cth.append((float(1 / r_th)) / float(1 / (rp * 1e+3)))

for i in range(t): #Isolation de chaque SET
    if (X1[i-1][1] == 0 and X1[i-2][1] == 0 and X1[i-3][1] == 0 and X1[i][1] != 0) or (
        i == 0 and X1[i][1] != 0
    ):
        a_s.append(i)
    if (i < t - 3 and X1[i + 1][1] == 0 and X1[i + 2][1] == 0 and X1[i + 3][1] == 0 and X1[i][1] != 0) or (
        i==t-1 and X1[i][1] != 0
    ):
        b_s.append(i)

    if (X0[i - 1][1] == 0 and X0[i - 2][1] == 0 and X0[i - 3][1] == 0 and X0[i][1] != 0) or (
            i == 0 and X0[i][1] != 0
    ):
        a_re.append(i)
    if (i < t - 3 and X0[i + 1][1] == 0 and X0[i + 2][1] == 0 and X0[i + 3][1] == 0 and X0[i][1] != 0) or (
            X0[i][1] != 0 and i == t - 1
    ):
        b_re.append(i)




# plt.grid()
# Plot pour la résistance
color = 'tab:red'
ax1.set_ylabel('Resistance in Ohm', fontsize=font, color=color)
ax1.plot(X0[:, 0], X0[:, 1], 'r.', X1[:, 0], X1[:, 1], 'k.', markersize=dot_plot)
#ax1.plot(X1[:, 0], Rp[:], 'c-', label='Rp= ' + str(f"{(rp):.2f}") + ' k')  # Plot Rp
#ax1.plot(X1[:, 0], Rth[:], 'b-', label='Rth=' + str(f"{(r_th*1e-3):.2f}") + ' k', markersize=dot_plot)  # Plot R th
ax1.tick_params(axis='y', labelcolor=color, labelsize=labsize)
ax1.set_yscale('log')
ax1.set_ylim([_c, _d])
#ax1.set_xlim([_a,_b])
ax1.set_xticks([])
ax1.grid(color='k', axis='y')
#ax1.set_title(name, fontsize=font)
#ax1.legend(loc='upper right', fontsize=fontLEG)
# Plot pour la tension
color = 'tab:blue'
ax2.set_ylabel("Voltage(V)", fontsize=font, color=color)
ax2.plot(X0[:, 3], X0[:, 2], 'r.', X1[:, 3], X1[:, 2], 'k.', markersize=dot_plot)  # Plot tension
ax2.set_xticks([])
ax2.set_yticks(np.arange(2, 10, step=2))
ax2.set_ylim([1, 8])
#ax2.set_xlim([_a,_b])
ax2.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax2.tick_params(axis='y', labelcolor=color, labelsize=labsize)
fig.tight_layout()
# Plot du temps
ax3.set_ylabel('Time(ns)', fontsize=font, color=color)
ax3.plot(X0[:, 3], X0[:, 4], 'r.', X1[:, 3], X1[:, 4], 'k.', markersize=dot_plot)  # Plot temps de pulse
ax3.set_yscale('linear')
#ax3.set_xticks([])
ax3.set_yticks(np.arange(50, 350, step=100))
ax3.set_ylim([10, 350])
#ax3.set_xlim([_a,_b])
ax3.set_xlabel('Number of pulse', fontsize=font)
ax3.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax3.tick_params(axis='y', labelcolor=color, labelsize=labsize)
ax3.tick_params(axis='x', labelsize=labsize)
fig.tight_layout()
fig.savefig("" + dos + "/" + name + ".svg")
plt.show()