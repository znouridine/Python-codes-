# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:37:32 2024
@author: nouri
Objectif Prog:
Trouver sur le nombre total de cycle combien de fois on tombe dans une
bande en SET et RESET
"""
globals().clear()
import numpy as np
import pandas as pd
from math import pi
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

dos= str(20250312)
dot_plot = 12  # taille point plot
labsize = 18 # Taille graduation axe
font = 20 # Taille titres des axes
fontLEG = 16 # taille legende
l=6 #largeur fig
L=7 #longueur fig
name = input("Nom fichier :")
Via = 1
rp = float(input('Résistance pristine: '))
#ep = float(input('Epaisseur en nm: '))
ep=50
####Chargement fichier de données
cp = float(rp*1e+3)
Cp = []
cth = []
data = np.loadtxt(""+dos+"/"+name+".NL", skiprows=4, usecols=range(0, 14))
c4 = data[:, 3]
t_ = int(len(c4))
# Garder un même nombre de pulse pour la comparaison.
l_b = 0
l_h = t_  ## t_ SI TOUT LE PLOT
### Fenêtre Memoire
R_on= 2000
R_off= 20000

c2 = data[l_b:l_h, 1]
c4 = data[l_b:l_h, 3]
c9 = data[l_b:l_h, 8]
c10 = data[l_b:l_h, 9]
c12 = data[l_b:l_h, 11]
c11 = data[l_b:l_h, 10]
t = int(len(c4))

# Tracé des résisitance
A0 = []
X0 = np.zeros((int(t), 6))
X0_ = np.zeros((int(t), 6))
X1 = np.zeros((int(t), 6))
X1_ = np.zeros((int(t), 6))
k = 0
r = 0
for i in range(t):
    # RESET conditions
    if c9[i] == 6E-9:
        X0[i][1] = c4[i]
        X0[i][2] = c12[i]
        X0[i][4] = int((c9[i] + c10[i] + c11[i]) * 10 ** 9)
        X0[i][0] = i+1
        X0[i][3] = i+1
    # Set conditions
    else:
        if c9[i] != 0:
            X1[i][1] = c4[i]
            X1[i][2] = c12[i]
            X1[i][4] = int((c9[i] + c10[i] + c11[i]) * 10 ** 9)
            X1[i][0] = i+1
            X1[i][3] = i+1

Cp = []
cth = []
Rp = []
Rth = []
VIA = Via / 2
r_th = (35 * ep * 1E-7) / (pi * VIA * 1e-4 * VIA * 1e-4)  # La résistivité du 30% Cr est 35 Ohm.cm
c_th = (1 / (r_th)) * 1e+6
for p in range(t):
    Rp.append(float(rp * 1e+3))  # Mettre la valeur de
    Cp.append(float(1 / (rp * 1e+3)) / float(1 / (rp * 1e+3)))  # Cp normalisée
    # R th dans la list
    Rth.append(float(r_th))
    cth.append((float(1 / r_th)) / float(1 / (rp * 1e+3)))

a_s = []; a_re = []
b_s = []; b_re = []
c_s = []; c_re = []
d_s = []; d_re = []


def nbt_cycle ():
    for pul in range(t):
        # Isolation de chaque SET
        if (X1[pul - 1][1] == 0 and X1[pul - 2][1] == 0 and X1[pul - 3][1] == 0 and X1[pul][1] != 0) or (
                pul == 0 and X1[pul][1] != 0
        ):
            a_s.append(pul)
        if (pul < t - 3 and X1[pul + 1][1] == 0 and X1[pul + 2][1] == 0 and X1[pul + 3][1] == 0 and X1[pul][1] != 0) or (
                pul == t - 1 and X1[pul][1] != 0
        ):
            b_s.append(pul)
        # Isolation de chaque RESET
        if (X0[pul - 1][1] == 0 and X0[pul - 2][1] == 0 and X0[pul - 3][1] == 0 and X0[pul][1] != 0) or (
                pul == 0 and X0[pul][1] != 0
        ):
            a_re.append(pul)
        if (pul < t - 3 and X0[pul + 1][1] == 0 and X0[pul + 2][1] == 0 and X0[pul + 3][1] == 0 and X0[pul][1] != 0) or (
                X0[pul][1] != 0 and pul == t - 1
        ):
            b_re.append(pul)

    return min( len(a_s) , len(a_re) )
nbt_cy = nbt_cycle ()

# plt.grid()
# Plot pour la résistance
plt.figure(1, figsize= (L,l))
color = 'tab:red'
plt.ylabel('Résistance en Ohm', color=color, fontsize=font)
plt.xlabel('# Pulses', color=color, fontsize=font)
plt.plot(X0[:, 0], X0[:, 1], 'r.', X1[:, 0], X1[:, 1], 'k.', markersize=dot_plot)
plt.plot(X1[:, 0], Rp[:], 'c-', label='Rp= ' + str(f"{(rp):.2f}") + ' k', markersize=dot_plot)  # Plot Cp
plt.plot(X1[:, 0], Rth[:], 'b-', label='Rth=' + str(f"{(r_th*1e-3):.2f}") + ' k', markersize=dot_plot)  # Plot C th
plt.tick_params(axis='y', labelcolor=color, labelsize= labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize= labsize)
plt.yscale('log')
plt.ylim([800, 50000])
_a = 0
_b = 630  # Limits of x axis
#plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données brutes ('+ str(f"{(nbt_cy):.2f}") +' cycles)', fontsize=font)
plt.legend(loc='upper right', fontsize=fontLEG)
plt.savefig(""+dos+"/Données brutes.svg")

#### Allocation bande de conductance
# Calcul largeur et espacement des bande
nb_bande = 8
bande = np.zeros((nb_bande + 1, 6))
gap_log_ce_et_bo = 25  ## % écart entre le centre et le bord
R_on = np.log10(R_on)  ## Limite basse fenêtre mémoire
R_off = np.log10(R_off) ## Limite haute fenêtre mémoire
gap_log_c = (R_off - R_on)/nb_bande   ## Gap entre les centres des bandes
color = iter(cm.rainbow(np.linspace(0, 1, nb_bande)))
bande[0][0] = R_on
bande[7][0] = R_off
#color = ['r', 'b', 'k', 'y', 'c', 'm', 'r', 'b']
for i in range(nb_bande):
    bande[i + 1][0] = bande[i][0] + (R_off - R_on)/(nb_bande - 1) ## Centre des bandes en log
    bande[i][1] = bande[i][0] - (gap_log_c * gap_log_ce_et_bo/100) ## Borne inf.
    bande[i][2] = bande[i][0] + (gap_log_c * gap_log_ce_et_bo/100) ## Borne sup.
    bande[i][3] = 10 ** bande[i][0]
    bande[i][4] = 10 ** bande[i][1]
    bande[i][5] = 10 ** bande[i][2]
    c = next(color)
    plt.axhspan(bande[i][4], bande[i][5],
                xmin=0, xmax=t,
                color=c, alpha=0.5)
DF = pd.DataFrame(bande)
DF.to_csv(""+dos+"/bande.csv", index=False )
plt.savefig(""+dos+"/Data brutes + bandes.svg")

plt.show()

