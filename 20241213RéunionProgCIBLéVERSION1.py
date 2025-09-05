# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:37:32 2024

@author: nouri
"""
globals().clear()
import numpy as np
import pandas as pd
from statistics import mean
from statistics import median
from statistics import stdev
from math import pi
from math import sqrt
from math import exp
import matplotlib.pyplot as plt
import csv
from matplotlib.pyplot import cm
### Fenêtre Memoire
r_on=1500
r_off=15000
# % écart entre le centre et le bord
ecart=25
dos = str(20241218)
dot_plot = 10  # taille point plot
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
name = input("Nom fichier :")
data1 = np.loadtxt("" + dos + "/Ligne 15/" + name + " - bis cible.NL", skiprows=1)
c2 = data1[:, 1] #V_set
c3 = data1[:, 2] #R_set
c4 = data1[:, 3] #V_reset
c5 = data1[:, 4] #R_reset
t = int(len(c4)) # Egalisation nb de pulse
A0 = []
X0 = np.zeros((int(t), 5))
X1 = np.zeros((int(t), 5))
k = 0
r = 0

data2 = np.loadtxt("" + dos + "/Ligne 15/" + name + " - ter.NL", skiprows=1)
c2_ = data2[:, 1]
c3_ = data2[:, 2]
c4_ = data2[:, 3]

t_ = int(len(c4_)) # Egalisation nb de pulse
A0_ = []
X0_ = np.zeros((int(t_), 5))
X1_ = np.zeros((int(t_), 5))


# Limits axe des y
_c = 500
_d = 50000
# Limits axe des x
_a = 831
_b = 1800

#### Allocation bande de conductance
# Calcul largeur et espacement des bande
nb_bande = 8
bande = np.zeros((nb_bande + 1, 6))
gap_log_ce_et_bo = ecart  ## % écart entre le centre et le bord
R_on = np.log10(r_on)  ## Limite basse fenêtre mémoire
R_off = np.log10(r_off) ## Limite haute fenêtre mémoire
gap_log_c = (R_off - R_on)/nb_bande   ## Gap entre les centres des bandes
bande[0][0] = R_on
bande[7][0] = R_off
for i in range(nb_bande):
    bande[i + 1][0] = bande[i][0] + (R_off - R_on)/(nb_bande - 1) ## Centre des bandes en log
    bande[i][1] = bande[i][0] - (gap_log_c * gap_log_ce_et_bo/100) ## Borne inf.
    bande[i][2] = bande[i][0] + (gap_log_c * gap_log_ce_et_bo/100) ## Borne sup.
    bande[i][3] = 10 ** bande[i][0]
    bande[i][4] = 10 ** bande[i][1]
    bande[i][5] = 10 ** bande[i][2]
#DF = pd.DataFrame(bande)
#DF.to_csv(""+dos+"/bande.csv", index=False )
#plt.savefig(""+dos+"/Data brutes + bandes.svg")

fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 9), height_ratios=[3, 1])
Rp = []
Rth = []
Rp_ = []
Rth_ = []
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
    # R th dans la list
    Rth.append(float(r_th))
for ut in range(t_):
    Rp_.append(float(rp * 1e+3))  # Mettre la valeur de
    # R th dans la list
    Rth_.append(float(r_th))
# Plot pour la résistance
color = iter(cm.rainbow(np.linspace(0, 1, nb_bande)))
for i in range(nb_bande):
    c = next(color)
    ax1.axhspan(bande[i][4], bande[i][5],
                xmin=0, xmax=t,
                color=c, alpha=0.5)
color = 'tab:red'
x=np.linspace(1,t,t)
ax1.set_ylabel('Résistance en Ohm', fontsize=font, color=color)
ax1.plot(x[:], c3[:], 'b.', x[:], c5[:], 'g.', markersize=dot_plot)
ax1.plot(x[:], Rp[:], 'c-', label='Rp= ' + str(f"{(rp):.2f}") + ' k')  # Plot Rp
ax1.plot(x[:], Rth[:], 'b-', label='Rth=' + str(f"{(r_th*1e-3):.2f}") + ' k', markersize=dot_plot)  # Plot R th
ax1.tick_params(axis='y', labelcolor=color, labelsize=labsize)
ax1.set_yscale('log')
ax1.set_ylim([_c, _d])
#ax1.set_xlim([_a,_b])
ax1.set_xticks([])
ax1.grid(color='k', axis='y')
ax1.set_title(name, fontsize=font)
ax1.legend(loc='upper right', fontsize=fontLEG)
# Plot pour la tension
color = 'tab:blue'
ax2.set_ylabel("Voltage(V)", fontsize=font, color=color)
ax2.plot(x[:], c2[:], 'g.', x[:], c4[:], 'b.', markersize=dot_plot)  # Plot tension
ax2.set_xticks([])
ax2.set_yticks(np.arange(2, 10, step=2))
ax2.set_ylim([1, 8])
#ax2.set_xlim([_a,_b])
ax2.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax2.tick_params(axis='y', labelcolor=color, labelsize=labsize)
fig.tight_layout()
"""
# Plot du temps
ax3.set_ylabel('Time(ns)', fontsize=font, color=color)
ax3.plot(X0[:, 3], X0[:, 4], 'r.', X1[:, 3], X1[:, 4], 'k.', markersize=dot_plot)  # Plot temps de pulse
ax3.set_yscale('linear')
ax3.set_yticks(np.arange(50, 350, step=100))
ax3.set_ylim([10, 350])
#ax3.set_xlim([_a,_b])
ax3.set_xlabel('Number of pulse', fontsize=font)
ax3.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax3.tick_params(axis='y', labelcolor=color, labelsize=labsize)
ax3.tick_params(axis='x', labelsize=labsize)
"""
fig.tight_layout()
fig.savefig("" + dos + "/Ligne 15/" + name + " - bis cible.svg")

figg, (ax1_, ax2_) = plt.subplots(2, figsize=(10, 9), height_ratios=[3, 1])

color = iter(cm.rainbow(np.linspace(0, 1, nb_bande)))
for i in range(nb_bande):
    c = next(color)
    ax1_.axhspan(bande[i][4], bande[i][5],
                xmin=0, xmax=t_,
                color=c, alpha=0.5)
color = 'tab:red'
x=np.linspace(1,t_,t_)

ax1_.set_ylabel('Résistance en Ohm', fontsize=font, color=color)
ax1_.plot(x[:], c2_[:], 'b.', x[:], c3_[:], 'g.', markersize=dot_plot)
ax1_.plot(x[:], Rp_[:], 'c-', label='Rp= ' + str(f"{(rp):.2f}") + ' k')  # Plot Rp
ax1_.plot(x[:], Rth_[:], 'b-', label='Rth=' + str(f"{(r_th*1e-3):.2f}") + ' k', markersize=dot_plot)  # Plot R th
ax1_.tick_params(axis='y', labelcolor=color, labelsize=labsize)
ax1_.set_yscale('log')
ax1_.set_ylim([_c, _d])
#ax1.set_xlim([_a,_b])
ax1_.set_xticks([])
ax1_.grid(color='k', axis='y')
ax1_.set_title(name, fontsize=font)
ax1_.legend(loc='upper right', fontsize=fontLEG)
# Plot pour la tension
color = 'tab:blue'
ax2_.set_ylabel("# pulses", fontsize=font, color=color)
ax2_.plot(x[:], c4_[:], 'k.', markersize=dot_plot)  # Plot tension
ax2_.set_xticks([])
ax2_.set_yticks(np.arange(1, 50, step=5))
ax2_.set_ylim([1, 50])
#ax2_.set_xlim([_a,_b])
ax2_.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax2_.tick_params(axis='y', labelcolor=color, labelsize=labsize)
figg.tight_layout()
figg.savefig("" + dos + "/Ligne 15/" + name + " - ter.svg")
plt.show()