# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:37:32 2025

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
dos = str(20250122)
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

data1 = np.loadtxt("" + dos + "/Ligne 15 PVDE66/" + name + " - bis cible.NL", skiprows=1)
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

data2 = np.loadtxt("" + dos + "/Ligne 15 PVDE66/" + name + " - ter.NL", skiprows=1)
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
fig.savefig("" + dos + "/Ligne 15 PVDE66/" + name + " - bis cible.svg")

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
figg.savefig("" + dos + "/Ligne 15 PVDE66/" + name + " - ter.svg")

###Traitement phase 1

b1_R = [];b1_v = [];b1_t = [];b1_Pulse = []
b2_R = [];b2_v = [];b2_t = [];b2_Pulse = []
b3_R = [];b3_v = [];b3_t = [];b3_Pulse = []
b4_R = [];b4_v = [];b4_t = [];b4_Pulse = []
b5_R = [];b5_v = [];b5_t = [];b5_Pulse = []
b6_R = [];b6_v = [];b6_t = [];b6_Pulse = []
b7_R = [];b7_v = [];b7_t = [];b7_Pulse = []
b8_R = [];b8_v = [];b8_t = [];b8_Pulse = []

for i in range(t_):
    if bande[0][4] <= c3_[i] <= bande[0][5]:
        b1_R.append(c3_[i])
        b1_Pulse.append(i + 1)
        continue

    elif bande[1][4] <= c3_[i] <= bande[1][5]:
        b2_R.append(c3_[i])
        b2_Pulse.append(i + 1)

    elif bande[2][4] <= c3_[i] <= bande[2][5]:
        b3_R.append(c3_[i])
        b3_Pulse.append(i + 1)

    elif bande[3][4] <= c3_[i] <= bande[3][5]:
        b4_R.append(c3_[i])
        b4_Pulse.append(i + 1)

    elif bande[4][4] <= c3_[i] <= bande[4][5]:
        b5_R.append(c3_[i])
        b5_Pulse.append(i + 1)
        continue

    elif bande[5][4] <= c3_[i] <= bande[5][5]:
        b6_R.append(c3_[i])
        b6_Pulse.append(i + 1)

    elif bande[6][4] <= c3_[i] <= bande[6][5]:
        b7_R.append(c3_[i])
        b7_Pulse.append(i + 1)

    elif bande[7][4] <= c3_[i] <= bande[7][5]:
        b8_R.append(c3_[i])
        b8_Pulse.append(i + 1)

plt.figure(figsize= (L,l))
color = 'tab:red'
plt.ylabel("Résistance en Ohm", fontsize=font, color=color)
plt.xlabel('# Pulses', fontsize=font, color=color)
plt.plot(b1_Pulse[:], b1_R[:], 'r.', markersize=dot_plot)
plt.plot(b2_Pulse[:], b2_R[:], 'b.', markersize=dot_plot)
plt.plot(b3_Pulse[:], b3_R[:], 'g.', markersize=dot_plot)
plt.plot(b4_Pulse[:], b4_R[:], 'k.', markersize=dot_plot)
plt.plot(b5_Pulse[:], b5_R[:], 'y.', markersize=dot_plot)
plt.plot(b6_Pulse[:], b6_R[:], 'c.', markersize=dot_plot)
plt.plot(b7_Pulse[:], b7_R[:], 'm.', markersize=dot_plot)
plt.plot(b8_Pulse[:], b8_R[:], 'b.', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color, labelsize= labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize= labsize)
plt.yscale('log')
#plt.ylim([0.8, 70])
_a = 0
_b = 630  # Limits of x axis
#plt.xlim([_a, _b])
#plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées Phase 1', fontsize=font)
plt.savefig(""+dos+"/Ligne 15 PVDE66/Données traitées Phase 1.svg")


####Statistique
## CDF & PDF=Gaussiennes
####Calcul de CDF: Données axe des ordonés

x0_s = np.sort(b1_R)
T0_s = len(x0_s)
y0_s = np.arange(start=1, stop=T0_s + 1, step=1) / float(T0_s)
if len(b1_R)>1:
    pdf_B1_s = np.gradient(y0_s * 100, x0_s)
x1_s = np.sort(b2_R)
T1_s = len(x1_s)
y1_s = np.arange(start=1, stop=T1_s + 1, step=1) / float(T1_s)
if len(b2_R)>1:
    pdf_B2_s = np.gradient(y1_s * 100, x1_s)
x2_s = np.sort(b3_R)
T2_s = len(x2_s)
y2_s = np.arange(start=1, stop=T2_s + 1, step=1) / float(T2_s)
if len(b3_R)>1:
    pdf_B3_s = np.gradient(y2_s * 100, x2_s)
x3_s = np.sort(b4_R)
T3_s = len(x3_s)
y3_s = np.arange(start=1, stop=T3_s + 1, step=1) / float(T3_s)
if len(b4_R)>1:
    pdf_B4_s = np.gradient(y3_s * 100, x3_s)
x4_s = np.sort(b5_R)
T4_s = len(x4_s)
y4_s = np.arange(start=1, stop=T4_s + 1, step=1) / float(T4_s)
if len(b5_R)>1:
    pdf_B5_s = np.gradient(y4_s * 100, x4_s)
x5_s = np.sort(b6_R)
T5_s = len(x5_s)
y5_s = np.arange(start=1, stop=T5_s + 1, step=1) / float(T5_s)
if len(b6_R)>1:
    pdf_B6_s = np.gradient(y5_s * 100, x5_s)
x6_s = np.sort(b7_R)
T6_s = len(x6_s)
y6_s = np.arange(start=1, stop=T6_s + 1, step=1) / float(T6_s)
if len(b7_R)>1:
    pdf_B7_s = np.gradient(y6_s * 100, x6_s)
x7_s = np.sort(c2_)
T7_s = len(x7_s)
y7_s = np.arange(start=1, stop=T7_s + 1, step=1) / float(T7_s)
if len(c2_)>1:
    pdf_B8_s = np.gradient(y7_s * 100, x7_s)
###Plot CDF
plt.figure()  ## CDF SET & RESET
plt.title('CDF SET & RESET',fontsize="14")
color = iter(cm.rainbow(np.linspace(0, 1, nb_bande)))
for i in range(nb_bande-1):
    c = next(color)
    plt.axvspan(bande[i][4], bande[i][5],
                ymin=0, ymax=150,
                color=c, alpha=0.5)

plt.plot(x0_s, y0_s * 100, 'r.', markersize=dot_plot)
if b2_R!=[]: plt.plot(x1_s, y1_s * 100, 'b.', markersize=dot_plot)
if b3_R!=[]: plt.plot(x2_s, y2_s * 100, 'g.', markersize=dot_plot)
if b4_R!=[]: plt.plot(x3_s, y3_s * 100, 'y.', markersize=dot_plot)
if b5_R!=[]: plt.plot(x4_s, y4_s * 100, 'c.', markersize=dot_plot)
if b6_R!=[]: plt.plot(x5_s, y5_s * 100, 'm.', markersize=dot_plot)
if b7_R!=[]: plt.plot(x6_s, y6_s * 100, 'k.', markersize=dot_plot)
plt.plot(x7_s, y7_s * 100, 'b.', markersize=dot_plot)
# plt.ylim([0,3e-6])
#plt.xlim([0.8, 70])
plt.xscale('log')
plt.xlabel("Résistance en ohm",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig(""+dos+"/Ligne 15 PVDE66/CDF SET & RESET.svg")


plt.show()