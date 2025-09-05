# -*- coding: utf-8 -*-
"""
Created on MON Oct 14 16:37:32 2024

@author: nouri
"""
globals().clear()
import numpy as np
import pandas as pd
from statistics import mean
from statistics import stdev
from math import pi
from math import sqrt
from math import exp
import matplotlib.pyplot as plt
import csv
from matplotlib.pyplot import cm

dot_plot = 12  # taille point plot
####Chargement fichier de données
rp = 47
Rp = []
Rth = []
e = 55
data = np.loadtxt("20241016/Ligne06_PVDE55-Ligne06_PVDE55-B_Rp47k_20240903-4p3V.NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1]
c4 = data[:, 3]
c9 = data[:, 8]
c10 = data[:, 9]
c12 = data[:, 11]
c11 = data[:, 10]
t = int(len(c4))
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
plt.tick_params(axis='x', labelcolor=color)
plt.yscale('log')
plt.ylim([500, 100000])
_a = 425;
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données brutes')
plt.legend(loc='upper right', fontsize="6")
plt.savefig("20241016/Données brutes.svg")

####Allocation bande de résistance
# Calcul largeur et espacement des bande
nb_bande = 4
bande = np.zeros((nb_bande + 1, 6))
gap_log = 0.35
min_bande1 = np.log10(900)
max_bande1 = np.log10(1100)
larg_range = max_bande1 - min_bande1
bande[0][0] = min_bande1
bande[0][1] = max_bande1
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
plt.savefig("20241016/Data brutes + bandes.svg")
###Traitement phase 1

b1_R = []
b1_Pulse = []
b2_R = []
b2_Pulse = []
b3_R = []
b3_Pulse = []
b4_R = []
b4_Pulse = []

for i in range(t):
    if bande[0][3] <= c4[i] <= bande[0][4]:
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
plt.ylabel('Resistance in ohm', fontsize="18", color=color)
plt.xlabel('# Pulses', fontsize="18", color=color)
plt.plot(b1_Pulse[:], b1_R[:], 'r.', markersize=dot_plot)
plt.plot(b2_Pulse[:], b2_R[:], 'b.', markersize=dot_plot)
plt.plot(b3_Pulse[:], b3_R[:], 'g.', markersize=dot_plot)
plt.plot(b4_Pulse[:], b4_R[:], 'k.', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color)
plt.yscale('log')
plt.ylim([500, 100000])
_a = 425
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
#plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées Phase 1', fontsize="14")
plt.savefig("20241016/Traitement Ph1.svg")

###traitment phase 2
b1_R_set = []
b1_R_reset = []
b1_Pulse_set = []
b1_Pulse_reset = []
b2_R_set = []
b2_R_reset = []
b2_Pulse_set = []
b2_Pulse_reset = []
b3_R_set = []
b3_R_reset = []
b3_Pulse_set = []
b3_Pulse_reset = []
b4_R_set = []
b4_R_reset = []
b4_Pulse_set = []
b4_Pulse_reset = []

for i in range(t):
    ##Bande 1 SET
    if bande[0][3] <= X1[i][1] <= bande[0][4]:
        b1_R_set.append(X1[i][1])
        b1_Pulse_set.append(i + 1)
    ##Bande 1 RESET
    if bande[0][3] <= X0[i][1] <= bande[0][4]:
        b1_R_reset.append(X0[i][1])
        b1_Pulse_reset.append(i + 1)
    ##Bande 2 SET
    elif bande[1][3] <= X1[i][1] <= bande[1][4]:
        b2_R_set.append(X1[i][1])
        b2_Pulse_set.append(i + 1)
    ##Bande 2 RESET
    elif bande[1][3] <= X0[i][1] <= bande[1][4]:
        b2_R_reset.append(X0[i][1])
        b2_Pulse_reset.append(i + 1)
    ##Bande 3 SET
    elif bande[2][3] <= X1[i][1] <= bande[2][4]:
        b3_R_set.append(X1[i][1])
        b3_Pulse_set.append(i + 1)
    ##Bande 3 RESET
    elif bande[2][3] <= X0[i][1] <= bande[2][4]:
        b3_R_reset.append(X0[i][1])
        b3_Pulse_reset.append(i + 1)
    ##Bande 4 SET
    elif bande[3][3] <= X1[i][1] <= bande[3][4]:
        b4_R_set.append(X1[i][1])
        b4_Pulse_set.append(i + 1)
    ##Bande 4 RESET
    elif bande[3][3] <= X0[i][1] <= bande[3][4]:
        b4_R_reset.append(X0[i][1])
        b4_Pulse_reset.append(i + 1)

b1_set = len(b1_R_set)
b1_reset = len(b1_R_reset)
b2_set = len(b2_R_set)
b2_reset = len(b2_R_reset)
b3_set = len(b3_R_set)
b3_reset = len(b3_R_reset)
b4_set = len(b4_R_set)
b4_reset = len(b4_R_reset)

b1_s = 1
b1_re = 1
b2_s = 1
b2_re = 1
b3_s = 1
b3_re = 1
b4_s = 1
b4_re = 1
for j in range(50):
    # SET bande 1
    for i in range(1, b1_set):
        if i < b1_s and b1_Pulse_set[i] - b1_Pulse_set[i - 1] <= 8:
            ind = b1_R_set.index(b1_R_set[i])
            b1_R_set.remove(b1_R_set[i])
            b1_R_set.insert(ind, b1_R_set[i - 1])
            b1_Pulse_set.remove(b1_Pulse_set[i])
            b1_Pulse_set.insert(ind, b1_Pulse_set[i - 1])
        b1_s = len(b1_R_set)
    # RESET bande 1
    for i in range(1, b1_reset):
        if i < b1_re and b1_Pulse_reset[i] - b1_Pulse_reset[i - 1] <= 8:
            ind = b1_R_reset.index(b1_R_reset[i])
            b1_R_reset.remove(b1_R_reset[i])
            b1_R_reset.insert(ind, b1_R_reset[i - 1])
            b1_Pulse_reset.remove(b1_Pulse_reset[i])
            b1_Pulse_reset.insert(ind, b1_Pulse_reset[i - 1])
        b1_re = len(b1_R_reset)
        # SET bande 2
        for i in range(1, b2_set):
            if i < b2_s and b2_Pulse_set[i] - b2_Pulse_set[i - 1] <= 8:
                ind = b2_R_set.index(b2_R_set[i])
                b2_R_set.remove(b2_R_set[i])
                b2_R_set.insert(ind, b2_R_set[i - 1])
                b2_Pulse_set.remove(b2_Pulse_set[i])
                b2_Pulse_set.insert(ind, b2_Pulse_set[i - 1])
            b2_s = len(b2_R_set)
        # RESET bande 2
        for i in range(1, b2_reset):
            if i < b2_re and b2_Pulse_reset[i] - b2_Pulse_reset[i - 1] <= 8:
                ind = b2_R_reset.index(b2_R_reset[i])
                b2_R_reset.remove(b2_R_reset[i])
                b2_R_reset.insert(ind, b2_R_reset[i - 1])
                b2_Pulse_reset.remove(b2_Pulse_reset[i])
                b2_Pulse_reset.insert(ind, b2_Pulse_reset[i - 1])
            b2_re = len(b2_R_reset)
        # SET bande 3
        for i in range(1, b3_set):
            if i < b3_s and b3_Pulse_set[i] - b3_Pulse_set[i - 1] <= 8:
                ind = b3_R_set.index(b3_R_set[i])
                b3_R_set.remove(b3_R_set[i])
                b3_R_set.insert(ind, b3_R_set[i - 1])
                b3_Pulse_set.remove(b3_Pulse_set[i])
                b3_Pulse_set.insert(ind, b3_Pulse_set[i - 1])
            b3_s = len(b3_R_set)
        # RESET bande 3
        for i in range(1, b3_reset):
            if i < b3_re and b3_Pulse_reset[i] - b3_Pulse_reset[i - 1] <= 8:
                ind = b3_R_reset.index(b3_R_reset[i])
                b3_R_reset.remove(b3_R_reset[i])
                b3_R_reset.insert(ind, b3_R_reset[i - 1])
                b3_Pulse_reset.remove(b3_Pulse_reset[i])
                b3_Pulse_reset.insert(ind, b3_Pulse_reset[i - 1])
            b3_re = len(b3_R_reset)
        # SET bande 4
        for i in range(1, b4_set):
            if i < b4_s and b4_Pulse_set[i] - b4_Pulse_set[i - 1] <= 8:
                ind = b4_R_set.index(b4_R_set[i])
                b4_R_set.remove(b4_R_set[i])
                b4_R_set.insert(ind, b4_R_set[i - 1])
                b4_Pulse_set.remove(b4_Pulse_set[i])
                b4_Pulse_set.insert(ind, b4_Pulse_set[i - 1])
            b4_s = len(b4_R_set)
        # RESET bande 4
        for i in range(1, b4_reset):
            if i < b4_re and b4_Pulse_reset[i] - b4_Pulse_reset[i - 1] <= 8:
                ind = b4_R_reset.index(b4_R_reset[i])
                b4_R_reset.remove(b4_R_reset[i])
                b4_R_reset.insert(ind, b4_R_reset[i - 1])
                b4_Pulse_reset.remove(b4_Pulse_reset[i])
                b4_Pulse_reset.insert(ind, b4_Pulse_reset[i - 1])
            b4_re = len(b4_R_reset)
###Eliminer les duplicata dans mes list

b1_R_set = list(dict.fromkeys(b1_R_set))
b1_R_reset = list(dict.fromkeys(b1_R_reset))
b2_R_set = list(dict.fromkeys(b2_R_set))
b2_R_reset = list(dict.fromkeys(b2_R_reset))
b3_R_set = list(dict.fromkeys(b3_R_set))
b3_R_reset = list(dict.fromkeys(b3_R_reset))
b4_R_set = list(dict.fromkeys(b4_R_set))
b4_R_reset = list(dict.fromkeys(b4_R_reset))
b1_Pulse_set = list(dict.fromkeys(b1_Pulse_set))
b1_Pulse_reset = list(dict.fromkeys(b1_Pulse_reset))
b2_Pulse_set = list(dict.fromkeys(b2_Pulse_set))
b2_Pulse_reset = list(dict.fromkeys(b2_Pulse_reset))
b3_Pulse_set = list(dict.fromkeys(b3_Pulse_set))
b3_Pulse_reset = list(dict.fromkeys(b3_Pulse_reset))
b4_Pulse_set = list(dict.fromkeys(b4_Pulse_set))
b4_Pulse_reset = list(dict.fromkeys(b4_Pulse_reset))

####Plot figure

plt.figure(3)
color = 'tab:red'
plt.ylabel('Resistance in ohm',fontsize="18", color=color)
plt.xlabel('# Pulses',fontsize="18", color=color)
plt.plot(b1_Pulse_set[:], b1_R_set[:], 'r.', markersize=dot_plot)
plt.plot(b1_Pulse_reset[:], b1_R_reset[:], 'r*', markersize=dot_plot)
plt.plot(b2_Pulse_set[:], b2_R_set[:], 'b.', markersize=dot_plot)
plt.plot(b2_Pulse_reset[:], b2_R_reset[:], 'b*', markersize=dot_plot)
plt.plot(b3_Pulse_set[:], b3_R_set[:], 'g.', markersize=dot_plot)
plt.plot(b3_Pulse_reset[:], b3_R_reset[:], 'g*', markersize=dot_plot)
plt.plot(b4_Pulse_set[:], b4_R_set[:], 'k.', markersize=dot_plot)
plt.plot(b4_Pulse_reset[:], b4_R_reset[:], 'k*', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color)
plt.tick_params(axis='x', labelcolor=color)
plt.yscale('log')
plt.ylim([500, 100000])
_a = 425
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées Ph2: SET & RESET',fontsize="14")
plt.savefig("20241016/Données traitées Ph2 SET & RESET.svg")

##Plot données SET
plt.figure(4)
color = 'tab:red'
plt.ylabel('Resistance in ohm',fontsize="18", color=color)
plt.xlabel('# Pulses',fontsize="18", color=color)
plt.plot(b1_Pulse_set[:], b1_R_set[:], 'r.', markersize=dot_plot)
plt.plot(b2_Pulse_set[:], b2_R_set[:], 'b.', markersize=dot_plot)
plt.plot(b3_Pulse_set[:], b3_R_set[:], 'g.', markersize=dot_plot)
plt.plot(b4_Pulse_set[:], b4_R_set[:], 'k.', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color)
plt.tick_params(axis='x', labelcolor=color)
plt.yscale('log')
plt.ylim([500, 100000])
_a = 425
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées SET only',fontsize="14")
plt.savefig("20241016/Données traitées SET only.svg")

##Plot données RESET
plt.figure(5)
color = 'tab:red'
plt.ylabel('Resistance in ohm',fontsize="18", color=color)
plt.xlabel('# Pulses',fontsize="18", color=color)
plt.plot(b1_Pulse_reset[:], b1_R_reset[:], 'r*', markersize=dot_plot)
plt.plot(b2_Pulse_reset[:], b2_R_reset[:], 'b*', markersize=dot_plot)
plt.plot(b3_Pulse_reset[:], b3_R_reset[:], 'g*', markersize=dot_plot)
plt.plot(b4_Pulse_reset[:], b4_R_reset[:], 'k*', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color)
plt.tick_params(axis='x', labelcolor=color)
plt.yscale('log')
plt.ylim([500, 100000])
_a = 425
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées RESET only',fontsize="14")
plt.savefig("20241016/Données traitées RESET only.svg")

####Statistique
## CDF

##Calcul de probilité cumulative: Données axe des ordonés
x0_s = np.sort(b1_R_set)
x0_re = np.sort(b1_R_reset)
x1_s = np.sort(b2_R_set)
x1_re = np.sort(b2_R_reset)
x2_s = np.sort(b3_R_set)
x2_re = np.sort(b3_R_reset)
x3_s = np.sort(b4_R_set)
x3_re = np.sort(b4_R_reset)
T0_s = len(x0_s)
T0_re = len(x0_re)
T1_s = len(x1_s)
T1_re = len(x1_re)
T2_s = len(x2_s)
T2_re = len(x2_re)
T3_s = len(x3_s)
T3_re = len(x3_re)
y0_s = np.arange(start=1, stop=T0_s + 1, step=1) / float(T0_s)
y0_re = np.arange(start=1, stop=T0_re + 1, step=1) / float(T0_re)
y1_s = np.arange(start=1, stop=T1_s + 1, step=1) / float(T1_s)
y1_re = np.arange(start=1, stop=T1_re + 1, step=1) / float(T1_re)
y2_s = np.arange(start=1, stop=T2_s + 1, step=1) / float(T2_s)
y2_re = np.arange(start=1, stop=T2_re + 1, step=1) / float(T2_re)
y3_s = np.arange(start=1, stop=T3_s + 1, step=1) / float(T3_s)
y3_re = np.arange(start=1, stop=T3_re + 1, step=1) / float(T3_re)

plt.figure(6)  ## CDF SET & RESET
plt.title('CDF SET & RESET',fontsize="14")
plt.plot(x0_s, y0_s * 100, 'r.', markersize=dot_plot)
plt.plot(x0_re, y0_re * 100, 'r*', markersize=dot_plot)
plt.plot(x1_s, y1_s * 100, 'b.', markersize=dot_plot)
plt.plot(x1_re, y1_re * 100, 'b*', markersize=dot_plot)
plt.plot(x2_s, y2_s * 100, 'g.', markersize=dot_plot)
plt.plot(x2_re, y2_re * 100, 'g*', markersize=dot_plot)
plt.plot(x3_s, y3_s * 100, 'k.', markersize=dot_plot)
plt.plot(x3_re, y3_re * 100, 'k*', markersize=dot_plot)
# plt.ylim([0,3e-6])
plt.xlim([500, 100000])
plt.xscale('log')
plt.xlabel("Résistance (ohm)",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig("20241016/CDF SET & RESET.svg")

plt.figure(7)  ## CDF SET ONLY
plt.title('CDF SET ONLY',fontsize="14")
plt.plot(x0_s, y0_s * 100, 'r.', markersize=dot_plot)
plt.plot(x1_s, y1_s * 100, 'b.', markersize=dot_plot)
plt.plot(x2_s, y2_s * 100, 'g.', markersize=dot_plot)
plt.plot(x3_s, y3_s * 100, 'k.', markersize=dot_plot)
# plt.ylim([0,3e-6])
plt.xlim([500, 100000])
plt.xscale('log')
plt.xlabel("Résistance (ohm)",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig("20241016/CDF SET ONLY.svg")

plt.figure(8)  ## CDF RESET ONLY
plt.title('CDF RESET ONLY',fontsize="14")
plt.plot(x0_re, y0_re * 100, 'r*', markersize=dot_plot)
plt.plot(x1_re, y1_re * 100, 'b*', markersize=dot_plot)
plt.plot(x2_re, y2_re * 100, 'g*', markersize=dot_plot)
plt.plot(x3_re, y3_re * 100, 'k*', markersize=dot_plot)
# plt.ylim([0,3e-6])
plt.xlim([500, 100000])
plt.xscale('log')
plt.xlabel("Résistance (ohm)",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig("20241016/CDF RESET ONLY.svg")

## Histogramme
plt.figure(9)
# bande1
plt.hist(b1_R_set,weights=b1_Pulse_set, rwidth=0.9, color='k', label='SET')
plt.hist(b1_R_reset,weights=b1_Pulse_reset, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(int(bande[0][3])) + "," + str(int(bande[0][4])) + "]",fontsize="14")
plt.xlabel("Résistance (ohm)",fontsize="18")
plt.ylabel("Nombre de Pulse",fontsize="18")
plt.legend()
plt.savefig("20241016/hist-bande1.svg")
plt.figure(10)
# bande2
plt.hist(b2_R_set,weights=b2_Pulse_set, rwidth=0.9, color='k', label='SET')
plt.hist(b2_R_reset,weights=b2_Pulse_reset, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(int(bande[1][3])) + "," + str(int(bande[1][4])) + "]",fontsize="14")
plt.xlabel("Résistance (ohm)",fontsize="18")
plt.ylabel("Nombre de Pulse",fontsize="18")
plt.legend()
plt.savefig("20241016/hist-bande2.svg")
plt.figure(11)
# bande3
plt.hist(b3_R_set, rwidth=0.9, color='k', label='SET')
plt.hist(b3_R_reset, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(int(bande[2][3])) + "," + str(int(bande[2][4])) + "]",fontsize="14")
plt.xlabel("Résistance (ohm)",fontsize="18")
plt.ylabel("Nombre de Pulse",fontsize="18")
plt.legend()
plt.savefig("20241016/hist-bande3.svg")
plt.figure(12)
# bande4
plt.hist(b4_R_set, rwidth=0.9, color='k', label='SET')
plt.hist(b4_R_reset, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(int(bande[3][3])) + "," + str(int(bande[3][4])) + "]",fontsize="14")
plt.xlabel("Résistance (ohm)",fontsize="18")
plt.ylabel("Nombre de Pulse",fontsize="18")
plt.legend()
plt.savefig("20241016/hist-bande4.svg")
## Gaussiennes
plt.figure(13)
# bande1
gauss=[]
x_g=[]
me = mean(b1_R_set[:])
std = stdev(b1_R_set[:])
variance = np.square(std)
x_g [:]= b1_Pulse_set[:]
g=len(x_g)
for l in range(g):
    gauss.append(exp(-np.square(x_g[l]-me)/2*variance)/(sqrt(2*pi*variance)))

plt.plot(x_g[:], gauss[:])
plt.ylabel('Distribution de Gauss')
plt.savefig("20241016/Gauss-bande1.svg")
plt.show()
