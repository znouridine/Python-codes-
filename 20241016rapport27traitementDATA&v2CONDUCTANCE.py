# -*- coding: utf-8 -*-
"""
Created on MON Oct 17 16:37:32 2024

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
name = input("Nom fichier :")
rp = float(input('Résistance pristine: '))
ep = float(input('Epaisseur en nm: '))
####Chargement fichier de données
cp = (1/rp*1e+3)*1e+6
Cp = []
cth = []
ep = 55
data = np.loadtxt("20241016/"+name+".NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1]
c4 = (1/data[:, 3])*1e+6
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

Cp = []
cth = []
c_th =(1/((35 * ep * 1E-7) / (pi * 0.6e-4 * 0.6e-4)))*1e+6
for p in range(t):
    Cp.append(float(cp))  # Mettre la valeur de
    # R th dans la list
    cth.append(float(c_th))

# plt.grid()
# Plot pour la résistance
plt.figure(1)
color = 'tab:red'
plt.ylabel('Conductance in 10^(-6) S', color=color)
plt.xlabel('# Pulses', color=color)
plt.plot(X0[:, 0], X0[:, 1], 'r.',
         X1[:, 0], X1[:, 1], 'k.',
         X0[:, 0], Cp[:], 'c-',
         X0[:, 0], cth[:], 'b-')
plt.plot(X1[:, 0], Cp[:], 'c-',
         label='Cp=' + str(round( cp , 2)) + 'uS')
plt.plot(X1[:, 0], cth[:], 'b-',
         label='cth=' + str(round(c_th ,2 )) + ' uS')
plt.tick_params(axis='y', labelcolor=color)
plt.tick_params(axis='x', labelcolor=color)
plt.yscale('log')
plt.ylim([1e+6/100000, 1e+6/500])
_a = 425;
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données brutes')
plt.legend(loc='upper right', fontsize="6")
plt.savefig("20241016/Données brutes.svg")

####Allocation bande de conductance
# Calcul largeur et espacement des bande
nb_bande = 4
bande = np.zeros((nb_bande + 1, 6))
gap_log = 0.30
min_bande1 = np.log10(900)
max_bande1 = np.log10(1050)
larg_range = max_bande1 - min_bande1
bande[0][0] = min_bande1
bande[0][1] = max_bande1
color = iter(cm.rainbow(np.linspace(0, 1, nb_bande)))
for i in range(nb_bande):
    bande[i + 1][0] = bande[i][1] + gap_log
    bande[i + 1][1] = bande[i + 1][0] + larg_range
    bande[i][2] = (bande[i][0] + bande[i][1]) / 2
    bande[i][3] = 1e+6/(10 ** bande[i][1])
    bande[i][4] = 1e+6/(10 ** bande[i][0])
    bande[i][5] = 1e+6/((bande[i][3] + bande[i][4]) / 2)
    c = next(color)
    plt.axhspan(bande[i][3], bande[i][4],
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
plt.ylabel('Conductance in 10^(-6) S', fontsize="18", color=color)
plt.xlabel('# Pulses', fontsize="18", color=color)
plt.plot(b1_Pulse[:], b1_R[:], 'r.', markersize=dot_plot)
plt.plot(b2_Pulse[:], b2_R[:], 'b.', markersize=dot_plot)
plt.plot(b3_Pulse[:], b3_R[:], 'g.', markersize=dot_plot)
plt.plot(b4_Pulse[:], b4_R[:], 'k.', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color)
plt.yscale('log')
plt.ylim([1e+6/100000, 1e+6/500])
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
        b1_Pulse_set.append(i)
    ##Bande 1 RESET
    if bande[0][3] <= X0[i][1] <= bande[0][4]:
        b1_R_reset.append(X0[i][1])
        b1_Pulse_reset.append(i)
    ##Bande 2 SET
    elif bande[1][3] <= X1[i][1] <= bande[1][4]:
        b2_R_set.append(X1[i][1])
        b2_Pulse_set.append(i)
    ##Bande 2 RESET
    elif bande[1][3] <= X0[i][1] <= bande[1][4]:
        b2_R_reset.append(X0[i][1])
        b2_Pulse_reset.append(i)
    ##Bande 3 SET
    elif bande[2][3] <= X1[i][1] <= bande[2][4]:
        b3_R_set.append(X1[i][1])
        b3_Pulse_set.append(i)
    ##Bande 3 RESET
    elif bande[2][3] <= X0[i][1] <= bande[2][4]:
        b3_R_reset.append(X0[i][1])
        b3_Pulse_reset.append(i)
    ##Bande 4 SET
    elif bande[3][3] <= X1[i][1] <= bande[3][4]:
        b4_R_set.append(X1[i][1])
        b4_Pulse_set.append(i)
    ##Bande 4 RESET
    elif bande[3][3] <= X0[i][1] <= bande[3][4]:
        b4_R_reset.append(X0[i][1])
        b4_Pulse_reset.append(i)
### Données dans les bandes SET & RESET
plt.figure(3)
color = 'tab:red'
plt.ylabel('Conductance in 10^(-6) S',fontsize="18", color=color)
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
plt.ylim([1e+6/100000, 1e+6/500])
_a = 425
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées Ph1: SET & RESET',fontsize="14")
plt.savefig("20241016/Données traitées Ph1 SET & RESET.svg")



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
Pulse_av=5
for j in range(50):
    # SET bande 1
    for i in range(2, b1_set):
        if i < b1_s and abs(b1_Pulse_set[i] - b1_Pulse_set[i - 1]) <= Pulse_av and len(b1_Pulse_set)!= 1\
                and len(b1_Pulse_set)!= 0:
            b1_R_set[i - 1] = b1_R_set[i]
            b1_Pulse_set[i - 1] = b1_Pulse_set[i]
        b1_s = len(b1_Pulse_set)
    # RESET bande 1
    for i in range(2, b1_reset):
        if i < b1_re and abs(b1_Pulse_reset[i] - b1_Pulse_reset[i - 1]) <= Pulse_av and len(b1_R_reset)!= 1\
                and len(b1_R_reset)!= 0:
            b1_R_reset[i-1] = b1_R_reset[i]
            b1_Pulse_reset[i-1] = b1_Pulse_reset[i]
        b1_re = len(b1_R_reset)
        # SET bande 2
        for i in range(2, b2_set):
            if i < b2_s and abs(b2_Pulse_set[i] - b2_Pulse_set[i - 1]) <= Pulse_av and len(b2_Pulse_set) != 1 \
                    and len(b2_Pulse_set) != 0:
                b2_R_set[i - 1] = b2_R_set[i]
                b2_Pulse_set[i - 1] = b2_Pulse_set[i]
            b2_s = len(b2_Pulse_set)
        # RESET bande 2
        for i in range(2, b2_reset):
            if i < b2_re and abs(b2_Pulse_reset[i] - b2_Pulse_reset[i - 1]) <= Pulse_av and len(b2_R_reset) != 1 \
                    and len(b2_R_reset) != 0:
                b2_R_reset[i - 1] = b2_R_reset[i]
                b2_Pulse_reset[i - 1] = b2_Pulse_reset[i]
            b2_re = len(b2_R_reset)
        # SET bande 3
        for i in range(2, b3_set):
            if i < b3_s and abs(b3_Pulse_set[i] - b3_Pulse_set[i - 1]) <= Pulse_av and len(b3_Pulse_set) != 1 \
                    and len(b3_Pulse_set) != 0:
                b3_R_set[i - 1] = b3_R_set[i]
                b3_Pulse_set[i - 1] = b3_Pulse_set[i]
            b3_s = len(b3_R_set)
        # RESET bande 3
        for i in range(2, b3_reset):
            if i < b3_re and abs(b3_Pulse_reset[i] - b3_Pulse_reset[i - 1]) <= Pulse_av and len(b3_R_reset) != 1 \
                    and len(b3_R_reset) != 0:
                b3_R_reset[i - 1] = b3_R_reset[i]
                b3_Pulse_reset[i - 1] = b3_Pulse_reset[i]
            b3_re = len(b3_R_reset)
        # SET bande 4
        for i in range(2, b4_set):
            if i < b4_s and abs(b4_Pulse_set[i] - b4_Pulse_set[i - 1]) <= Pulse_av and len(b4_Pulse_set) != 1 \
                    and len(b4_Pulse_set) != 0:
                b4_R_set[i - 1] = b4_R_set[i]
                b4_Pulse_set[i - 1] = b4_Pulse_set[i]
            b4_s = len(b4_R_set)
        # RESET bande 4
        for i in range(2, b4_reset):
            if i < b4_re and abs(b4_Pulse_reset[i] - b4_Pulse_reset[i - 1]) <= Pulse_av and len(b4_R_reset) != 1 \
                    and len(b4_R_reset) != 0:
                b4_R_reset[i - 1] = b4_R_reset[i]
                b4_Pulse_reset[i - 1] = b4_Pulse_reset[i]
            b4_re = len(b4_R_reset)
###Eliminer les duplicata dans mes list
'''
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
'''
####Plot figure

plt.figure(4)
color = 'tab:red'
plt.ylabel('Conductance in 10^(-6) S',fontsize="18", color=color)
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
plt.ylim([1e+6/100000, 1e+6/500])
_a = 425
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées Ph2: SET & RESET',fontsize="14")
plt.savefig("20241016/Données traitées Ph2 SET & RESET.svg")

##Plot données SET
plt.figure(5)
color = 'tab:red'
plt.ylabel('Conductance in 10^(-6) S',fontsize="18", color=color)
plt.xlabel('# Pulses',fontsize="18", color=color)
plt.plot(b1_Pulse_set[:], b1_R_set[:], 'r.', markersize=dot_plot)
plt.plot(b2_Pulse_set[:], b2_R_set[:], 'b.', markersize=dot_plot)
plt.plot(b3_Pulse_set[:], b3_R_set[:], 'g.', markersize=dot_plot)
plt.plot(b4_Pulse_set[:], b4_R_set[:], 'k.', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color)
plt.tick_params(axis='x', labelcolor=color)
plt.yscale('log')
plt.ylim([1e+6/100000, 1e+6/500])
_a = 425
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées SET only',fontsize="14")
plt.savefig("20241016/Données traitées SET only.svg")

##Plot données RESET
plt.figure(6)
color = 'tab:red'
plt.ylabel('Conductance in 10^(-6) S',fontsize="18", color=color)
plt.xlabel('# Pulses',fontsize="18", color=color)
plt.plot(b1_Pulse_reset[:], b1_R_reset[:], 'r*', markersize=dot_plot)
plt.plot(b2_Pulse_reset[:], b2_R_reset[:], 'b*', markersize=dot_plot)
plt.plot(b3_Pulse_reset[:], b3_R_reset[:], 'g*', markersize=dot_plot)
plt.plot(b4_Pulse_reset[:], b4_R_reset[:], 'k*', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color)
plt.tick_params(axis='x', labelcolor=color)
plt.yscale('log')
plt.ylim([1e+6/100000, 1e+6/500])
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

plt.figure(7)  ## CDF SET & RESET
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
plt.xlim([1e+6/100000, 1e+6/500])
plt.xscale('log')
plt.xlabel("Conductance in 10^(-6) S",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig("20241016/CDF SET & RESET.svg")

plt.figure(8)  ## CDF SET ONLY
plt.title('CDF SET ONLY',fontsize="14")
plt.plot(x0_s, y0_s * 100, 'r.', markersize=dot_plot)
plt.plot(x1_s, y1_s * 100, 'b.', markersize=dot_plot)
plt.plot(x2_s, y2_s * 100, 'g.', markersize=dot_plot)
plt.plot(x3_s, y3_s * 100, 'k.', markersize=dot_plot)
# plt.ylim([0,3e-6])
plt.xlim([1e+6/100000, 1e+6/500])
plt.xscale('log')
plt.xlabel("Conductance in 10^(-6) S",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig("20241016/CDF SET ONLY.svg")

plt.figure(9)  ## CDF RESET ONLY
plt.title('CDF RESET ONLY',fontsize="14")
plt.plot(x0_re, y0_re * 100, 'r*', markersize=dot_plot)
plt.plot(x1_re, y1_re * 100, 'b*', markersize=dot_plot)
plt.plot(x2_re, y2_re * 100, 'g*', markersize=dot_plot)
plt.plot(x3_re, y3_re * 100, 'k*', markersize=dot_plot)
# plt.ylim([0,3e-6])
plt.xlim([1e+6/100000, 1e+6/500])
plt.xscale('log')
plt.xlabel("Conductance in 10^(-6) S",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig("20241016/CDF RESET ONLY.svg")

## Histogramme
plt.figure(10)
# bande1
plt.hist(b1_R_set, rwidth=0.9, color='k', label='SET')
plt.hist(b1_R_reset, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(round(bande[0][3],2)) + "," + str(round(bande[0][4],2)) + "]",fontsize="14")
plt.xlabel("Conductance in 10^(-6) S",fontsize="18")
plt.ylabel("Nombre de Pulse",fontsize="18")
plt.legend()
plt.savefig("20241016/hist-bande1.svg")
plt.figure(11)
# bande2
plt.hist(b2_R_set, rwidth=0.9, color='k', label='SET')
plt.hist(b2_R_reset, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(round(bande[1][3],2)) + "," + str(round(bande[1][4],2)) + "]",fontsize="14")
plt.xlabel("Conductance in 10^(-6) S",fontsize="18")
plt.ylabel("Nombre de Pulse",fontsize="18")
plt.legend()
plt.savefig("20241016/hist-bande2.svg")
plt.figure(12)
# bande3
plt.hist(b3_R_set, rwidth=0.9, color='k', label='SET')
plt.hist(b3_R_reset, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(round(bande[2][3],2)) + "," + str(round(bande[2][4],2)) + "]",fontsize="14")
plt.xlabel("Conductance in 10^(-6) S",fontsize="18")
plt.ylabel("Nombre de Pulse",fontsize="18")
plt.legend()
plt.savefig("20241016/hist-bande3.svg")
plt.figure(13)
# bande4
plt.hist(b4_R_set, rwidth=0.9, color='k', label='SET')
plt.hist(b4_R_reset, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(round(bande[3][3],2)) + "," + str(round(bande[3][4],2)) + "]",fontsize="14")
plt.xlabel("Conductance in 10^(-6) S",fontsize="18")
plt.ylabel("Nombre de Pulse",fontsize="18")
plt.legend()
plt.savefig("20241016/hist-bande4.svg")
## Gaussiennes Calculées et non à partir de données expé
d=100
plt.figure(14)
# bande1
gauss=[]
x_g=[]
x_g [:]= b1_R[:]
if x_g != []:
    me = mean(x_g[:])
    std = stdev(x_g[:])
    variance = np.square(std)
    g=len(x_g)
    for l in range(g):
        gauss.append(exp(-np.square(x_g[l]-me)*1e-6/2*variance)/(sqrt(2*pi*variance)))
    ##Fit pour avoir une courbe
    # calculate polynomial
    z = np.polyfit(x_g, gauss, d)
    f = np.poly1d(z)
    # calculate new x's and y's
    x_g_new = np.linspace(min(x_g[:]), max(x_g[:]), 50)
    gauss_new = f(x_g_new)
    plt.plot(x_g, gauss, 'r.',markersize=dot_plot)
    #plt.plot(x_g_new, gauss_new,'r')
# bande2
gauss=[]
x_g=[]
x_g [:]= b2_R[:]
if x_g != []:
    me = mean(x_g[:])
    std = stdev(x_g[:])
    variance = np.square(std)
    g=len(x_g)
    for l in range(g):
        gauss.append(exp(-np.square(x_g[l]-me)*1e-6/2*variance)/(sqrt(2*pi*variance)))
    ##Fit pour avoir une courbe
    # calculate polynomial
    z = np.polyfit(x_g, gauss, d)
    f = np.poly1d(z)
    # calculate new x's and y's
    x_g_new = np.linspace(min(x_g[:]), max(x_g[:]), 50)
    gauss_new = f(x_g_new)
    plt.plot(x_g, gauss, 'b.',markersize=dot_plot)
    #plt.plot( x_g_new, gauss_new,'b')
# bande3
gauss=[]
x_g=[]
x_g [:]= b3_R[:]
if x_g != []:
    me = mean(x_g[:])
    std = stdev(x_g[:])
    variance = np.square(std)
    g=len(x_g)
    for l in range(g):
        gauss.append(exp(-np.square(x_g[l]-me)*1e-6/2*variance)/(sqrt(2*pi*variance)))
    ##Fit pour avoir une courbe
    # calculate polynomial
    z = np.polyfit(x_g, gauss, d)
    f = np.poly1d(z)
    # calculate new x's and y's
    x_g_new = np.linspace(min(x_g[:]), max(x_g[:]), 50)
    gauss_new = f(x_g_new)
    plt.plot(x_g, gauss, 'g.',markersize=dot_plot)
#plt.plot(x_g_new, gauss_new,'g')
# bande4
gauss=[]
x_g=[]
x_g [:]= b4_R[:]
if x_g != []:
    me = mean(x_g[:])
    std = stdev(x_g[:])
    variance = np.square(std)
    g=len(x_g)
    for l in range(g):
        gauss.append(exp(-np.square(x_g[l]-me)*1e-6/2*variance)/(sqrt(2*pi*variance)))
    ##Fit pour avoir une courbe
    # calculate polynomial
    z = np.polyfit(x_g, gauss, d)
    f = np.poly1d(z)
    # calculate new x's and y's
    x_g_new = np.linspace(min(x_g[:]), max(x_g[:]), 50)
    gauss_new = f(x_g_new)
    plt.plot(x_g, gauss, 'k.',markersize=dot_plot)
    #plt.plot(x_g_new, gauss_new,'k')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Distribution de Gauss',fontsize="18")
    plt.xlabel('Conductance in 10^(-6) S ',fontsize="18")
    plt.title("Gaussienne ",fontsize="20")
    plt.savefig("20241016/Gauss-bande.svg")
plt.show()