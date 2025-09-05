# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:37:32 2024

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

dos= str(20250122)
dot_plot = 12  # taille point plot
labsize = 18 # Taille graduation axe
font = 20 # Taille titres des axes
fontLEG = 16 # taille legende
l=6 #largeur fig
L=7 #longueur fig
name = input("Nom fichier :")
rp = float(input('Résistance pristine: '))
#ep = float(input('Epaisseur en nm: '))
ep=55
####Chargement fichier de données
cp = float(1/rp*1e+3)
Cp = []
cth = []
ep = 55
data = np.loadtxt(""+dos+"/"+name+".NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1]
c4 = ((1 / data[:, 3]) / (cp)) * 1e+6 # Conductance normalisée
c9 = data[:, 8]
c10 = data[:, 9]
c12 = data[:, 11]
c11 = data[:, 10]
t = int(len(c4))
# Garder un même nombre de pulse pour la comparaison.
#t = 597
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
r_th = (35 * ep * 1E-7) / (pi * 0.6 * 1e-4 * 0.6 * 1e-4)  # La résistivité du 30% Cr est 35 Ohm.cm
c_th = (1 / (r_th)) * 1e+6
for p in range(t):
    Rp.append(float(rp * 1e+3))  # Mettre la valeur de
    Cp.append(float(1 / (rp * 1e+3)) / float(1 / (rp * 1e+3)))  # Cp normalisée
    # R th dans la list
    Rth.append(float(r_th))
    cth.append((float(1 / r_th)) / float(1 / (rp * 1e+3)))

# plt.grid()
# Plot pour la résistance
plt.figure(1, figsize= (L,l))
color = 'tab:red'
plt.ylabel('C/Cp', color=color, fontsize=font)
plt.xlabel('# Pulses', color=color, fontsize=font)
plt.plot(X0[:, 0], X0[:, 1], 'r.', X1[:, 0], X1[:, 1], 'k.', markersize=dot_plot)
plt.plot(X1[:, 0], Cp[:], 'c-', label='Cp= ' + str(f"{(cp):.2f}") + ' uS', markersize=dot_plot)  # Plot Cp
plt.plot(X1[:, 0], cth[:], 'b-', label='Cth=' + str(f"{(c_th/cp):.2f}") + ' Cp', markersize=dot_plot)  # Plot C th
plt.tick_params(axis='y', labelcolor=color, labelsize= labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize= labsize)
plt.yscale('log')
plt.ylim([0.8, 300])
_a = 0;
_b = 630  # Limits of x axis
#plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données brutes', fontsize=font)
plt.legend(loc='upper right', fontsize=fontLEG)
plt.savefig(""+dos+"/Données brutes.svg")

####Allocation bande de conductance
### SET ## Confi 1: 900 - 1050 et gap=0.3  ## Confi 2: 800 - 1550 et gap=0.07
### RESET ## Confi 1: 900 - 1200 et gap=0.28  ## Confi 2: 800 - 1800 et gap=0.09
# Calcul largeur et espacement des bande
nb_bande = 4
bande = np.zeros((nb_bande + 1, 6))
gap_log = 0.15
min_bande1 = np.log10(900)
max_bande1 = np.log10(2000)
larg_range = max_bande1 - min_bande1
bande[0][0] = min_bande1
bande[0][1] = max_bande1
color = iter(cm.rainbow(np.linspace(0, 1, nb_bande)))
for i in range(nb_bande):
    bande[i + 1][0] = bande[i][1] + gap_log
    bande[i + 1][1] = bande[i + 1][0] + larg_range
    bande[i][2] = (bande[i][0] + bande[i][1]) / 2
    bande[i][3] = (1e+6/(10 ** bande[i][1]))/cp
    bande[i][4] = (1e+6/(10 ** bande[i][0]))/cp
    bande[i][5] = (bande[i][3] + bande[i][4]) / 2
    c = next(color)
    plt.axhspan(bande[i][3], bande[i][4],
                xmin=0, xmax=t,
                color=c, alpha=0.5)
DF = pd.DataFrame(bande)
DF.to_csv(""+dos+"/bande.csv")
plt.savefig(""+dos+"/Data brutes + bandes.svg")
##########Traitement phase 1: Récuperation de tout les points qui tombent dans les bandes

b1_R = []
b1_v = []
b1_t = []
b1_Pulse = []
b2_R = []
b2_v = []
b2_t = []
b2_Pulse = []
b3_R = []
b3_v = []
b3_t = []
b3_Pulse = []
b4_R = []
b4_v = []
b4_t = []
b4_Pulse = []

for i in range(t):
    if bande[0][3] <= c4[i] <= bande[0][4]:
        b1_R.append(c4[i])
        b1_v.append(c12[i])
        b1_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b1_Pulse.append(i + 1)
        continue

    elif bande[1][3] <= c4[i] <= bande[1][4]:
        b2_R.append(c4[i])
        b2_v.append(c12[i])
        b2_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b2_Pulse.append(i + 1)

    elif bande[2][3] <= c4[i] <= bande[2][4]:
        b3_R.append(c4[i])
        b3_v.append(c12[i])
        b3_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b3_Pulse.append(i + 1)

    elif bande[3][3] <= c4[i] <= bande[3][4]:
        b4_R.append(c4[i])
        b4_v.append(c12[i])
        b4_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b4_Pulse.append(i + 1)
#Visualisation
plt.figure(figsize= (L,l))
color = 'tab:red'
plt.ylabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) uS", fontsize=font, color=color)
plt.xlabel('# Pulses', fontsize=font, color=color)
plt.plot(b1_Pulse[:], b1_R[:], 'r.', markersize=dot_plot)
plt.plot(b2_Pulse[:], b2_R[:], 'b.', markersize=dot_plot)
plt.plot(b3_Pulse[:], b3_R[:], 'g.', markersize=dot_plot)
plt.plot(b4_Pulse[:], b4_R[:], 'k.', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color, labelsize= labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize= labsize)
plt.yscale('log')
plt.ylim([0.8, 300])
_a = 0
_b = 630  # Limits of x axis
#plt.xlim([_a, _b])
#plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées Phase 1', fontsize=font)
plt.savefig(""+dos+"/Données traitées Phase 1.svg")

###traitment phase 2
b1_R_set = []
b1_v_set = []
b1_t_set = []
b1_R_reset = []
b1_v_reset = []
b1_t_reset = []
b1_Pulse_set = []
b1_Pulse_reset = []
b2_R_set = []
b2_v_set = []
b2_t_set = []
b2_R_reset = []
b2_v_reset = []
b2_t_reset = []
b2_Pulse_set = []
b2_Pulse_reset = []
b3_R_set = []
b3_v_set = []
b3_t_set = []
b3_R_reset = []
b3_v_reset = []
b3_t_reset = []
b3_Pulse_set = []
b3_Pulse_reset = []
b4_R_set = []
b4_v_set = []
b4_t_set = []
b4_R_reset = []
b4_v_reset = []
b4_t_reset = []
b4_Pulse_set = []
b4_Pulse_reset = []

##Distinction entre SET et RESET
for i in range(t):
    ##Bande 1 SET
    if bande[0][3] <= X1[i][1] <= bande[0][4]:
        b1_R_set.append(X1[i][1])
        b1_Pulse_set.append(X1[i][0])
    ##Bande 1 RESET
    if bande[0][3] <= X0[i][1] <= bande[0][4]:
        b1_R_reset.append(X0[i][1])
        b1_Pulse_reset.append(X0[i][0])
    ##Bande 2 SET
    elif bande[1][3] <= X1[i][1] <= bande[1][4]:
        b2_R_set.append(X1[i][1])
        b2_Pulse_set.append(X1[i][0])
    ##Bande 2 RESET
    elif bande[1][3] <= X0[i][1] <= bande[1][4]:
        b2_R_reset.append(X0[i][1])
        b2_Pulse_reset.append(X0[i][0])
    ##Bande 3 SET
    elif bande[2][3] <= X1[i][1] <= bande[2][4]:
        b3_R_set.append(X1[i][1])
        b3_Pulse_set.append(X1[i][0])
    ##Bande 3 RESET
    elif bande[2][3] <= X0[i][1] <= bande[2][4]:
        b3_R_reset.append(X0[i][1])
        b3_Pulse_reset.append(X0[i][0])
    ##Bande 4 SET
    elif bande[3][3] <= X1[i][1] <= bande[3][4]:
        b4_R_set.append(X1[i][1])
        b4_Pulse_set.append(X1[i][0])
    ##Bande 4 RESET
    elif bande[3][3] <= X0[i][1] <= bande[3][4]:
        b4_R_reset.append(X0[i][1])
        b4_Pulse_reset.append(X0[i][0])
### Données dans les bandes SET & RESET: Visualisation + Data brute
plt.figure(figsize= (L,l))
color = 'tab:red'
plt.ylabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) uS",fontsize=font, color=color)
plt.xlabel('# Pulses',fontsize=font, color=color)
plt.plot(b1_Pulse_set[:], b1_R_set[:], 'g1', markersize=dot_plot)
plt.plot(b1_Pulse_reset[:], b1_R_reset[:], 'g1', markersize=dot_plot)
plt.plot(b2_Pulse_set[:], b2_R_set[:], 'g1', markersize=dot_plot)
plt.plot(b2_Pulse_reset[:], b2_R_reset[:], 'g1', markersize=dot_plot)
plt.plot(b3_Pulse_set[:], b3_R_set[:], 'g1', markersize=dot_plot)
plt.plot(b3_Pulse_reset[:], b3_R_reset[:], 'g1', markersize=dot_plot)
plt.plot(b4_Pulse_set[:], b4_R_set[:], 'g1', markersize=dot_plot)
plt.plot(b4_Pulse_reset[:], b4_R_reset[:], 'g1', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color, labelsize= labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize= labsize)
plt.yscale('log')
plt.ylim([0.8, 300])
_a = 0
_b = 630  # Limits of x axis
#plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données brutes et point dans les bandes',fontsize=font)
plt.savefig(""+dos+"/Données brutes et point dans les bandes.svg")

### Données dans les bandes SET & RESET: Visualisation
plt.figure(figsize= (L,l))
color = 'tab:red'
plt.ylabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) uS",fontsize=font, color=color)
plt.xlabel('# Pulses',fontsize=font, color=color)
plt.plot(b1_Pulse_set[:], b1_R_set[:], 'r.', markersize=dot_plot)
plt.plot(b1_Pulse_reset[:], b1_R_reset[:], 'r*', markersize=dot_plot)
plt.plot(b2_Pulse_set[:], b2_R_set[:], 'b.', markersize=dot_plot)
plt.plot(b2_Pulse_reset[:], b2_R_reset[:], 'b*', markersize=dot_plot)
plt.plot(b3_Pulse_set[:], b3_R_set[:], 'g.', markersize=dot_plot)
plt.plot(b3_Pulse_reset[:], b3_R_reset[:], 'g*', markersize=dot_plot)
plt.plot(b4_Pulse_set[:], b4_R_set[:], 'k.', markersize=dot_plot)
plt.plot(b4_Pulse_reset[:], b4_R_reset[:], 'k*', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color, labelsize= labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize= labsize)
plt.yscale('log')
plt.ylim([0.8, 300])
_a = 0
_b = 630  # Limits of x axis
#plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées Ph1: SET & RESET',fontsize=font)
plt.savefig(""+dos+"/Données traitées Ph1 SET & RESET.svg")



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

a_s = []; a_re = []
b_s = []; b_re = []
c_s = []; c_re = []
d_s = []; d_re = []

def nb_p1cy ():
    pluse = []
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
            for tyy in range(len(a_s)):
                pluse.append(abs(b_s[tyy] - a_s[tyy]) + abs(b_re[tyy] - a_re[tyy]))
    Pulse_av = min(pluse) - 1
    return Pulse_av

Pulse_av = nb_p1cy ()# Comment déterminer ce paramètre ?

## Traitement pour récuperer Le premier point à tomber dans le bande voulu
## et ainsi éviter que pour le même SET ou RESET on récupère plus d'un point
## L'idée c'est d'arrêter le programme dès que l'on a atteint la bande cible
## Pour ça, je regarde les numéros de pulse: Si deux conductance dans un même
## niveau ont pas un écart de numéro de pulse inférieur à Pulse_av (fixé),
## je ne conserve que la première valeur à être tombée dans la bande cible.

for interation in range(50):

    # SET bande 4
    for i in range(0, b4_set):
        if abs(b4_Pulse_set[i] - b4_Pulse_set[i - 1]) <= Pulse_av and len(b4_Pulse_set) != 1 \
                and len(b4_Pulse_set) != 0:
            b4_R_set[i] = b4_R_set[i - 1]

            b4_Pulse_set[i] = b4_Pulse_set[i - 1]
        b4_s = len(b4_Pulse_set)
    # SET bande 3
    for i in range(0, b3_set):
        if abs(b3_Pulse_set[i] - b3_Pulse_set[i - 1]) <= Pulse_av and len(b3_Pulse_set) != 1 \
                and len(b3_Pulse_set) != 0:
            b3_R_set[i] = b3_R_set[i - 1]

            b3_Pulse_set[i] = b3_Pulse_set[i - 1]
        b3_s = len(b3_Pulse_set)
    # SET bande 2
    for i in range(0, b2_set):
        if abs(b2_Pulse_set[i] - b2_Pulse_set[i - 1]) <= Pulse_av and len(b2_Pulse_set) != 1 \
                and len(b2_Pulse_set) != 0:
            b2_R_set[i] = b2_R_set[i - 1]

            b2_Pulse_set[i] = b2_Pulse_set[i - 1]
        b2_s = len(b2_Pulse_set)
    # SET bande 1
    for i in range(0, b1_set):
        if abs(b1_Pulse_set[i] - b1_Pulse_set[i - 1]) <= Pulse_av and len(b1_Pulse_set) != 1 \
                and len(b1_Pulse_set) != 0:
            b1_R_set[i] = b1_R_set[i - 1]

            b1_Pulse_set[i] = b1_Pulse_set[i - 1]
        b1_s = len(b1_Pulse_set)
    # RESET bande 1
    for i in range(0, b1_reset):
        if abs(b1_Pulse_reset[i] - b1_Pulse_reset[i - 1]) <= Pulse_av and len(b1_R_reset)!= 1\
                and len(b1_R_reset)!= 0:
            b1_R_reset[i] = b1_R_reset[i-1]

            b1_Pulse_reset[i] = b1_Pulse_reset[i-1]
        b1_re = len(b1_Pulse_reset)

    # RESET bande 2
    for i in range(0, b2_reset):
        if abs(b2_Pulse_reset[i] - b2_Pulse_reset[i - 1]) <= Pulse_av and len(b2_R_reset) != 1 \
                and len(b2_R_reset) != 0:
            b2_R_reset[i] = b2_R_reset[i - 1]

            b2_Pulse_reset[i] = b2_Pulse_reset[i - 1]
        b2_re = len(b2_Pulse_reset)

    # RESET bande 3
    for i in range(0, b3_reset):
        if abs(b3_Pulse_reset[i] - b3_Pulse_reset[i - 1]) <= Pulse_av and len(b3_R_reset) != 1 \
                and len(b3_R_reset) != 0:
            b3_R_reset[i] = b3_R_reset[i - 1]

            b3_Pulse_reset[i] = b3_Pulse_reset[i - 1]
        b3_re = len(b3_Pulse_reset)
    # RESET bande 4
    for i in range(0, b4_reset):
        if abs(b4_Pulse_reset[i] - b4_Pulse_reset[i - 1]) <= Pulse_av and len(b4_R_reset) != 1 \
                and len(b4_R_reset) != 0:
            b4_R_reset[i] = b4_R_reset[i - 1]
            b4_Pulse_reset[i] = b4_Pulse_reset[i - 1]
        b4_re = len(b4_Pulse_reset)


### Après ce traitement j'obtiens des list avec des élements qui se repétent. J'élimine les repétitions
###Eliminer les duplicata dans mes list
##Initialisation list
b1_R_set_f =[];b1_R_reset_f =[];b1_v_set_f =[];b1_v_reset_f =[];b1_t_set_f =[];b1_t_reset_f =[]
b2_R_set_f =[];b2_R_reset_f =[];b2_v_set_f =[];b2_v_reset_f =[];b2_t_set_f =[];b2_t_reset_f =[]
b3_R_set_f =[];b3_R_reset_f =[];b3_v_set_f =[];b3_v_reset_f =[];b3_t_set_f =[];b3_t_reset_f =[]
b4_R_set_f =[];b4_R_reset_f =[];b4_v_set_f =[];b4_v_reset_f =[];b4_t_set_f =[];b4_t_reset_f =[]
b1_Pulse_set_f =[];b1_Pulse_reset_f =[]
b2_Pulse_set_f =[];b2_Pulse_reset_f =[]
b3_Pulse_set_f =[];b3_Pulse_reset_f =[]
b4_Pulse_set_f =[];b4_Pulse_reset_f =[]
## Elimination des repétitions
#bande1
for w in b1_Pulse_set: #Set
    if w not in b1_Pulse_set_f:
        b1_Pulse_set_f.append(w)
        ind=b1_Pulse_set.index(w)
        b1_R_set_f.append(b1_R_set[ind])
        b1_v_set_f.append(b1_v[b1_Pulse.index(w)])
        b1_t_set_f.append(b1_t[b1_Pulse.index(w)])
for w in b1_Pulse_reset: #Reset
    if w not in b1_Pulse_reset_f:
        b1_Pulse_reset_f.append(w)
        ind=b1_Pulse_reset.index(w)
        b1_R_reset_f.append(b1_R_reset[ind])
        b1_v_reset_f.append(b1_v[b1_Pulse.index(w)])
        b1_t_reset_f.append(b1_t[b1_Pulse.index(w)])
#bande2
for w in b2_Pulse_set:#Set
    if w not in b2_Pulse_set_f:
        b2_Pulse_set_f.append(w)
        ind=b2_Pulse_set.index(w)
        b2_R_set_f.append(b2_R_set[ind])
        b2_v_set_f.append(b2_v[b2_Pulse.index(w)])
        b2_t_set_f.append(b2_t[b2_Pulse.index(w)])
for w in b2_Pulse_reset:#Reset
    if w not in b2_Pulse_reset_f:
        b2_Pulse_reset_f.append(w)
        ind=b2_Pulse_reset.index(w)
        b2_R_reset_f.append(b2_R_reset[ind])
        b2_v_reset_f.append(b2_v[b2_Pulse.index(w)])
        b2_t_reset_f.append(b2_t[b2_Pulse.index(w)])
#bande3
for w in b3_Pulse_set:#Set
    if w not in b3_Pulse_set_f:
        b3_Pulse_set_f.append(w)
        ind=b3_Pulse_set.index(w)
        b3_R_set_f.append(b3_R_set[ind])
        b3_v_set_f.append(b3_v[b3_Pulse.index(w)])
        b3_t_set_f.append(b3_t[b3_Pulse.index(w)])
for w in b3_Pulse_reset:#Reset
    if w not in b3_Pulse_reset_f:
        b3_Pulse_reset_f.append(w)
        ind=b3_Pulse_reset.index(w)
        b3_R_reset_f.append(b3_R_reset[ind])
        b3_v_reset_f.append(b3_v[b3_Pulse.index(w)])
        b3_t_reset_f.append(b3_t[b3_Pulse.index(w)])
#bande4
for w in b4_Pulse_set:#Set
    if w not in b4_Pulse_set_f:
        b4_Pulse_set_f.append(w)
        ind=b4_Pulse_set.index(w)
        b4_R_set_f.append(b4_R_set[ind])
        b4_v_set_f.append(b4_v[b4_Pulse.index(w)])
        b4_t_set_f.append(b4_t[b4_Pulse.index(w)])
for w in b4_Pulse_reset:#Reset
    if w not in b4_Pulse_reset_f:
        b4_Pulse_reset_f.append(w)
        ind=b4_Pulse_reset.index(w)
        b4_R_reset_f.append(b4_R_reset[ind])
        b4_v_reset_f.append(b4_v[b4_Pulse.index(w)])
        b4_t_reset_f.append(b4_t[b4_Pulse.index(w)])

###Statistiques sur les tensions
###Ecart-type & Médiane & Moyenne:
#Je fais la disticntion pour les niveaux qui ont zéro ou un seul point.

## Bande 1 SET
if len(b1_v_set_f)>1 :
    sd_s_1 = stdev(b1_v_set_f)
    md_s_1 = median(b1_v_set_f)
    m_s_1 = mean(b1_v_set_f)
elif len(b1_v_set_f)==0:
    sd_s_1 = 0.001
    m_s_1 = 0.001
    md_s_1 = 0.001
elif len(b1_v_set_f)==1:
    sd_s_1 = 0.001
    m_s_1 = b1_v_set_f[0]
    md_s_1 = b1_v_set_f[0]
## Bande 1 RESET
if len(b1_v_reset_f)>1 :
    m_re_1 = mean(b1_v_reset_f)
    sd_re_1 = stdev(b1_v_reset_f)
    md_re_1 = median(b1_v_reset_f)
elif len(b1_v_reset_f)==0:
    sd_re_1 = 0.001
    m_re_1 = 0.001
    md_re_1 = 0.001
elif len(b1_v_reset_f)==1:
    sd_re_1 = 0.001
    m_re_1 = b1_v_reset_f[0]
    md_re_1 = b1_v_reset_f[0]
## Bande 2 SET
if len(b2_v_set_f) > 1 :
    sd_s_2= stdev(b2_v_set_f)
    md_s_2 = median(b2_v_set_f)
    m_s_2 = mean(b2_v_set_f)
elif len(b2_v_set_f)==0 :
    sd_s_2 = 0.001
    m_s_2 = 0.001
    md_s_2 = 0.001
elif len(b2_v_set_f)==1 :
    sd_s_2 = 0.001
    m_s_2 = b2_v_set_f[0]
    md_s_2 = b2_v_set_f[0]
## Bande 2 RESET
if len(b2_v_reset_f) > 1 :
    m_re_2 = mean(b2_v_reset_f)
    sd_re_2 = stdev(b2_v_reset_f)
    md_re_2 = median(b2_v_reset_f)
elif len(b2_v_reset_f) == 0:
    sd_re_2 = 0.001
    m_re_2= 0.001
    md_re_2 = 0.001
elif len(b2_v_reset_f) == 1:
    sd_re_2 = 0.001
    m_re_2= b2_v_reset_f[0]
    md_re_2 = b2_v_reset_f[0]
## Bande 3 SET
if len(b3_v_set_f) > 1 :
    sd_s_3 = stdev(b3_v_set_f)
    md_s_3 = median(b3_v_set_f)
    m_s_3 = mean(b3_v_set_f)
elif len(b3_v_set_f) == 0 :
    sd_s_3 = 0.001
    m_s_3 = 0.001
    md_s_3 = 0.001
elif len(b3_v_set_f) == 1:
    sd_s_3 = 0.001
    m_s_3 = b3_v_set_f[0]
    md_s_3 = b3_v_set_f[0]
## Bande 3 RESET
if len(b3_v_reset_f) > 1 :
    m_re_3 = mean(b3_v_reset_f)
    sd_re_3 = stdev(b3_v_reset_f)
    md_re_3 = median(b3_v_reset_f)
elif len(b3_v_reset_f) == 0 :
    sd_re_3 = 0.001
    m_re_3 = 0.001
    md_re_3 = 0.001
elif len(b3_v_reset_f) == 1 :
    sd_re_3 = 0.001
    m_re_3 = b3_v_reset_f[0]
    md_re_3 = b3_v_reset_f[0]
## Bande 4 SET
if len(b4_v_set_f) > 1 :
    sd_s_4 = stdev(b4_v_set_f)
    md_s_4 = median(b4_v_set_f)
    m_s_4 = mean(b4_v_set_f)
elif len(b4_v_set_f) == 0:
    sd_s_4 = 0.001
    m_s_4 = 0.001
    md_s_4 = 0.001
elif len(b4_v_set_f) == 1:
    sd_s_4 = 0.001
    m_s_4 = b4_v_set_f[0]
    md_s_4 = b4_v_set_f[0]
## Bande 4 RESET
if len(b4_v_reset_f) > 1 :
    m_re_4 = mean(b4_v_reset_f)
    sd_re_4 = stdev(b4_v_reset_f)
    md_re_4 = median(b4_v_reset_f)
elif len(b4_v_reset_f) == 0 :
    sd_re_4 = 0.001
    m_re_4 = 0.001
    md_re_4 = 0.001
elif len(b4_v_reset_f) == 1 :
    sd_re_4 = 0.001
    m_re_4 = b4_v_reset_f[0]
    md_re_4 = b4_v_reset_f[0]

#### Visualisation Plot figure des points sélectionnée pour l'analyse finale
plt.figure(1,figsize= (L,l))
color = 'tab:red'
plt.ylabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) uS",fontsize=font, color=color)
plt.xlabel('# Pulses',fontsize=font, color=color)
plt.plot(b1_Pulse_set_f[:], b1_R_set_f[:], 'b1', markersize=dot_plot + 6)
plt.plot(b1_Pulse_reset_f[:], b1_R_reset_f[:], 'g2', markersize=dot_plot + 6)
plt.plot(b2_Pulse_set_f[:], b2_R_set_f[:], 'b1', markersize=dot_plot + 6)
plt.plot(b2_Pulse_reset_f[:], b2_R_reset_f[:], 'g2', markersize=dot_plot + 6)
plt.plot(b3_Pulse_set_f[:], b3_R_set_f[:], 'b1', markersize=dot_plot + 6)
plt.plot(b3_Pulse_reset_f[:], b3_R_reset_f[:], 'g2', markersize=dot_plot + 6)
plt.plot(b4_Pulse_set_f[:], b4_R_set_f[:], 'b1', markersize=dot_plot + 6)
plt.plot(b4_Pulse_reset_f[:], b4_R_reset_f[:], 'g2', markersize=dot_plot + 6)
plt.tick_params(axis='y', labelcolor=color, labelsize=labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize=labsize)
plt.yscale('log')
plt.ylim([0.8, 300])
#plt.xticks([])
#ax1.legend(loc='upper right', fontsize="12")
_a = 0
_b = t # Limits of x axis
#ax1.set_xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données Sélectionnées pour analyse',fontsize=font)
plt.savefig(""+dos+"/Data brute et Points selectionnées.svg")

#### Visualisation Plot figure
#fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 11), height_ratios=[1, 1])
plt.figure(figsize= (L,l))
color = 'tab:red'
plt.ylabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) uS",fontsize=font, color=color)
plt.xlabel('# Pulses',fontsize=font, color=color)
plt.plot(b1_Pulse_set_f[:], b1_R_set_f[:], 'r.', markersize=dot_plot)
plt.plot(b1_Pulse_reset_f[:], b1_R_reset_f[:], 'r*', markersize=dot_plot)
plt.plot(b2_Pulse_set_f[:], b2_R_set_f[:], 'b.', markersize=dot_plot)
plt.plot(b2_Pulse_reset_f[:], b2_R_reset_f[:], 'b*', markersize=dot_plot)
plt.plot(b3_Pulse_set_f[:], b3_R_set_f[:], 'g.', markersize=dot_plot)
plt.plot(b3_Pulse_reset_f[:], b3_R_reset_f[:], 'g*', markersize=dot_plot)
plt.plot(b4_Pulse_set_f[:], b4_R_set_f[:], 'k.', markersize=dot_plot)
plt.plot(b4_Pulse_reset_f[:], b4_R_reset_f[:], 'k*', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color, labelsize=labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize=labsize)
plt.yscale('log')
plt.ylim([0.8, 300])
#plt.xticks([])
#ax1.legend(loc='upper right', fontsize="12")
_a = 0
_b = t # Limits of x axis
#ax1.set_xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées Ph2: SET & RESET',fontsize=font)
plt.savefig(""+dos+"/Data traitées Ph2 SET & RESET.svg")


# Plot pour la tension
plt.figure(figsize= (L,l))
color = 'tab:blue'
plt.ylabel("Voltage(V)", fontsize=font, color=color)
plt.xlabel('# Pulses',fontsize=font, color=color)
plt.plot(b1_Pulse_set_f[:], b1_v_set_f[:], 'r.',label='V_s1: Med='+str(f"{(md_s_1):.2f}")+' My='+str(f"{(m_s_1):.2f}")+' sd='+str(f"{(sd_s_1):.2f}")+' ('+str(f"{(sd_s_1*100/m_s_1):.2f}")+'%)', markersize=dot_plot)
plt.plot(b1_Pulse_reset_f[:], b1_v_reset_f[:], 'r*',label='V_re1: Med='+str(f"{(md_re_1):.2f}")+' My='+str(f"{(m_re_1):.2f}")+' sd='+str(f"{(sd_re_1):.2f}")+' ('+str(f"{(sd_re_1*100/m_re_1):.2f}")+'%)', markersize=dot_plot)
plt.plot(b2_Pulse_set_f[:], b2_v_set_f[:], 'b.',label='V_s2: Med='+str(f"{(md_s_2):.2f}")+' My='+str(f"{(m_s_2):.2f}")+' sd='+str(f"{(sd_s_2):.2f}")+' ('+str(f"{(sd_s_2*100/m_s_2):.2f}")+'%)', markersize=dot_plot)
plt.plot(b2_Pulse_reset_f[:], b2_v_reset_f[:], 'b*',label='V_re2: Med='+str(f"{(md_re_2):.2f}")+' My='+str(f"{(m_re_2):.2f}")+' sd='+str(f"{(sd_re_2):.2f}")+' ('+str(f"{(sd_re_2*100/m_re_2):.2f}")+'%)', markersize=dot_plot)
plt.plot(b3_Pulse_set_f[:], b3_v_set_f[:], 'g.',label='V_s3: Med='+str(f"{(md_s_3):.2f}")+' My='+str(f"{(m_s_3):.2f}")+' sd='+str(f"{(sd_s_3):.2f}")+' ('+str(f"{(sd_s_3*100/m_s_3):.2f}")+'%)', markersize=dot_plot)
plt.plot(b3_Pulse_reset_f[:], b3_v_reset_f[:], 'g*',label='V_re3: Med='+str(f"{(md_re_3):.2f}")+' My='+str(f"{(m_re_3):.2f}")+' sd='+str(f"{(sd_re_3):.2f}")+' ('+str(f"{(sd_re_3*100/m_re_3):.2f}")+'%)', markersize=dot_plot)
plt.plot(b4_Pulse_set_f[:], b4_v_set_f[:], 'k.',label='V_s4: Med='+str(f"{(md_s_4):.2f}")+' My='+str(f"{(m_s_4):.2f}")+' sd='+str(f"{(sd_s_4):.2f}")+' ('+str(f"{(sd_s_4*100/m_s_4):.2f}")+'%)', markersize=dot_plot)
plt.plot(b4_Pulse_reset_f[:], b4_v_reset_f[:], 'k*',label='V_re4: Med='+str(f"{(md_re_4):.2f}")+' My='+str(f"{(m_re_4):.2f}")+' sd='+str(f"{(sd_re_4):.2f}")+' ('+str(f"{(sd_re_4*100/m_re_4):.2f}")+'%)', markersize=dot_plot)
plt.yticks([2, 4, 6, 8, 10])
plt.ylim([1, 10])
plt.legend(fontsize="12")
#plt.xlim([_a,_b])
plt.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
plt.tick_params(axis='y', labelcolor=color, labelsize=labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize=labsize)
#Plot du temps
'''
ax3.set_ylabel('Time(ns)', fontsize="18", color=color)
ax3.plot(b1_Pulse_set_f[:], b1_t_set_f[:], 'r.', markersize=dot_plot-7)
ax3.plot(b1_Pulse_reset_f[:], b1_v_reset_f[:], 'r*', markersize=dot_plot-7)
ax3.plot(b2_Pulse_set_f[:], b2_t_set_f[:], 'b.', markersize=dot_plot-7)
ax3.plot(b2_Pulse_reset_f[:], b2_t_reset_f[:], 'b*', markersize=dot_plot-7)
ax3.plot(b3_Pulse_set_f[:], b3_t_set_f[:], 'g.', markersize=dot_plot-7)
ax3.plot(b3_Pulse_reset_f[:], b3_t_reset_f[:], 'g*', markersize=dot_plot-7)
ax3.plot(b4_Pulse_set_f[:], b4_t_set_f[:], 'k.', markersize=dot_plot-7)
ax3.plot(b4_Pulse_reset_f[:], b4_v_reset_f[:], 'k*', markersize=dot_plot-7)
ax3.set_yscale('linear')
ax3.set_yticks([50, 100, 150, 200, 250])
ax3.set_ylim([10, 250])
# ax3.set_xlim([_a,_b])
ax3.set_xlabel('Number of pulse', fontsize="20")
ax3.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax3.tick_params(axis='y', labelcolor=color, labelsize=15)
ax3.tick_params(axis='x', labelsize=15)
'''
#fig.tight_layout()
plt.savefig(""+dos+"/Tension Ph2 SET & RESET.svg")

##Plot des tensions moyenne pour chaque niveau de conduction
## Bande 1 = Max conductance
## Bande 4 = Min conductance
x_tb = np.linspace(1,4,4)
tb1s = [m_s_1, m_s_2, m_s_3, m_s_4]
sd_s = [sd_s_1, sd_s_2, sd_s_3, sd_s_4]
tb1re = [m_re_1, m_re_2, m_re_3, m_re_4]
sd_re = [sd_re_1, sd_re_2, sd_re_3, sd_re_4]
'''
plt.figure(figsize= (L,l))
plt.plot(tb1s, x_tb , 'ks', label='M SET / Niv.')
plt.plot(tb1re, x_tb , 'rs', label='M RESET / Niv.')
plt.xlabel("Tension Moyenne / Ni. ",fontsize=font)
plt.ylabel('Niveau Conductance',fontsize=font)
plt.xlim([2, 6])
plt.yticks([1,2,3,4])
plt.xticks(np.arange(2,6, step=1))
plt.grid(color='k', axis='x')
plt.title(name,fontsize=font)
plt.tick_params(axis='y', labelcolor=color, labelsize=labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize=labsize)
plt.savefig(""+dos+"/Tension des niveaux.svg")
'''
plt.figure(figsize= (L,l))
t1="[U_SET]="+str(f"{(m_s_1):.2f}")+" "; t2="[U_RESET]= "
color = iter(cm.rainbow(np.linspace(0, 1, nb_bande)))
for i in range(nb_bande):
    c = next(color)
    plt.axhspan(bande[i][3], bande[i][4],
                xmin=0, xmax=t,
                color='tab:orange', alpha=0.45)
color = 'tab:red'
plt.ylabel('G/Gp', color=color, fontsize=font)
plt.xlabel("Tension en V",fontsize=font)
for ae in range(4):

    plt.errorbar(tb1s[ae], bande[ae][5], xerr=sd_s[ae], fmt='k.',
                 ms=dot_plot, ecolor="k", elinewidth=2, capsize=6, capthick=2)
    #plt.plot(tb1s, bande[0:4, 5] , 'k.', label='M SET / Niv.',markersize=dot_plot)
    plt.errorbar(tb1re[ae], bande[ae][5], xerr=sd_re[ae], fmt='r*',
                 ms=dot_plot, ecolor="r", elinewidth=2, capsize=6, capthick=2)
    #plt.plot(tb1re, bande[0:4, 5] , 'r*', label='M RESET / Niv.',markersize=dot_plot)
plt.plot(X1[:, 0], Cp[:], 'c-', label='Gp= ' + str(f"{(cp):.2f}") + ' uS', markersize=dot_plot)  # Plot Cp
plt.plot(X1[:, 0], cth[:], 'b-', label='Gth=' + str(f"{(c_th/cp):.2f}") + ' Cp', markersize=dot_plot)  # Plot C th
plt.tick_params(axis='y', labelcolor=color, labelsize= labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize= labsize)
plt.yscale('log')
plt.ylim([0.8, 300])
plt.xlim([1, 8])
plt.xticks(np.arange(1,8, step=1))
plt.grid(color='k', axis='y')
plt.title('Moy. tension SET & RESET / Ni.', fontsize=font)
plt.legend(loc='upper right', fontsize=fontLEG)
plt.savefig(""+dos+"/Tension traitées.svg")
plt.show()


##Plot données SET
plt.figure()
color = 'tab:red'
plt.ylabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18", color=color)
plt.xlabel('# Pulses',fontsize="18", color=color)
plt.plot(b1_Pulse_set_f[:], b1_R_set_f[:], 'r.', markersize=dot_plot)
plt.plot(b2_Pulse_set_f[:], b2_R_set_f[:], 'b.', markersize=dot_plot)
plt.plot(b3_Pulse_set_f[:], b3_R_set_f[:], 'g.', markersize=dot_plot)
plt.plot(b4_Pulse_set_f[:], b4_R_set_f[:], 'k.', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color)
plt.tick_params(axis='x', labelcolor=color)
plt.yscale('log')
plt.ylim([0.8, 70])
_a = 425
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées SET only',fontsize="14")
plt.savefig(""+dos+"/Données traitées SET only.svg")

##Plot données RESET
plt.figure()
color = 'tab:red'
plt.ylabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18", color=color)
plt.xlabel('# Pulses',fontsize="18", color=color)
plt.plot(b1_Pulse_reset_f[:], b1_R_reset_f[:], 'r*', markersize=dot_plot)
plt.plot(b2_Pulse_reset_f[:], b2_R_reset_f[:], 'b*', markersize=dot_plot)
plt.plot(b3_Pulse_reset_f[:], b3_R_reset_f[:], 'g*', markersize=dot_plot)
plt.plot(b4_Pulse_reset_f[:], b4_R_reset_f[:], 'k*', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color)
plt.tick_params(axis='x', labelcolor=color)
plt.yscale('log')
plt.ylim([0.8, 70])
_a = 425
_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées RESET only',fontsize="14")
plt.savefig(""+dos+"/Données traitées RESET only.svg")

####Statistique
## CDF & PDF=Gaussiennes

####Calcul de CDF: Données axe des ordonés

x0_s = np.sort(b1_R_set_f)
T0_s = len(x0_s)
y0_s = np.arange(start=1, stop=T0_s + 1, step=1) / float(T0_s)
if len(b1_R_set_f)>1:
    pdf_B1_s = np.gradient(y0_s * 100, x0_s)
x0_re = np.sort(b1_R_reset_f)
T0_re = len(x0_re)
y0_re = np.arange(start=1, stop=T0_re + 1, step=1) / float(T0_re)
if len(b1_R_reset_f)>1:
    pdf_B1_re = np.gradient(y0_re * 100, x0_re)
x1_s = np.sort(b2_R_set_f)
T1_s = len(x1_s)
y1_s = np.arange(start=1, stop=T1_s + 1, step=1) / float(T1_s)
if len(b2_R_set_f)>1:
    pdf_B2_s = np.gradient(y1_s * 100, x1_s)
x1_re = np.sort(b2_R_reset_f)
T1_re = len(x1_re)
y1_re = np.arange(start=1, stop=T1_re + 1, step=1) / float(T1_re)
if len(b2_R_reset_f)>1:
    pdf_B2_re = np.gradient(y1_re * 100, x1_re)
x2_s = np.sort(b3_R_set_f)
T2_s = len(x2_s)
y2_s = np.arange(start=1, stop=T2_s + 1, step=1) / float(T2_s)
if len(b3_R_set_f)>1:
    pdf_B3_s = np.gradient(y2_s * 100, x2_s)
x2_re = np.sort(b3_R_reset_f)
T2_re = len(x2_re)
y2_re = np.arange(start=1, stop=T2_re + 1, step=1) / float(T2_re)
if len(b3_R_reset_f )>1:
    pdf_B3_re = np.gradient(y2_re * 100, x2_re)
x3_s = np.sort(b4_R_set_f)
T3_s = len(x3_s)
y3_s = np.arange(start=1, stop=T3_s + 1, step=1) / float(T3_s)
if len(b4_R_set_f )>1:
    pdf_B4_s = np.gradient(y3_s * 100, x3_s)
x3_re = np.sort(b4_R_reset_f)
T3_re = len(x3_re)
y3_re = np.arange(start=1, stop=T3_re + 1, step=1) / float(T3_re)
if len(b4_R_reset_f )>1:
    pdf_B4_re = np.gradient(y3_re * 100, x3_re)
###Plot CDF
plt.figure()  ## CDF SET & RESET
plt.title('CDF SET & RESET',fontsize="14")
if b1_R_set_f!=[]: plt.plot(x0_s, y0_s * 100, 'r.', markersize=dot_plot)
if b1_R_reset_f!=[]: plt.plot(x0_re, y0_re * 100, 'r*', markersize=dot_plot)
if b2_R_set_f!=[]: plt.plot(x1_s, y1_s * 100, 'b.', markersize=dot_plot)
if b2_R_reset_f!=[]: plt.plot(x1_re, y1_re * 100, 'b*', markersize=dot_plot)
if b3_R_set_f!=[]: plt.plot(x2_s, y2_s * 100, 'g.', markersize=dot_plot)
if b3_R_reset_f!=[]: plt.plot(x2_re, y2_re * 100, 'g*', markersize=dot_plot)
if b4_R_set_f!=[]: plt.plot(x3_s, y3_s * 100, 'k.', markersize=dot_plot)
if b4_R_reset_f!=[]: plt.plot(x3_re, y3_re * 100, 'k*', markersize=dot_plot)
# plt.ylim([0,3e-6])
plt.xlim([0.8, 70])
plt.xscale('log')
plt.xlabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig(""+dos+"/CDF SET & RESET.svg")

plt.figure()  ## CDF SET ONLY
plt.title('CDF SET ONLY',fontsize="14")
if b1_R_set_f!=[]: plt.plot(x0_s, y0_s * 100, 'r.', markersize=dot_plot)
if b2_R_set_f!=[]: plt.plot(x1_s, y1_s * 100, 'b.', markersize=dot_plot)
if b3_R_set_f!=[]: plt.plot(x2_s, y2_s * 100, 'g.', markersize=dot_plot)
if b4_R_set_f!=[]: plt.plot(x3_s, y3_s * 100, 'k.', markersize=dot_plot)
# plt.ylim([0,3e-6])
plt.xlim([0.8, 70])
plt.xscale('log')
plt.xlabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig(""+dos+"/CDF SET ONLY.svg")

plt.figure()  ## CDF RESET ONLY
plt.title('CDF RESET ONLY',fontsize="14")
if b1_R_reset_f!=[]: plt.plot(x0_re, y0_re * 100, 'r*', markersize=dot_plot)
if b2_R_reset_f!=[]: plt.plot(x1_re, y1_re * 100, 'b*', markersize=dot_plot)
if b3_R_reset_f!=[]: plt.plot(x2_re, y2_re * 100, 'g*', markersize=dot_plot)
if b4_R_reset_f!=[]: plt.plot(x3_re, y3_re * 100, 'k*', markersize=dot_plot)
# plt.ylim([0,3e-6])
plt.xlim([0.8, 70])
plt.xscale('log')
plt.xlabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18")
plt.ylabel("CDF en %",fontsize="18")
plt.grid()
plt.savefig(""+dos+"/CDF RESET ONLY.svg")

### PDF
plt.figure()  ##  PDF SET & RESET
plt.title('PDF SET & RESET',fontsize="14")
if len(b1_R_set_f)>1: plt.plot(x0_s, pdf_B1_s, 'r1', markersize=dot_plot-4)
if len(b1_R_reset_f)>1: plt.plot(x0_re, pdf_B1_re, 'r>', markersize=dot_plot-4)
if len(b2_R_set_f)>1: plt.plot(x1_s, pdf_B2_s, 'b1', markersize=dot_plot-4)
if len(b2_R_reset_f)>1: plt.plot(x1_re, pdf_B2_re, 'b>', markersize=dot_plot-4)
if len(b3_R_set_f)>1: plt.plot(x2_s, pdf_B3_s, 'g1', markersize=dot_plot-4)
if len(b3_R_reset_f)>1: plt.plot(x2_re, pdf_B3_re, 'g>', markersize=dot_plot-4)
if len(b4_R_set_f)>1: plt.plot(x3_s, pdf_B4_s, 'k1', markersize=dot_plot-4)
if len(b4_R_reset_f)>1: plt.plot(x3_re, pdf_B4_re, 'k>', markersize=dot_plot-4)
# plt.ylim([0,3e-6])
plt.xlim([0.8, 70])
#plt.xscale('log')
plt.xlabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18")
plt.ylabel("PDF",fontsize="18")
plt.grid()
plt.savefig(""+dos+"/PDF SET & RESET.svg")

## Histogramme
plt.figure()
# bande1
plt.hist(b1_R_set_f, rwidth=0.9, color='k', label='SET')
plt.hist(b1_R_reset_f, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(round(bande[0][3],2)) + "*(Cp="+str(f"{(cp):.2f}")+")," + str(round(bande[0][4],2)) + "*(Cp="+str(f"{(cp):.2f}")+")]",fontsize="14")
plt.xlabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18")
plt.ylabel("Nombre de valeur",fontsize="18")
plt.legend()
plt.savefig(""+dos+"/hist-bande1.svg")
plt.figure()
# bande2
plt.hist(b2_R_set_f, rwidth=0.9, color='k', label='SET')
plt.hist(b2_R_reset_f, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(round(bande[1][3],2)) + "*(Cp="+str(f"{(cp):.2f}")+")," + str(round(bande[1][4],2)) + "*(Cp="+str(f"{(cp):.2f}")+")]",fontsize="14")
plt.xlabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18")
plt.ylabel("Nombre de valeur",fontsize="18")
plt.legend()
plt.savefig(""+dos+"/hist-bande2.svg")
plt.figure()
# bande3
plt.hist(b3_R_set_f, rwidth=0.9, color='k', label='SET')
plt.hist(b3_R_reset_f, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(round(bande[2][3],2)) + "*(Cp="+str(f"{(cp):.2f}")+")," + str(round(bande[2][4],2)) + "*(Cp="+str(f"{(cp):.2f}")+")]",fontsize="14")
plt.xlabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18")
plt.ylabel("Nombre de valeur",fontsize="18")
plt.legend()
plt.savefig(""+dos+"/hist-bande3.svg")
plt.figure()
# bande4
plt.hist(b4_R_set_f, rwidth=0.9, color='k', label='SET')
plt.hist(b4_R_reset_f, rwidth=0.9, color='r', label='RESET')
plt.title("Hist data Bande [" + str(round(bande[3][3],2)) + "*(Cp="+str(f"{(cp):.2f}")+")," + str(round(bande[3][4],2)) + "*(Cp="+str(f"{(cp):.2f}")+")]",fontsize="14")
plt.xlabel("Conductance in ( *(Cp="+str(f"{(cp):.2f}")+")) S",fontsize="18")
plt.ylabel("Nombre de Valeur",fontsize="18")
plt.legend()
plt.savefig(""+dos+"/hist-bande4.svg")

plt.show()