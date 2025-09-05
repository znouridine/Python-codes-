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
L=8 #longueur fig
name = "Test cycle synapse Ligne011 PVDE66 Rp146k_01"
Via = 1
rp = float(146)
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
l_h = 2500  ## t_ SI TOUT LE PLOT
### Fenêtre Memoire
R_on= 1500
R_off= 35000

c2 = data[l_b:l_h, 1]
c4 = data[l_b:l_h, 3]
c9 = data[l_b:l_h, 8]
c10 = data[l_b:l_h, 9]
c12 = data[l_b:l_h, 11]
c11 = data[l_b:l_h, 10]
t = int(len(c4))


font_ = {'family': 'cursive',
        'color':  'k',
        'weight': 'bold',
        'size': 20,
        }
font__ = {'family': 'cursive',
        'color':  'r',
        'weight': 'bold',
        'size': 20,
        }
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
plt.ylim([800, 30000])
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
###Traitement phase 1

b1_R = [];b1_v = [];b1_t = [];b1_Pulse = []
b2_R = [];b2_v = [];b2_t = [];b2_Pulse = []
b3_R = [];b3_v = [];b3_t = [];b3_Pulse = []
b4_R = [];b4_v = [];b4_t = [];b4_Pulse = []
b5_R = [];b5_v = [];b5_t = [];b5_Pulse = []
b6_R = [];b6_v = [];b6_t = [];b6_Pulse = []
b7_R = [];b7_v = [];b7_t = [];b7_Pulse = []
b8_R = [];b8_v = [];b8_t = [];b8_Pulse = []

for i in range(t):
    if bande[0][4] <= c4[i] <= bande[0][5]:
        b1_R.append(c4[i])
        b1_v.append(c12[i])
        b1_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b1_Pulse.append(i + 1)
        continue

    elif bande[1][4] <= c4[i] <= bande[1][5]:
        b2_R.append(c4[i])
        b2_v.append(c12[i])
        b2_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b2_Pulse.append(i + 1)

    elif bande[2][4] <= c4[i] <= bande[2][5]:
        b3_R.append(c4[i])
        b3_v.append(c12[i])
        b3_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b3_Pulse.append(i + 1)

    elif bande[3][4] <= c4[i] <= bande[3][5]:
        b4_R.append(c4[i])
        b4_v.append(c12[i])
        b4_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b4_Pulse.append(i + 1)
    elif bande[4][4] <= c4[i] <= bande[4][5]:
        b5_R.append(c4[i])
        b5_v.append(c12[i])
        b5_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b5_Pulse.append(i + 1)
        continue

    elif bande[5][4] <= c4[i] <= bande[5][5]:
        b6_R.append(c4[i])
        b6_v.append(c12[i])
        b6_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b6_Pulse.append(i + 1)

    elif bande[6][4] <= c4[i] <= bande[6][5]:
        b7_R.append(c4[i])
        b7_v.append(c12[i])
        b7_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b7_Pulse.append(i + 1)

    elif bande[7][4] <= c4[i] <= bande[7][5]:
        b8_R.append(c4[i])
        b8_v.append(c12[i])
        b8_t.append(int((c9[i] + c10[i] + c11[i]) * 10 ** 9))
        b8_Pulse.append(i + 1)


###traitment phase 2
b1_R_set = [];b1_v_set = [];b1_t_set = [];b1_R_reset = [];b1_v_reset = [];b1_t_reset = [];b1_Pulse_set = [];b1_Pulse_reset = []
b2_R_set = [];b2_v_set = [];b2_t_set = [];b2_R_reset = [];b2_v_reset = [];b2_t_reset = [];b2_Pulse_set = [];b2_Pulse_reset = []
b3_R_set = [];b3_v_set = [];b3_t_set = [];b3_R_reset = [];b3_v_reset = [];b3_t_reset = [];b3_Pulse_set = [];b3_Pulse_reset = []
b4_R_set = [];b4_v_set = [];b4_t_set = [];b4_R_reset = [];b4_v_reset = [];b4_t_reset = [];b4_Pulse_set = [];b4_Pulse_reset = []
b5_R_set = [];b5_v_set = [];b5_t_set = [];b5_R_reset = [];b5_v_reset = [];b5_t_reset = [];b5_Pulse_set = [];b5_Pulse_reset = []
b6_R_set = [];b6_v_set = [];b6_t_set = [];b6_R_reset = [];b6_v_reset = [];b6_t_reset = [];b6_Pulse_set = [];b6_Pulse_reset = []
b7_R_set = [];b7_v_set = [];b7_t_set = [];b7_R_reset = [];b7_v_reset = [];b7_t_reset = [];b7_Pulse_set = [];b7_Pulse_reset = []
b8_R_set = [];b8_v_set = [];b8_t_set = [];b8_R_reset = [];b8_v_reset = [];b8_t_reset = [];b8_Pulse_set = [];b8_Pulse_reset = []

for i in range(t):
    ##Bande 1 SET
    if bande[0][4] <= X1[i][1] <= bande[0][5]:
        b1_R_set.append(X1[i][1])
        b1_Pulse_set.append(X1[i][0])
    ##Bande 1 RESET
    if bande[0][4] <= X0[i][1] <= bande[0][5]:
        b1_R_reset.append(X0[i][1])
        b1_Pulse_reset.append(X0[i][0])
    ##Bande 2 SET
    elif bande[1][4] <= X1[i][1] <= bande[1][5]:
        b2_R_set.append(X1[i][1])
        b2_Pulse_set.append(X1[i][0])
    ##Bande 2 RESET
    elif bande[1][4] <= X0[i][1] <= bande[1][5]:
        b2_R_reset.append(X0[i][1])
        b2_Pulse_reset.append(X0[i][0])
    ##Bande 3 SET
    elif bande[2][4] <= X1[i][1] <= bande[2][5]:
        b3_R_set.append(X1[i][1])
        b3_Pulse_set.append(X1[i][0])
    ##Bande 3 RESET
    elif bande[2][4] <= X0[i][1] <= bande[2][5]:
        b3_R_reset.append(X0[i][1])
        b3_Pulse_reset.append(X0[i][0])
    ##Bande 4 SET
    elif bande[3][4] <= X1[i][1] <= bande[3][5]:
        b4_R_set.append(X1[i][1])
        b4_Pulse_set.append(X1[i][0])
    ##Bande 4 RESET
    elif bande[3][4] <= X0[i][1] <= bande[3][5]:
        b4_R_reset.append(X0[i][1])
        b4_Pulse_reset.append(X0[i][0])
    ##Bande 5 SET
    elif bande[4][4] <= X1[i][1] <= bande[4][5]:
        b5_R_set.append(X1[i][1])
        b5_Pulse_set.append(X1[i][0])
    ##Bande 5 RESET
    elif bande[4][4] <= X0[i][1] <= bande[4][5]:
        b5_R_reset.append(X0[i][1])
        b5_Pulse_reset.append(X0[i][0])
    ##Bande 6 SET
    elif bande[5][4] <= X1[i][1] <= bande[5][5]:
        b6_R_set.append(X1[i][1])
        b6_Pulse_set.append(X1[i][0])
    ##Bande 6 RESET
    elif bande[5][4] <= X0[i][1] <= bande[5][5]:
        b6_R_reset.append(X0[i][1])
        b6_Pulse_reset.append(X0[i][0])
    ##Bande 7 SET
    elif bande[6][4] <= X1[i][1] <= bande[6][5]:
        b7_R_set.append(X1[i][1])
        b7_Pulse_set.append(X1[i][0])
    ##Bande 7 RESET
    elif bande[6][4] <= X0[i][1] <= bande[6][5]:
        b7_R_reset.append(X0[i][1])
        b7_Pulse_reset.append(X0[i][0])
    ##Bande 8 SET
    elif bande[7][4] <= X1[i][1] <= bande[7][5]:
        b8_R_set.append(X1[i][1])
        b8_Pulse_set.append(X1[i][0])
    ##Bande 8 RESET
    elif bande[7][4] <= X0[i][1] <= bande[7][5]:
        b8_R_reset.append(X0[i][1])
        b8_Pulse_reset.append(X0[i][0])


b1_set = len(b1_R_set);b1_reset = len(b1_R_reset)
b2_set = len(b2_R_set);b2_reset = len(b2_R_reset)
b3_set = len(b3_R_set);b3_reset = len(b3_R_reset)
b4_set = len(b4_R_set);b4_reset = len(b4_R_reset)
b5_set = len(b5_R_set);b5_reset = len(b5_R_reset)
b6_set = len(b6_R_set);b6_reset = len(b6_R_reset)
b7_set = len(b7_R_set);b7_reset = len(b7_R_reset)
b8_set = len(b8_R_set);b8_reset = len(b8_R_reset)

b1_s = 1;b1_re = 1
b2_s = 1;b2_re = 1
b3_s = 1;b3_re = 1
b4_s = 1;b4_re = 1
b5_s = 1;b5_re = 1
b6_s = 1;b6_re = 1
b7_s = 1;b7_re = 1
b8_s = 1;b8_re = 1


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
            for tyy in range(min(len(a_re) , len(a_s), len(b_s), len(b_re))):
                pluse.append(abs(b_s[tyy] - a_s[tyy]) + abs(b_re[tyy] - a_re[tyy]))
    Pulse_av = min(pluse) - 1
    return Pulse_av




Pulse_av = nb_p1cy ()

## Traitement pour récuperer Le premier point à tomber dans le bande voulu
## et ainsi éviter que pour le même SET ou RESET on récupère plus d'un point
## L'idée c'est d'arrêter le programme dès que l'on a atteint la bande cible
## Pour ça, je regarde les numéros de pulse: Si deux conductance dans un même
## niveau ont pas un écart de numéro de pulse inférieur à Pulse_av (fixé),
## je ne conserve que la première valeur à être tombée dans la bande cible.

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
        b1_re = len(b1_Pulse_reset)
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
        b2_re = len(b2_Pulse_reset)
    # SET bande 3
    for i in range(2, b3_set):
        if i < b3_s and abs(b3_Pulse_set[i] - b3_Pulse_set[i - 1]) <= Pulse_av and len(b3_Pulse_set) != 1 \
                and len(b3_Pulse_set) != 0:
            b3_R_set[i - 1] = b3_R_set[i]

            b3_Pulse_set[i - 1] = b3_Pulse_set[i]
        b3_s = len(b3_Pulse_set)
    # RESET bande 3
    for i in range(2, b3_reset):
        if i < b3_re and abs(b3_Pulse_reset[i] - b3_Pulse_reset[i - 1]) <= Pulse_av and len(b3_R_reset) != 1 \
                and len(b3_R_reset) != 0:
            b3_R_reset[i - 1] = b3_R_reset[i]

            b3_Pulse_reset[i - 1] = b3_Pulse_reset[i]
        b3_re = len(b3_Pulse_reset)
    # SET bande 4
    for i in range(2, b4_set):
        if i < b4_s and abs(b4_Pulse_set[i] - b4_Pulse_set[i - 1]) <= Pulse_av and len(b4_Pulse_set) != 1 \
                and len(b4_Pulse_set) != 0:
            b4_R_set[i - 1] = b4_R_set[i]

            b4_Pulse_set[i - 1] = b4_Pulse_set[i]
        b4_s = len(b4_Pulse_set)
    # RESET bande 4
    for i in range(2, b4_reset):
        if i < b4_re and abs(b4_Pulse_reset[i] - b4_Pulse_reset[i - 1]) <= Pulse_av and len(b4_R_reset) != 1 \
                and len(b4_R_reset) != 0:
            b4_R_reset[i - 1] = b4_R_reset[i]
            b4_Pulse_reset[i - 1] = b4_Pulse_reset[i]
        b4_re = len(b4_Pulse_reset)
    # SET bande 5
    for i in range(2, b5_set):
        if i < b5_s and abs(b5_Pulse_set[i] - b5_Pulse_set[i - 1]) <= Pulse_av and len(b5_Pulse_set) != 1 \
                and len(b5_Pulse_set) != 0:
            b5_R_set[i - 1] = b5_R_set[i]

            b5_Pulse_set[i - 1] = b5_Pulse_set[i]
        b5_s = len(b5_Pulse_set)
    # RESET bande 5
    for i in range(2, b5_reset):
        if i < b5_re and abs(b5_Pulse_reset[i] - b5_Pulse_reset[i - 1]) <= Pulse_av and len(b5_R_reset) != 1 \
                and len(b5_R_reset) != 0:
            b5_R_reset[i - 1] = b5_R_reset[i]
            b5_Pulse_reset[i - 1] = b5_Pulse_reset[i]
        b5_re = len(b5_Pulse_reset)
    # SET bande 6
    for i in range(2, b6_set):
        if i < b6_s and abs(b6_Pulse_set[i] - b6_Pulse_set[i - 1]) <= Pulse_av and len(b6_Pulse_set) != 1 \
                and len(b6_Pulse_set) != 0:
            b6_R_set[i - 1] = b6_R_set[i]

            b6_Pulse_set[i - 1] = b6_Pulse_set[i]
        b6_s = len(b6_Pulse_set)
    # RESET bande 6
    for i in range(2, b6_reset):
        if i < b6_re and abs(b6_Pulse_reset[i] - b6_Pulse_reset[i - 1]) <= Pulse_av and len(b6_R_reset) != 1 \
                and len(b6_R_reset) != 0:
            b6_R_reset[i - 1] = b6_R_reset[i]
            b6_Pulse_reset[i - 1] = b6_Pulse_reset[i]
        b6_re = len(b6_Pulse_reset)
    # SET bande 7
    for i in range(2, b7_set):
        if i < b7_s and abs(b7_Pulse_set[i] - b7_Pulse_set[i - 1]) <= Pulse_av and len(b7_Pulse_set) != 1 \
                and len(b7_Pulse_set) != 0:
            b7_R_set[i - 1] = b7_R_set[i]
            b7_Pulse_set[i - 1] = b7_Pulse_set[i]
        b7_s = len(b7_Pulse_set)
    # RESET bande 7
    for i in range(2, b7_reset):
        if i < b7_re and abs(b7_Pulse_reset[i] - b7_Pulse_reset[i - 1]) <= Pulse_av and len(b7_R_reset) != 1 \
                and len(b7_R_reset) != 0:
            b7_R_reset[i - 1] = b7_R_reset[i]
            b7_Pulse_reset[i - 1] = b7_Pulse_reset[i]
        b7_re = len(b7_Pulse_reset)
    # SET bande 8
    for i in range(2, b8_set):
        if i < b8_s and abs(b8_Pulse_set[i] - b8_Pulse_set[i - 1]) <= Pulse_av and len(b8_Pulse_set) != 1 \
                and len(b8_Pulse_set) != 0:
            b8_R_set[i - 1] = b8_R_set[i]
            b8_Pulse_set[i - 1] = b8_Pulse_set[i]
        b8_s = len(b8_Pulse_set)
    # RESET bande 8
    for i in range(2, b8_reset):
        if i < b8_re and abs(b8_Pulse_reset[i] - b8_Pulse_reset[i - 1]) <= Pulse_av and len(b8_R_reset) != 1 \
                and len(b8_R_reset) != 0:
            b8_R_reset[i - 1] = b8_R_reset[i]
            b8_Pulse_reset[i - 1] = b8_Pulse_reset[i]
        b8_re = len(b8_Pulse_reset)

### Après ce traitement j'obtiens des list avec des élements qui se repétent. J'élimine les repétitions
###Eliminer les duplicata dans mes list
##Initialisation list

b1_R_set_f =[];b1_R_reset_f =[];b1_v_set_f =[];b1_v_reset_f =[];b1_t_set_f =[];b1_t_reset_f =[]
b2_R_set_f =[];b2_R_reset_f =[];b2_v_set_f =[];b2_v_reset_f =[];b2_t_set_f =[];b2_t_reset_f =[]
b3_R_set_f =[];b3_R_reset_f =[];b3_v_set_f =[];b3_v_reset_f =[];b3_t_set_f =[];b3_t_reset_f =[]
b4_R_set_f =[];b4_R_reset_f =[];b4_v_set_f =[];b4_v_reset_f =[];b4_t_set_f =[];b4_t_reset_f =[]
b5_R_set_f =[];b5_R_reset_f =[];b5_v_set_f =[];b5_v_reset_f =[];b5_t_set_f =[];b5_t_reset_f =[]
b6_R_set_f =[];b6_R_reset_f =[];b6_v_set_f =[];b6_v_reset_f =[];b6_t_set_f =[];b6_t_reset_f =[]
b7_R_set_f =[];b7_R_reset_f =[];b7_v_set_f =[];b7_v_reset_f =[];b7_t_set_f =[];b7_t_reset_f =[]
b8_R_set_f =[];b8_R_reset_f =[];b8_v_set_f =[];b8_v_reset_f =[];b8_t_set_f =[];b8_t_reset_f =[]
b1_Pulse_set_f =[];b1_Pulse_reset_f =[]
b2_Pulse_set_f =[];b2_Pulse_reset_f =[]
b3_Pulse_set_f =[];b3_Pulse_reset_f =[]
b4_Pulse_set_f =[];b4_Pulse_reset_f =[]
b5_Pulse_set_f =[];b5_Pulse_reset_f =[]
b6_Pulse_set_f =[];b6_Pulse_reset_f =[]
b7_Pulse_set_f =[];b7_Pulse_reset_f =[]
b8_Pulse_set_f =[];b8_Pulse_reset_f =[]

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
#bande5
for w in b5_Pulse_set:#Set
    if w not in b5_Pulse_set_f:
        b5_Pulse_set_f.append(w)
        ind=b5_Pulse_set.index(w)
        b5_R_set_f.append(b5_R_set[ind])
        b5_v_set_f.append(b5_v[b5_Pulse.index(w)])
        b5_t_set_f.append(b5_t[b5_Pulse.index(w)])
for w in b5_Pulse_reset:#Reset
    if w not in b5_Pulse_reset_f:
        b5_Pulse_reset_f.append(w)
        ind=b5_Pulse_reset.index(w)
        b5_R_reset_f.append(b5_R_reset[ind])
        b5_v_reset_f.append(b5_v[b5_Pulse.index(w)])
        b5_t_reset_f.append(b5_t[b5_Pulse.index(w)])
#bande6
for w in b6_Pulse_set:#Set
    if w not in b6_Pulse_set_f:
        b6_Pulse_set_f.append(w)
        ind=b6_Pulse_set.index(w)
        b6_R_set_f.append(b6_R_set[ind])
        b6_v_set_f.append(b6_v[b6_Pulse.index(w)])
        b6_t_set_f.append(b6_t[b6_Pulse.index(w)])
for w in b6_Pulse_reset:#Reset
    if w not in b6_Pulse_reset_f:
        b6_Pulse_reset_f.append(w)
        ind=b6_Pulse_reset.index(w)
        b6_R_reset_f.append(b6_R_reset[ind])
        b6_v_reset_f.append(b6_v[b6_Pulse.index(w)])
        b6_t_reset_f.append(b6_t[b6_Pulse.index(w)])
#bande7
for w in b7_Pulse_set:#Set
    if w not in b7_Pulse_set_f:
        b7_Pulse_set_f.append(w)
        ind=b7_Pulse_set.index(w)
        b7_R_set_f.append(b7_R_set[ind])
        b7_v_set_f.append(b7_v[b7_Pulse.index(w)])
        b7_t_set_f.append(b7_t[b7_Pulse.index(w)])
for w in b7_Pulse_reset:#Reset
    if w not in b7_Pulse_reset_f:
        b7_Pulse_reset_f.append(w)
        ind=b7_Pulse_reset.index(w)
        b7_R_reset_f.append(b7_R_reset[ind])
        b7_v_reset_f.append(b7_v[b7_Pulse.index(w)])
        b7_t_reset_f.append(b7_t[b7_Pulse.index(w)])
#bande8
for w in b8_Pulse_set:#Set
    if w not in b8_Pulse_set_f:
        b8_Pulse_set_f.append(w)
        ind=b8_Pulse_set.index(w)
        b8_R_set_f.append(b8_R_set[ind])
        b8_v_set_f.append(b8_v[b8_Pulse.index(w)])
        b8_t_set_f.append(b8_t[b8_Pulse.index(w)])
for w in b8_Pulse_reset:#Reset
    if w not in b8_Pulse_reset_f:
        b8_Pulse_reset_f.append(w)
        ind=b8_Pulse_reset.index(w)
        b8_R_reset_f.append(b8_R_reset[ind])
        b8_v_reset_f.append(b8_v[b8_Pulse.index(w)])
        b8_t_reset_f.append(b8_t[b8_Pulse.index(w)])
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

## Bande 5 SET
if len(b5_v_set_f)>1 :
    sd_s_5 = stdev(b5_v_set_f)
    md_s_5 = median(b5_v_set_f)
    m_s_5 = mean(b5_v_set_f)
elif len(b5_v_set_f)==0:
    sd_s_5 = 0.001
    m_s_5 = 0.001
    md_s_5 = 0.001
elif len(b5_v_set_f)==1:
    sd_s_5 = 0.001
    m_s_5 = b5_v_set_f[0]
    md_s_5 = b5_v_set_f[0]
## Bande 5 RESET
if len(b5_v_reset_f)>1 :
    m_re_5 = mean(b5_v_reset_f)
    sd_re_5 = stdev(b5_v_reset_f)
    md_re_5 = median(b5_v_reset_f)
elif len(b5_v_reset_f)==0:
    sd_re_5 = 0.001
    m_re_5 = 0.001
    md_re_5 = 0.001
elif len(b5_v_reset_f)==1:
    sd_re_5 = 0.001
    m_re_5 = b5_v_reset_f[0]
    md_re_5 = b5_v_reset_f[0]
## Bande 6 SET
if len(b6_v_set_f) > 1 :
    sd_s_6= stdev(b6_v_set_f)
    md_s_6 = median(b6_v_set_f)
    m_s_6 = mean(b6_v_set_f)
elif len(b6_v_set_f)==0 :
    sd_s_6 = 0.001
    m_s_6 = 0.001
    md_s_6 = 0.001
elif len(b6_v_set_f)==1 :
    sd_s_6 = 0.001
    m_s_6 = b6_v_set_f[0]
    md_s_6 = b6_v_set_f[0]
## Bande 6 RESET
if len(b6_v_reset_f) > 1 :
    m_re_6 = mean(b6_v_reset_f)
    sd_re_6 = stdev(b6_v_reset_f)
    md_re_6 = median(b6_v_reset_f)
elif len(b6_v_reset_f) == 0:
    sd_re_6 = 0.001
    m_re_6= 0.001
    md_re_6 = 0.001
elif len(b6_v_reset_f) == 1:
    sd_re_6 = 0.001
    m_re_6= b6_v_reset_f[0]
    md_re_6 = b6_v_reset_f[0]
## Bande 7 SET
if len(b7_v_set_f) > 1 :
    sd_s_7 = stdev(b7_v_set_f)
    md_s_7 = median(b7_v_set_f)
    m_s_7 = mean(b7_v_set_f)
elif len(b7_v_set_f) == 0 :
    sd_s_7 = 0.001
    m_s_7 = 0.001
    md_s_7 = 0.001
elif len(b7_v_set_f) == 1:
    sd_s_7 = 0.001
    m_s_7 = b7_v_set_f[0]
    md_s_7 = b7_v_set_f[0]
## Bande 7 RESET
if len(b7_v_reset_f) > 1 :
    m_re_7 = mean(b7_v_reset_f)
    sd_re_7 = stdev(b7_v_reset_f)
    md_re_7 = median(b7_v_reset_f)
elif len(b7_v_reset_f) == 0 :
    sd_re_7 = 0.001
    m_re_7 = 0.001
    md_re_7 = 0.001
elif len(b7_v_reset_f) == 1 :
    sd_re_7 = 0.001
    m_re_7 = b7_v_reset_f[0]
    md_re_7 = b7_v_reset_f[0]
## Bande 8 SET
if len(b8_v_set_f) > 1 :
    sd_s_8 = stdev(b8_v_set_f)
    md_s_8 = median(b8_v_set_f)
    m_s_8 = mean(b8_v_set_f)
elif len(b8_v_set_f) == 0:
    sd_s_8 = 0.001
    m_s_8 = 0.001
    md_s_8 = 0.001
elif len(b8_v_set_f) == 1:
    sd_s_8 = 0.001
    m_s_8 = b8_v_set_f[0]
    md_s_8 = b8_v_set_f[0]
## Bande 4 RESET
if len(b8_v_reset_f) > 1 :
    m_re_8 = mean(b8_v_reset_f)
    sd_re_8 = stdev(b8_v_reset_f)
    md_re_8 = median(b8_v_reset_f)
elif len(b8_v_reset_f) == 0 :
    sd_re_8 = 0.001
    m_re_8 = 0.001
    md_re_8 = 0.001
elif len(b8_v_reset_f) == 1 :
    sd_re_8 = 0.001
    m_re_8 = b8_v_reset_f[0]
    md_re_8 = b8_v_reset_f[0]
####Plot figure

#fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 11), height_ratios=[1, 1])
plt.figure(figsize= (L,l))
color = 'tab:red'
plt.ylabel("Résistance en Ohm",fontsize=font, color=color)
plt.xlabel('# Pulses',fontsize=font, color=color)
plt.plot(b1_Pulse_set_f[:], b1_R_set_f[:], 'r.', markersize=dot_plot)
plt.plot(b1_Pulse_reset_f[:], b1_R_reset_f[:], 'r*', markersize=dot_plot)
plt.plot(b2_Pulse_set_f[:], b2_R_set_f[:], 'b.', markersize=dot_plot)
plt.plot(b2_Pulse_reset_f[:], b2_R_reset_f[:], 'b*', markersize=dot_plot)
plt.plot(b3_Pulse_set_f[:], b3_R_set_f[:], 'g.', markersize=dot_plot)
plt.plot(b3_Pulse_reset_f[:], b3_R_reset_f[:], 'g*', markersize=dot_plot)
plt.plot(b4_Pulse_set_f[:], b4_R_set_f[:], 'k.', markersize=dot_plot)
plt.plot(b4_Pulse_reset_f[:], b4_R_reset_f[:], 'k*', markersize=dot_plot)
plt.plot(b5_Pulse_set_f[:], b5_R_set_f[:], 'y.', markersize=dot_plot)
plt.plot(b5_Pulse_reset_f[:], b5_R_reset_f[:], 'y*', markersize=dot_plot)
plt.plot(b6_Pulse_set_f[:], b6_R_set_f[:], 'c.', markersize=dot_plot)
plt.plot(b6_Pulse_reset_f[:], b6_R_reset_f[:], 'c*', markersize=dot_plot)
plt.plot(b7_Pulse_set_f[:], b7_R_set_f[:], 'm.', markersize=dot_plot)
plt.plot(b7_Pulse_reset_f[:], b7_R_reset_f[:], 'm*', markersize=dot_plot)
plt.plot(b8_Pulse_set_f[:], b8_R_set_f[:], 'b.', markersize=dot_plot)
plt.plot(b8_Pulse_reset_f[:], b8_R_reset_f[:], 'b*', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color, labelsize=labsize)
plt.tick_params(axis='x', labelcolor=color, labelsize=labsize)
plt.yscale('log')
#plt.ylim([0.8, 70])
#plt.xticks([])
#ax1.legend(loc='upper right', fontsize="12")
_a = 0
_b = t # Limits of x axis
#ax1.set_xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données traitées Ph2: SET & RESET',fontsize=font)
plt.savefig(""+dos+"/Data traitées Ph2 SET & RESET.svg")


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

plt.show()

