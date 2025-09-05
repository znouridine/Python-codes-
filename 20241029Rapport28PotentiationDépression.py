# -*- coding: utf-8 -*-
"""
Created on Thus October 31 15:37:32 2024

@author: nouri
"""
'''
Choisir d'abord le Cycle que tu veux faire sortir.
Ce programme se compile en DEBBUG, au premier checkpoint allez au dossier 
source et ouvrir DEP et POT puis supprimer les zéros et les valeur trop proches puis les fermer ensuite revenir 
et continuer le programme debugg
'''
globals().clear()
import numpy as np
import pandas as pd
from math import pi
import matplotlib.pyplot as plt

dos = str(20241108)
dot_plot = 28  # taille point plot
labsize = 24 # Taille graduation axe
font = 26 # Taille titres des axes
fontLEG = 22 # taille legende
l=8 #largeur fig
L=9 #longueur fig
name = input("Nom fichier :")
rp = float(input('Résistance pristine: '))
ep = 55  # float(input('Epaisseur en nm: '))
Via = 1.2  # float(input('Taille du VIA en um: '))
Cr = 30  # float(input('% du Chrome: '))
cp = (1 / rp * 1e+3)
data = np.loadtxt("" + dos + "/" + name + ".NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1]
c4 = ((1 / data[:, 3]) / (cp)) * 1e+6  # conductance normalisée
c9 = data[:, 8]
c10 = data[:, 9]
c12 = data[:, 11]
c11 = data[:, 10]
t = len(c4)
A0 = []
X0 = np.zeros((int(t), 5))
X1 = np.zeros((int(t), 5))
k = 0
r = 0
#fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 9), height_ratios=[3, 1, 1])
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
for t in range(len(c12)):
    Rp.append(float(rp * 1e+3))  # Mettre la valeur de
    Cp.append(float(1 / (rp * 1e+3)) / float(1 / (rp * 1e+3)))  # Cp normalisée
    # R th dans la list
    Rth.append(float(r_th))
    cth.append((float(1 / r_th)) / float(1 / (rp * 1e+3)))
####
plt.figure(figsize=(L,l))
# plt.grid()
# Plot pour la résistance
color = 'tab:red'
plt.ylabel('C/Cp', fontsize=font)
plt.xlabel('# de Pulse', fontsize=font)
plt.plot(X0[:, 0], X0[:, 1], 'r.', X1[:, 0], X1[:, 1], 'k.', markersize=dot_plot)
#plt.plot(x_dp[:], Data_dep[:], 'r.', label='Dépression', markersize=dot_plot)
#plt.plot(X1[:, 0], Cp[:], 'c-', label='Cp', markersize=dot_plot)  # Plot Rp
#plt.plot(X1[:, 0], cth[:], 'b-', label='Cth=' + str(f"{(c_th/cp):.2f}") + ' Cp', markersize=dot_plot)  # Plot R th
plt.tick_params(axis='y', labelsize=labsize)
plt.yscale('log')
#plt.xscale('log')
plt.ylim([0.8, 70])
#plt.xlim([_a, _b])
#plt.xticks([])
#plt.grid(color='k', axis='y')
plt.title(name, fontsize=font)
plt.legend( fontsize=fontLEG)
plt.show()


#####
pot=[]
dep=[]
# Limits of x axis
_a = 0
_b = 33

for i in range(_a, _b):
    pot.append(X1[i][1])
    dep.append(X0[i][1])
###Sauvegarde data
DP = pd.DataFrame(pot, dtype=float)
DP.to_csv(""+dos+"/pot.csv", index=False)
DD = pd.DataFrame(dep, dtype=float)
DD.to_csv(""+dos+"/det.csv", index=False)
###Read data
Data_pet = pd.read_csv("" + dos + "/pot.csv")
Data_dep = pd.read_csv("" + dos + "/det.csv")

pt = len(Data_pet[:])
dp = len(Data_dep[:])
x_pt = np.linspace(1, pt, pt)
x_dp = np.linspace(pt, dp+pt, dp)
plt.figure(figsize=(L,l))
# plt.grid()
# Plot pour la résistance
color = 'tab:red'
plt.ylabel('C/Cp', fontsize=font)
plt.xlabel('# de Pulse', fontsize=font)
plt.plot(x_pt[:], Data_pet[:], 'b.',label='Potentiation', markersize=dot_plot)
plt.plot(x_dp[:], Data_dep[:], 'r.', label='Dépression', markersize=dot_plot)
#plt.plot(X1[:, 0], Cp[:], 'c-', label='Cp', markersize=dot_plot)  # Plot Rp
#plt.plot(X1[:, 0], cth[:], 'b-', label='Cth=' + str(f"{(c_th/cp):.2f}") + ' Cp', markersize=dot_plot)  # Plot R th
plt.tick_params(axis='y', labelsize=labsize)
plt.tick_params(axis='x', labelsize=labsize)
plt.yscale('log')
#plt.xscale('log')
plt.ylim([0.8, 100])
#plt.xlim([_a, _b])
#plt.xticks([])
#plt.grid(color='k', axis='y')
plt.title(name, fontsize=font)
plt.legend( fontsize=fontLEG)

plt.show()


####
plt.savefig("" + dos + "/" + name + ".svg")
