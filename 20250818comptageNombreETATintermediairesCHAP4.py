# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 21:02:27 2025

@author: nzemal
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
import matplotlib
plt.ioff()

dos= str(20250818)
dot_plot = 12  # taille point plot
labsize = 20 # Taille graduation axe
font = 12 # Taille titres des axes
fontLEG = 8 # taille legende
l=6 #largeur fig
L=9 #longueur fig
name = "Test cycle synapse Ligne011 PVDE66 Rp146k_01"
####Chargement fichier de données
data = np.loadtxt(""+dos+"/" + name + ".NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1]
c4 = data[0:1474, 3]
c9 = data[:, 8]
c10 = data[:, 9]
c12 = data[:, 11]
c11 = data[:, 10]
t = int(len(c4))
###Tracé des résisitance
A0 = []
X0 = np.zeros((int(t), 4))
X0_ = np.zeros((int(t), 4))
X1 = np.zeros((int(t), 4))
X1_ = np.zeros((int(t), 4))
k = 0
r = 0
X3 = []
X4 = []
########### Limits of y axis Data
_g = 0
_h = 2500

########### Limits of x axis Data
#_e = 0
#_f = 50
########### Egalisation du nombre de cycles - E.I
_a = 1023
_b = 1158
########### Egalisation Echelle des y - E.I
_c = 0
_d = 20

for i in range(t):
    # RESET conditions
    if c9[i] == 6E-9:
        X0[i][1] = c4[i]
        X0[i][2] = c12[i]
        X0[i][0] = i + 1
        X0[i][3] = i + 1
        # Set conditions
    else:
        if c9[i] != 0:
            X1[i][1] = c4[i]
            X1[i][2] = c12[i]
            X1[i][0] = i + 1
            X1[i][3] = i + 1

nb_set = []
ind_set = []
nb_reset = []
ind_reset = []
cPul_s=[]
cPul_re=[]
dPul_s=[]
dPul_re=[]
nbps = []
nbpre = []
nbt = []

#### Data brutes visualisation: Data brutes
plt.figure(1, figsize=(L/2.54,l/2.54))
color = 'k'
plt.ylabel(r"Résistance Log(k$\Omega$)", color=color,fontsize=font)
plt.xlabel('# Impulsions', color=color,fontsize=font)
plt.plot(X0[:, 0], X0[:, 1], 'r.', label='RESET', markersize= dot_plot)
plt.plot(X1[:, 0], X1[:, 1], 'k.', label='SET', markersize= dot_plot)
plt.yscale("log")
plt.tick_params(axis='both',
                direction='in',
               which='major',  # for major ticks
               length=5,
               width=1,
               labelsize=10)
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.subplots_adjust(left=0.15, right=0.97, top=0.99, bottom=0.2)
plt.tick_params(axis='both',
                direction='in',
               which='minor',  # for minor ticks
               length=3,
               width=1,
               labelsize=10,
               color='black')
#plt.ylim([_g, _h])
# plt.xlim([_a, _b])
#plt.grid(color='k', axis='y')
#plt.title(""+name+"",fontsize=font)
#plt.legend(loc='upper right', fontsize="6")
plt.savefig(""+dos+"/"+name+"BRUTE.svg")

#####Taux de variation à partir duquel je compte les états intermédiaires
Var_s=0.05
Var_re=0.05


a_s = []; a_re = []
b_s = []; b_re = []
c_s = []; c_re = []
d_s = []; d_re = []
#########################################Calcul E.I en SET####################################################
for i in range(t): #Isolation de chaque SET
    if (X1[i-1][1] == 0 and X1[i-2][1] == 0 and X1[i-3][1] == 0 and X1[i][1] != 0) or (
        i == 0 and X1[i][1] != 0
    ):
        a_s.append(i)
    if (i < t - 3 and X1[i + 1][1] == 0 and X1[i + 2][1] == 0 and X1[i + 3][1] == 0 and X1[i][1] != 0) or (
        i==t-1 and X1[i][1] != 0
    ):
        b_s.append(i)
for g in range (len(a_s)):
    # Insertion première valeur
    c_s = []
    cPul_s = []
    c_s.append(X1[a_s[g]][1])
    cPul_s.append(X1[a_s[g]][0])
    for j in range(a_s[g] , b_s[g]):
        #Conditions sur le taux de variation en set
        if j < t - 1 and X1[j + 1][1] != 0 and X1[j][1] != 0 and ((X1[j][1] - X1[j+ 1][1]) / X1[j][1]) >= Var_s:

            c_s.append(X1[j+1][1])
            cPul_s.append(X1[j + 1][0])
    if len(c_s)> 1:
       
        # insertion première valeur
        dPul_s = []
        d_s = []
        d_s.append(c_s[0])
        dPul_s.append(cPul_s[0])
        for h in range(1, len(c_s)):#Eliminer les SET qui fond des RESET

            #Conditions sur le taux de variation en SET
            if (-c_s[h] +max(c_s[:h]))/max(c_s[:h]) >= Var_s:
                d_s.append(c_s[h])
                dPul_s.append(cPul_s[h])


        plt.figure(1, figsize=(L/2.54,l/2.54))
        plt.plot(dPul_s[:], d_s[:], 'g1', label='SET', markersize=dot_plot+1)
        color = 'k'
        plt.ylabel(r"Résistance Log(k$\Omega$)", color=color, fontsize=font)
        plt.xlabel('# Impulsions', color=color, fontsize=font)
        plt.yscale("log")
        plt.tick_params(axis='both',
                        direction='in',
                       which='major',  # for major ticks
                       length=5,
                       width=1,
                       labelsize=10)
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        plt.subplots_adjust(left=0.15, right=0.97, top=0.99, bottom=0.2)
        plt.tick_params(axis='both',
                        direction='in',
                       which='minor',  # for minor ticks
                       length=3,
                       width=1,
                       labelsize=10,
                       color='black')
        #plt.ylim([_g, _h])
        # plt.xlim([_a, _b])
        # plt.xticks([])
        #plt.grid(color='k', axis='y')
        #plt.title("" + name + " Traité", fontsize=font)
        #plt.savefig("" + dos + "/" + name + "traité.svg")
    
        nb_set.append(len(d_s))
        c_s=[]
        cPul_s = []
        dPul_s = []
        d_s = []
    elif len(c_s)<= 1:
        nb_set.append(2)
        nbps.append(2)
        
        c_s = []
        d_s = []
        cPul_s = []
        dPul_s = []

##############################################Calcul E.I en RESET################################################
for i in range(t):
    if (X0[i - 1][1] == 0 and X0[i - 2][1] == 0 and X0[i - 3][1] == 0 and X0[i][1] != 0) or (
            i == 0 and X0[i][1] != 0
    ):
        a_re.append(i)
    if (i < t - 3 and X0[i + 1][1] == 0 and X0[i + 2][1] == 0 and X0[i + 3][1] == 0 and X0[i][1] != 0) or (
             X0[i][1] != 0 and i==t-1
    ):
        b_re.append(i)
for g in range(len(a_re)):
    # Insertion première valeur
    c_re = []
    cPul_re = []
    c_re.append(X0[a_re[g]][1])
    cPul_re.append(X0[a_re[g]][0])
    for j in range(a_re[g], b_re[g]):
        #Conditions sur la variation entre deux points de conductance successifs.
        if j < t - 1 and X0[j + 1][1] != 0 and X0[j][1] != 0 and ((X0[j+1][1] -X0[j][1]) / X0[j+1][1]) >= Var_re:

            c_re.append(X0[j + 1][1])
            cPul_re.append(X0[j + 1][0])
        
    if len(c_re) > 1:
        # insertion première valeur
        dPul_re = []
        d_re = []
        d_re.append(c_re[0])
        dPul_re.append(cPul_re[0])
        for h in range(1, len(c_re)): # Elimination des pulses qui fond des SET

            # Conditions sur le taux de variation
            if (-min(c_re[:h])+c_re[h]) / c_re[h] >= Var_re:

                d_re.append(c_re[h])
                dPul_re.append(cPul_re[h])


        plt.figure(1, figsize=(L/2.54,l/2.54))
        plt.plot(dPul_re[:], d_re[:], 'g1', label='RESET', markersize=dot_plot+4)
        color = 'k'
        plt.ylabel(r"Résistance Log($\Omega$)", color=color, fontsize=font)
        plt.xlabel('# Impulsions', color=color, fontsize=font)
        plt.yscale("log")
        plt.xlim([_a, _b])
        plt.tick_params(axis='both',
                        direction='in',
                       which='major',  # for major ticks
                       length=5,
                       width=1,
                       labelsize=10)
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        plt.subplots_adjust(left=0.15, right=0.97, top=0.99, bottom=0.2)
        plt.tick_params(axis='both',
                        direction='in',
                       which='minor',  # for minor ticks
                       length=3,
                       width=1,
                       labelsize=10,
                       color='black')
        #plt.grid(color='k', axis='y')
        #plt.title("" + name + "traité", fontsize=font)
        plt.savefig("" + dos + "/" + name + "traité.svg")
        
        nb_reset.append(len(d_re))
        c_re = []
        cPul_re = []
        dPul_re = []
        d_re = []
    else:
        nb_reset.append(2)
        nbpre.append(2)
       
        c_re = []
        d_re = []
        cPul_re = []
        dPul_re = []

plt.show()