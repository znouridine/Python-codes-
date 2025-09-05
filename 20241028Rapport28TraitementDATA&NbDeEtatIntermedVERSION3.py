# -*- coding: utf-8 -*-
"""
Created on MON Oct 23 16:37:32 2024
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
plt.ioff()

dos= str(20241108)
dot_plot = 12  # taille point plot
name = input("Nom fichier :")
####Chargement fichier de données
data = np.loadtxt(""+dos+"/" + name + ".NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1]
c4 = (1 / data[:, 3]) * 1e+6
c9 = data[:, 8]
c10 = data[:, 9]
c12 = data[:, 11]
c11 = data[:, 10]
t = int(len(c4))
###Tracé des résisitance
t = (len(c4))
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
        X0[i][0] = i + 1
        X0[i][3] = i + 1
        # Set conditions
    else:
        if c9[i] != 0:
            X1[i][1] = c4[i]
            X1[i][2] = c12[i]
            X1[i][0] = i + 1
            X1[i][3] = i + 1

n = 2
p = 2
nb_set = []
ind_set = []
nb_reset = []
ind_reset = []
cPul_s=[]
cPul_re=[]
nbps = []
nbpre = []
nbt = []

#### Data brutes visualisation
plt.figure()
color = 'tab:red'
plt.ylabel('Conductance in 10^(-6) S', color=color)
plt.xlabel('# Pulses', color=color)
plt.plot(X0[:, 0], X0[:, 1], 'r.',
         X1[:, 0], X1[:, 1], 'k.')
plt.tick_params(axis='y', labelcolor=color)
plt.tick_params(axis='x', labelcolor=color)
plt.yscale("linear")
plt.ylim([1e+6 / 100000, 1e+6 / 500])
#_a = 425
#_b = 1500  # Limits of x axis
# plt.xlim([_a, _b])
# plt.xticks([])
plt.grid(color='k', axis='y')
plt.title('Données brutes')
# plt.legend(loc='upper right', fontsize="6")


#####Taux de variation à partir duquel je compte les états intermédiaires
Var_s=0.1
Var_re=0.1
## Egalisation du nombre de cycles
_a = 0
_b = 18
## Egalisation Echelle des y
_c = 0
_d = 30

a_s=[]; a_re=[]
b_s=[]; b_re=[]
c_s=[]; c_re=[]
d_s = []; d_re = []
####Calcul E.I en SET
for i in range(t):
    if (X1[i-1][1] == 0 and X1[i-2][1] == 0 and X1[i-3][1] == 0 and X1[i][1] != 0) or (
        i == 0 and X1[i][1] != 0
    ):
        a_s.append(i)
    if (i < t - 3 and X1[i + 1][1] == 0 and X1[i + 2][1] == 0 and X1[i + 3][1] == 0 and X1[i][1] != 0) or (
        i==t-1 and X1[i][1] != 0
    ):
        b_s.append(i)
for g in range (len(a_s)):
    for j in range(a_s[g] , b_s[g]):
        if j < t - 1 and X1[j + 1][1] != 0 and X1[j][1] != 0 and ((X1[j + 1][1] - X1[j][1]) / X1[j + 1][1]) >= Var_s:
            c_s.append(X1[j+1][1])
            cPul_s.append(X1[j+1][0])
    if len(c_s)> 1:
        # nb de pulse total pour le SET
        p1 = cPul_s[0]
        p2 = cPul_s[-1]
        pu = abs(p1 - p2) + 1
        nbps.append(int(pu))
        for h in range(1, len(c_s)):#Eliminer les SET qui fond des RESET
            if (c_s[h] -max(c_s[:h]))/c_s[h] >= Var_s:
                n=n+1
                d_s.append(c_s[h])
                if max(c_s[:h]) not in d_s: #Avec la première valeur
                    d_s.insert(d_s.index(c_s[h])-1, max(c_s[:h]))
        print('\n', 'cycle set', g + 1)
        print('\n', c_s)
        print(d_s)
        print(len(d_s))
        print('\n')
        nb_set.append(n)
        n=2
        c_s=[]
        cPul_s = []
        d_s = []
    elif len(c_s)<= 1:
        nb_set.append(2)
        nbps.append(2)
        print('\n', 'cycle set', g + 1)
        print('\n', c_s)
        print(d_s)
        print(len(d_s))
        print('\n')
        n = 2
        c_s = []
        d_s = []

####Calcul E.I en RESET
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
    for j in range(a_re[g], b_re[g]):
        if j < t - 1 and X0[j + 1][1] != 0 and X0[j][1] != 0 and ((X0[j][1] -X0[j + 1][1]) / X0[j][1]) >= Var_re:
            c_re.append(X0[j + 1][1])
            cPul_re.append(X0[j + 1][0])
    if len(c_re) > 1:
        # nb de pulse total pour le RESET
        p1 = cPul_re[0]
        p2 = cPul_re[-1]
        pu = abs(p1 - p2) + 1
        nbpre.append(int(pu))
        for h in range(1, len(c_re)):
            if (min(c_re[:h])-c_re[h]) / min(c_re[:h]) >= Var_re:
                p = p + 1
                d_re.append(c_re[h])
                if min(c_re[:h]) not in d_re:
                    d_re.insert(d_re.index(c_re[h]) - 1, min(c_re[:h]))
        print('\n', 'cycle reset', g + 1)
        print('\n', c_re)
        print(d_re)
        print(len(d_re))
        print('\n')
        nb_reset.append(p)
        p = 2
        c_re = []
        cPul_re = []
        d_re = []
    else:
        nb_reset.append(2)
        nbpre.append(2)
        print('\n', 'cycle reset', g + 1)
        print('\n', c_re)
        print(d_re)
        print(len(d_re))
        print('\n')
        c_re = []
        d_re = []

#### Pulse totals
for c in range (min(len(nb_set),len(nb_reset))):
    if c < min(len(nb_set),len(nb_reset)): nbt.append(nb_set[c]+nb_reset[c])
#Veirification du traitement
print(nb_set)
print(b_s)
print(len(nb_set))
print(nb_reset)
print(b_re)
print(len(nb_reset))
print(nbps)
print(nbpre)
print(nbt)
# Limits of x axis
nb_set=nb_set[:_b]; nb_reset=nb_reset[:_b]; nbt=nbt[:_b]
#### Statistics
#Moyenne
m_s=mean(nb_set)
m_re=mean(nb_reset)
mt=mean(nbt)
#Médiane
med_s=median(nb_set)
med_re=median(nb_reset)
medt=median(nbt)
#Ecart-type
e_t_s=stdev(nb_set)
e_t_re=stdev(nb_reset)
e_t_t=stdev(nbt)
### Visualisation
# Sans les barre d'erreur
plt.figure(figsize=(8, 6))
color = 'tab:red'
plt.ylabel('# ETAT INTERMEDIAIRE', color=color,fontsize="18")
plt.xlabel('N° Cycle', color=color,fontsize="18")
pulse_s=np.linspace(1,len(nb_set),len(nb_set))
pulse_m=np.linspace(_a,_b,_b)
pulse_re=np.linspace(1,len(nb_reset),len(nb_reset))
pulse_t=np.linspace(1,len(nbt),len(nbt))
plt.plot(pulse_s[:], nb_set[:], 'k*', label='SET Med='+str(round(med_s,2))+' Moy='+str(round(m_s,2))+' Sd='+str(round(e_t_s,2))+' ('+str(round(e_t_s*100/m_s,2))+'%) ', markersize=dot_plot)
plt.plot(pulse_m, np.linspace(m_s,m_s,_b),'k--' , markersize=dot_plot)
plt.plot(pulse_re[:], nb_reset[:], 'r*', label='RESET Med='+str(round(med_re,2))+' Moy='+str(round(m_re,2))+' Sd='+str(round(e_t_re,2))+' ('+str(round(e_t_re*100/m_re,2))+'%) ', markersize=dot_plot)
plt.plot(pulse_m, np.linspace(m_re,m_re,_b),'r--' , markersize=dot_plot)
plt.plot(pulse_t[:], nbt[:], 'b*', label='Sum Med='+str(round(medt,2))+' Moy='+str(round(mt,2))+' Sd='+str(round(e_t_t,2))+' ('+str(round(e_t_t*100/mt,2))+'%) ', markersize=dot_plot)
plt.plot(pulse_m, np.linspace(mt,mt,_b),'b--' , markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color,labelsize=18)
plt.tick_params(axis='x', labelcolor=color,labelsize=18)
plt.yscale("linear")
plt.ylim([_c, _d])
plt.yticks(np.arange(_c, _d, step=4))
plt.xlim([_a, _b])
plt.xticks(np.arange(_a, _b, step=2))
plt.legend(labelcolor='linecolor')
plt.grid(color='k', axis='y')
plt.title(name,fontsize="20")
plt.savefig(""+dos+"/"+name+"SB.svg")

# Avec les barre d'erreur
plt.figure(figsize=(8, 6))
color = 'tab:red'
plt.ylabel('# ETAT INTERMEDIAIRE', color=color,fontsize="18")
plt.xlabel('N° Cycle', color=color,fontsize="18")
pulse_s=np.linspace(1,len(nb_set),len(nb_set))
pulse_m=np.linspace(_a,_b,_b)
pulse_re=np.linspace(1,len(nb_reset),len(nb_reset))
pulse_t=np.linspace(1,len(nbt),len(nbt))
plt.errorbar(pulse_s[:], nb_set[:], label='SET Med='+str(round(med_s,2))+' Moy='+str(round(m_s,2))+' Sd='+str(round(e_t_s,2))+' ('+str(round(e_t_s*100/m_s,2))+'%) ', yerr=e_t_s, fmt='k*',
                 ms=dot_plot, ecolor="k", elinewidth=2, capsize=2, capthick=2)
plt.plot(pulse_m, np.linspace(m_s,m_s,_b),'k--' , markersize=dot_plot)
#plt.plot(pulse_re[:], nb_reset[:], 'r*', label='RESET Med='+str(round(med_re,2))+' Moy='+str(round(m_re,2))+' Sd='+str(round(e_t_re,2))+' ('+str(round(e_t_re*100/m_re,2))+'%) ', markersize=dot_plot)
plt.errorbar(pulse_re[:], nb_reset[:], label='RESET Med='+str(round(med_re,2))+' Moy='+str(round(m_re,2))+' Sd='+str(round(e_t_re,2))+' ('+str(round(e_t_re*100/m_re,2))+'%) ', yerr=e_t_re, fmt='r*',
                 ms=dot_plot, ecolor="r", elinewidth=2, capsize=5, capthick=2)
plt.plot(pulse_m, np.linspace(m_re,m_re,_b),'r--' , markersize=dot_plot)
#plt.plot(pulse_t[:], nbt[:], 'b*', label='Sum Med='+str(round(medt,2))+' Moy='+str(round(mt,2))+' Sd='+str(round(e_t_t,2))+' ('+str(round(e_t_t*100/mt,2))+'%) ', markersize=dot_plot)
plt.errorbar(pulse_t[:], nbt[:], label='Sum Med='+str(round(medt,2))+' Moy='+str(round(mt,2))+' Sd='+str(round(e_t_t,2))+' ('+str(round(e_t_t*100/mt,2))+'%) ', yerr=e_t_t, fmt='b*',
                 ms=dot_plot, ecolor="b", elinewidth=2, capsize=8, capthick=2)
plt.plot(pulse_m, np.linspace(mt,mt,_b),'b--' , markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color,labelsize=18)
plt.tick_params(axis='x', labelcolor=color,labelsize=18)
plt.yscale("linear")
plt.ylim([_c, _d])
plt.yticks(np.arange(_c, _d, step=4))
plt.xlim([_a, _b])
plt.xticks(np.arange(_a, _b, step=2))
plt.legend(labelcolor='linecolor')
plt.grid(color='k', axis='y')
plt.title(name,fontsize="20")
plt.savefig(""+dos+"/"+name+"AB.svg")
plt.show()

