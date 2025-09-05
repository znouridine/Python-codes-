# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:10:34 2025

@author: nzemal
"""

globals().clear()
import numpy as np
from statistics import mean
import pandas as pd
from statistics import median
from statistics import stdev
import matplotlib.pyplot as plt
plt.ioff()

dos= str(20250122)
dot_plot = 20  # taille point plot
labsize = 20 # Taille graduation axe
font = 26 # Taille titres des axes
fontLEG = 20 # taille legende
l=8 #largeur fig
L=9 #longueur fig
name = input("Nom fichier :")
data = np.loadtxt(""+dos+"/" + name + ".NL", skiprows=4, usecols=range(0, 14))
c2 = data[:, 1]
c4 = data[:, 3]
t_ = int(len(c4))
k = 0
r = 0
########### Limits of y axis Data
_g = 800
_h = 90000

########### Limits of x axis Data
_a = 0
_b = 50
##############################################################" Nombre de cycle
_c = 1
_d = 41
# Garder un même nombre de pulse pour la comparaison.
l_b =0
l_h =t_  ## t_ SI TOUT LE PLOT

####Chargement fichier de données
data = np.loadtxt(""+dos+"/" + name + ".NL", skiprows=4, usecols=range(0, 14))
c2 = data[:, 1]
c4 = data[l_b:l_h, 3]
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



#####Taux de variation à partir duquel je compte les états intermédiaires
Var_s=0.001
Var_re=0.001


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

for na in range (min(len(a_s),len(b_s))):
    d_s.append(abs(a_s[na] - b_s[na]))
lez= max(d_s)

X2 = np.zeros((lez,len(a_s)*2))
xx = np.linspace(1, lez,lez)
xx_=[]

for g in range (min(len(a_s),len(b_s))):
    # Insertion première valeur
    c_s = []
    #cPul_s.append(X1[a_s[g]][0])
    X2[0:abs(b_s[g]-a_s[g]), g*2]=xx[0:abs(b_s[g]-a_s[g])]
    X2[0:abs(b_s[g]-a_s[g]), (g*2)+1]=X1[a_s[g]:b_s[g], 1]
    
   
    
    
#print(X2)
# SET
plt.figure(1, figsize= (L,l))
color = 'tab:red'
plt.ylabel('Resistance in Ohm', color=color,fontsize=font)
plt.xlabel('# Pulse', color=color,fontsize=font)
for pop in range(_c,_d):
    if pop >= len(a_s):
        pop=len (a_s)-1
        plt.plot(X2[:,pop*2], X2[:,(pop*2)+1], 'o-', markersize=dot_plot)
    plt.plot(X2[:,pop*2], X2[:,(pop*2)+1], 'o-', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color,labelsize=labsize)
plt.tick_params(axis='x', labelcolor=color,labelsize=labsize)
plt.yscale("log")
plt.ylim([_g, _h])
#plt.yticks(np.arange(_g, _h, step=1))
plt.xlim([_a, _b])
plt.xticks(np.arange(_a, _b, step=5))
plt.legend(labelcolor='linecolor',fontsize=font)
plt.grid(color='k', axis='y')
plt.title("SET",fontsize=font)
plt.show()

DF = pd.DataFrame(X2[:,(_c-1)*2:(_d-1)*2])
DF.to_csv(f"{dos}/SETraodDATA_{name}.csv", index=False, header=False)
#plt.savefig(""+dos+"/SETraodDATA_"+name+".svg")

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
for na in range (min(len(a_re),len(b_re))):
    d_re.append(abs(a_re[na] - b_re[na]))
lez= max(d_re)

X3 = np.zeros((lez,len(a_re)*2))
xxx = np.linspace(1, lez,lez)
xxx_=[]

for g in range (min(len(a_re),len(b_re))):
    X3[0:abs(b_re[g]-a_re[g]), g*2]=xxx[0:abs(b_re[g]-a_re[g])]
    X3[0:abs(b_re[g]-a_re[g]), (g*2)+1]=X0[a_re[g]:b_re[g], 1]

 

# Plot RESET
plt.figure(2, figsize= (L,l))
color = 'tab:red'
plt.ylabel('Resistance in Ohm', color=color,fontsize=font)
plt.xlabel('# Pulse', color=color,fontsize=font)
for pop in range(_c,_d):
    if pop >= len(a_re):
        pop=len (a_re)-1
        plt.plot(X3[:,pop*2], X3[:,(pop*2)+1], '1-', markersize=dot_plot)
    plt.plot(X3[:,pop*2], X3[:,(pop*2)+1], '1-', markersize=dot_plot)
plt.tick_params(axis='y', labelcolor=color,labelsize=labsize)
plt.tick_params(axis='x', labelcolor=color,labelsize=labsize)
plt.yscale("log")
plt.ylim([_g, _h])
#plt.yticks(np.arange(_g, _h, step=1))
plt.xlim([_a, _b])
plt.xticks(np.arange(_a, _b, step=5))
plt.legend(labelcolor='linecolor',fontsize=font)
plt.grid(color='k', axis='y')
plt.title("RESET",fontsize=font)
plt.show()
DF = pd.DataFrame(X3[:,(_c-1)*2:(_d-1)*2])
DF.to_csv(f"{dos}/RESETraodDATA_{name}.csv", index=False, header=False)
#plt.savefig(""+dos+"/RESETraodDATA_"+name+".svg") 