# -*- coding: utf-8 -*-
"""
Created on Wed July 16 15:37:32 2025

@author: nouri
"""
globals().clear()
import numpy as np
from math import pi
import matplotlib.pyplot as plt

plt.ioff()
name = input("Nom fichier :")
rp = float(input('Résistance pristine: '))
e = float(50)
Via = float(1)
Cr = float(30)
data = np.loadtxt("20250122/" + name + ".NL", skiprows=3, usecols=range(0, 14))
c2 = data[:, 1];
c4 = data[:, 3];
c9 = data[:, 8];
c10 = data[:, 9]
c12 = data[:, 11];
c11 = data[:, 10]
t = len(c4)
A0 = []
X0 = np.zeros((int(t), 5))
X1 = np.zeros((int(t), 5))
X3 = []
X4 = []
k = 0
r = 0
dot_plot = 12  # taille point plot


_a = 1;
_b = 52  # Limits of x axis


fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 5), height_ratios=[1, 1])
for i in range(t):
    # RESET conditions
    if c9[i] == 6E-9:  # and c10[i]==30E-9 and c11[i]==6E-9 :
        X0[i][1] = c4[i]/2000
        X0[i][2] = c12[i]
        X0[i][4] = int((c9[i] + c10[i] + c11[i]) * 10 ** 9)
        X0[i][0] = i
        X0[i][3] = i
    # Set conditions
    else:
        if c9[i] != 0:
            X1[i][1] = c4[i]/20000
            X1[i][2] = c12[i]
            X1[i][4] = int((c9[i] + c10[i] + c11[i]) * 10 ** 9)
            X1[i][0] = i
            X1[i][3] = i

Rp = []
Rth = []
VIA = Via/2
if Cr == 30:
    ro = 35
else:
    if Cr == 5:
        ro = 1.3
r_th = (ro * e * 1E-7) / (pi * VIA*1e-4 * VIA*1e-4)  # La résistivité du 30% Cr est 35 Ohm.cm
for t in range(len(c12)):
    Rp.append(float(rp * 1e+3))  # Mettre la valeur de
    # R th dans la list
    Rth.append(float(r_th))
    
    
    
a_s = []; a_re = []
b_s = []; b_re = [0]
c_s = []; c_re = []
d_s = []; d_re = []    
    
#================================================================================
for i in range(t): #Isolation de chaque SET
    if (X1[i-1][1] == 0 and X1[i-2][1] == 0 and X1[i-3][1] == 0 and X1[i][1] != 0) or (
        i == 0 and X1[i][1] != 0
    ):
        a_s.append(i)
    if (i < t - 3 and X1[i + 1][1] == 0 and X1[i + 2][1] == 0 and X1[i + 3][1] == 0 and X1[i][1] != 0) or (
        i==t-1 and X1[i][1] != 0
    ):
        b_s.append(i)  

#================================================================================
for i in range(t):
    if (X0[i - 1][1] == 0 and X0[i - 2][1] == 0 and X0[i - 3][1] == 0 and X0[i][1] != 0) or (
            i == 0 and X0[i][1] != 0
    ):
        a_re.append(i)
    if (i < t - 3 and X0[i + 1][1] == 0 and X0[i + 2][1] == 0 and X0[i + 3][1] == 0 and X0[i][1] != 0) or (
             X0[i][1] != 0 and i==t-1
    ):
        b_re.append(i)

#calcul du nombre de cycles en reset
for q in range(1,len(b_re)):
    for j in range(1,abs(b_re[q]-b_re[q-1])+1):
        X3.append(q+j/abs(b_re[q]-b_re[q-1]))
        
#calcul du nombre de cycles en set
for q in range(1,len(b_s)):
    for j in range(1,abs(b_s[q]-b_s[q-1])+1):
        X4.append(q+j/abs(b_s[q]-b_s[q-1]))


# plt.grid()
# Plot SET
color = 'tab:red'
ax1.set_ylabel('Resistance(ohm)', fontsize="18", color=color)
ax1.plot( X1[:, 0], X1[:, 1], 'k.',
         markersize=dot_plot)
#ax1.plot(X1[:, 0], Rp[:], 'c-', label='Rp=' + str(int(rp)) + 'k', markersize=dot_plot)  # Plot Rp
#ax1.plot(X1[:, 0], Rth[:], 'b-', label='Rth=' + str(int(r_th * 1e-3)) + ' k', markersize=dot_plot)  # Plot R th
ax1.tick_params(axis='y', labelcolor=color, labelsize=15)
ax1.set_yscale('log')
ax1.set_ylim([0.03, 3])

#ax1.set_xlim([_a,_b])
ax1.set_xticks([])
ax1.grid(color='k', axis='y')
#ax1.title(name, fontsize="14")
ax1.legend(loc='upper right', fontsize="12")
#Plot RESET
color = 'tab:red'
ax2.set_ylabel('Resistance(ohm)', fontsize="18", color=color)
ax2.plot(X0[:, 0], X0[:, 1], 'r.',
         markersize=dot_plot)  # Plot RESET

ax2.set_ylim([0, 12])
ax2.set_yticks([1, 5, 10, 12])
#ax2.set_xlim([_a,_b])
ax2.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
ax2.set_xlabel('Number of pulse', fontsize="20")
fig.tight_layout()
'''
# Second axe nombre de cycles
ax3 = ax1.twiny()
ax3.set_xlabel('X axis 2 (top)')

ax3.set_ylabel('Time(ns)', fontsize="18", color=color)
ax3.plot(X3[:], X0[:len(X3), 1], 'k.', markersize=dot_plot)  # Plot temps de pulse
#ax3.set_yscale('linear')
ax3.set_xticks(np.arange(0, len(b_re), step=25))
#ax3.set_ylim([10, 250])
ax3.set_xlim([75,100])
ax3.set_xlabel('Number of cycles', fontsize="20")
#ax3.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
#ax3.tick_params(axis='y', labelcolor=color, labelsize=15)
ax3.tick_params(axis='x', labelsize=15)
fig.tight_layout()
'''
plt.savefig("20250122/" + name + ".svg")
plt.show()

figg, (ax1, ax2) = plt.subplots(2, figsize=(8, 5), height_ratios=[1, 1])

# plt.grid()
# Plot SET
color = 'tab:red'
ax1.set_ylabel('Resistance(ohm)', fontsize="18", color=color)
ax1.plot( X4[:], X1[:len(X4), 1], 'k.',
         markersize=dot_plot)
ax1.set_xticks(np.arange(0, len(b_s), step=20))
#ax1.plot(X1[:, 0], Rp[:], 'c-', label='Rp=' + str(int(rp)) + 'k', markersize=dot_plot)  # Plot Rp
#ax1.plot(X1[:, 0], Rth[:], 'b-', label='Rth=' + str(int(r_th * 1e-3)) + ' k', markersize=dot_plot)  # Plot R th
ax1.tick_params(axis='y', labelcolor=color, labelsize=15)
ax1.set_yscale('log')
ax1.set_ylim([0.03, 3])



#ax1.set_xlim([_a,_b])
#ax1.set_xticks([])
ax1.grid(color='k', axis='y')
#ax1.title(name, fontsize="14")
ax1.legend(loc='upper right', fontsize="12")
#Plot RESET
color = 'tab:red'
ax2.set_ylabel('Resistance(ohm)', fontsize="18", color=color)
ax2.plot(X3[:], X0[:len(X3), 1], 'r.',
         markersize=dot_plot)  # Plot RESET
ax2.set_xticks(np.arange(0, len(b_re), step=20))
ax2.set_yscale('log')
ax2.set_ylim([0.5, 20])
ax2.set_yticks([1,10])




#ax2.set_xlim([_a,_b])
ax2.grid(color='b', linestyle='--', linewidth=0.7, axis='y')
ax2.tick_params(axis='y', labelcolor=color, labelsize=15)
ax2.set_xlabel('Number of cycle', fontsize="20")
figg.tight_layout()

plt.savefig("20250122/" + name + ".svg")
plt.show()