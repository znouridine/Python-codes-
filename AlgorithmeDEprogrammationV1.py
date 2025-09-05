# -*- coding: utf-8 -*-
"""
Created on Mon October 14 09:37:32 2024

@author: nouri
"""
globals().clear()
import numpy as np
from math import pi
import matplotlib.pyplot as plt

###################Initialisation des paramètres
# Fenêtre mémoire
hrs = 1
lrs = 1
# Condition de pulse de SET
U_set = np.zeros()
t_up_set = np.zeros()
t_on_set = np.zeros()
t_down_set = np.zeros()
# Condition de pulse de RESET
U_reset = np.zeros()
t_up_reset = np.zeros()
t_on_reset = np.zeros()
t_down_reset = np.zeros()
######Fichier conditions de pulses SET & RESET


# Allocation de bandes de résistance
nb = 4  # Le nombre de Banbe de résistance
R_inf = 1
R_sup = 1
# Mesure bas niveau Rp
Rp_mes = 1  # Valeur fournie par le SMU
# Initialisation R_mes
R_mes = Rp_mes
# Initialisation compteur
i = -1  # Compteur SET
j = -1  # Compteur RESET
####################Nombre d'itération totale avant échec de l'algo = 5 Pour une bande
for b in range(nb): #Boucle sur le nombre de bande
    for iteration in range(4):
        while R_mes >= R_sup: # Mettre R_sup et R_inf dans une matrice
            # Incrémentation compteur
            i = i + 1
            # Applique le pulse de SET
            U_set = U_set[i]
            t_up_set = t_up_set[i]
            t_on_set = t_on_set[i]
            t_down_set = t_down_set[i]
            # Mesure SMU de R_mes

            if R_mes <= R_inf:
                ###Continue d'appliquer des set jusqu'à lrs
                while lrs - 1 <= R_mes <= lrs + 1:
                    # Incrémentation compteur
                    i = i + 1
                    # Applique le pulse de SET
                    U_set = U_set[i]
                    t_up_set = t_up_set[i]
                    t_on_set = t_on_set[i]
                    t_down_set = t_down_set[i]
                    # Mesure SMU de R_mes
                while R_mes <= R_inf:
                    j = j + 1
                    U_reset = U_reset[j]
                    t_up_reset = t_up_reset[j]
                    t_on_reset = t_on_reset[j]
                    t_down_reset = t_down_reset[j]

            # Mesure SMU de R_mes

            elif R_inf <= R_mes <= R_sup:
                break
                print('Prgrammation réussie !')
            elif R_mes >= R_sup:
                ###Continue d'appliquer des reset jusqu'à HRS
                while hrs - 1 <= R_mes <= hrs + 1:
                    j = j + 1
                    U_reset = U_reset[j]
                    t_up_reset = t_up_reset[j]
                    t_on_reset = t_on_reset[j]
                    t_down_reset = t_down_reset[j]
                continue
        if iteration == 4 and R_mes < R_inf or R_mes > R_sup:
            print('La programmation a échouée')
