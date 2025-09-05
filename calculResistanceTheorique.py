# -*- coding: utf-8 -*-
"""
Created on Mon Sept 17 15:37:32 2024

@author: nouri
"""
globals().clear()
import numpy as np
from math import pi
import matplotlib.pyplot as plt

plt.ioff()
e = float(input('Epaisseur en nm: '))
z = float(input('Taille via en um: '))
Cr = float(input('% Chrome: '))
if Cr == 30:
    ro = 35  # unit 0hm.cm
else:
    if Cr == 5:
        ro = 1.2  # unit 0hm.cm

Z = (z / 2) * 1e-4
r_th = (ro * e * 1E-7) / (pi * Z * Z)
print(r_th)
