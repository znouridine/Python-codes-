import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from statistics import stdev
from math import pi
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

dot_plot = 12
# Chargement de fichier des valeurs préstine
Rp = np.zeros((20, 6))
X = np.linspace(1, 20, 20)
name = input("Nom échantillon:")
e = float(input('Epaisseur dépôt en nm: '))
via = float(input('A tape 1, B tape 2, C tape 3, D tape 4 : '))
text = ('N°      Rp(Kohm)          V 1er Set    Vreset       R 1er Set       R reset')
text1 = ("Les lignes suivantes sont exploitables: ")
text2 = ("La valeur moyenne est: ")
text3 = ("Ecart-type= ")
text4 = ("La valeur moyenne (sans valeur abérrante) est: ")
text5 = ("Ecart-type(sans valeur abérrante)=: ")
with open("Résistance pristine/" + name + ".txt", "w") as f:
    f.write(text)
    f.write('\n')
    for t in range(20):
        Rp[t][0] = t + 1
        Rp[t][1] = input("Résistance pristine ligne" + str(t + 1) + ":")
        Rp[t][2] = input("Résistance pristine relaxée ligne" + str(t + 1) + ":")
        """
        Rp[t][3] = float(input("V ReSet ligne" + str(t + 1) + ":"))
        Rp[t][4] = float(input("R premier Set ligne" + str(t + 1) + ":"))
        Rp[t][5] = float(input("R premier ReSet ligne" + str(t + 1) + ":"))
        """
        f.write(str(Rp[t][0]))
        f.write('\t')
        f.write('\t')
        f.write(str(Rp[t][1]))
        f.write('\t')
        f.write('\t')
        f.write(str(Rp[t][2]))
        f.write('\t')
        f.write('\t')
        f.write(str(Rp[t][3]))
        f.write('\t')
        f.write('\t')
        f.write(str(Rp[t][4]))
        f.write('\t')
        f.write('\t')
        f.write(str(Rp[t][5]))
        f.write('\n')

data = np.loadtxt("Résistance pristine/" + name + ".txt", skiprows=1)
a = data[:, 0]
b = data[:, 1]
c = data[:, 2]
d = data[:, 3]
f = data[:, 4]
g = data[:, 5]
# Calcul de la resistance théorique + incertitute
r_th_A = 1e-3 * (35 * e * 1e-7) / (pi * 1e-4 * 1e-4)
r_th_B = 1e-3 * (35 * e * 1E-7) / (pi * 0.6e-4 * 0.6e-4)
r_th_C = 1e-3 * (35 * e * 1E-7) / (pi * 0.4e-4 * 0.4e-4)
r_th_D = 1e-3 * (35 * e * 1E-7) / (pi * 0.25e-4 * 0.25e-4)

# Moyenne des valeurs des résistance préstine

k = mean(b)
# Taille A
if via == 1:
    val_inf = r_th_A - (r_th_A / 2)
    val_sup = r_th_A + (r_th_A / 2)
    g = list();
    s = list()
    print("La moyenne des résistance pristine est : ", k)
    for h in range(20):
        if val_inf < b[h] < val_sup:
            g.append(b[h])
    q = mean(g)
    print("La moyenne (sans valeurs abérrantes) des résistances pristines est : ", str(q))
    # Calcul de l'écart type
    with open("Résistance pristine/" + name + ".txt", "a") as f:
        f.write(text1)
        f.write('\n')
        f.write('\n')
        f.write(text)
        f.write('\n')
        sigma = stdev(b)
        print("L", "'", "écart-type sigma", "est égale à : ", str(sigma))
        for d in range(20):
            if val_inf < b[d] < val_sup:
                s.append(b[d])
        sigma_prime = stdev(s)
        print("Ecart-type (sans valeurs abérrantes) des résistance pristine est : ", str(sigma_prime))
    """
    #Selection des lignes
        Rt=80
        Rp_inf=Rt-Rt/2
        Rp_sup=Rt+Rt/2
        b=np.zeros((20,2))
        j=0

        for i in range(20):

            b[i][1]=l[i]
            b[i][0]=a[i]
            if Rp_inf<b[i][1]<Rp_sup: 
                    j=j+1
                    print('ligne=', str(b[i][0]), 'Rp=',str(b[i][1]),'K ohm' )
                    f.write(str(b[i][0]))
                    f.write('\t')
                    f.write('\t')
                    f.write('\t')
                    f.write('\t')
                    f.write(str(b[i][1]))
                    f.write('\n')
        print(j, "lignes sont exploitables sur la puce",name)
    with open("Résistance pristine/"+name+".txt", "a") as f: 
            f.write(text2)
            f.write(str(k))
            f.write('\n')
            f.write(text4)
            f.write(str(q))
            f.write('\n')
            f.write(text3)
            f.write(str(sigma))
            f.write('\n')
            f.write(text5)
            f.write(str(sigma_prime))
            f.write('\n')
"""

    plt.figure()
    for t in range(20):
        plt.plot(X[t], Rp[t][1], 'r.')
        plt.plot(X[t], r_th_A, 'c_')
        plt.title(name)
        plt.xlabel("N° ligne")
        plt.ylabel("Résistance (K ohm)")
    plt.xlim([0, 21])
    # plt.ylim([0,100])
    plt.grid()
    plt.yscale('log')
    plt.xticks([0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
                10, 11, 12, 13,
                14, 15, 16, 17, 19, 20])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# Taille B

if via == 2:
    val_inf = r_th_B - (r_th_B / 2)
    val_sup = r_th_B + (r_th_B / 2)
    g = list();
    s = list()
    print("La moyenne des résistance pristine est : ", k)
    """
    for h in range(20):
        if val_inf < b[h] < val_sup:
            g.append(b[h])
    q = mean(g)
    print("La moyenne (sans valeurs abérrantes) des résistances pristines est : ", str(q))
    """
    # Calcul de l'écart type
    with open("Résistance pristine/" + name + ".txt", "a") as f:
        f.write(text1)
        f.write('\n')
        f.write('\n')
        f.write(text)
        f.write('\n')
        sigma = stdev(b)
        print("L", "'", "écart-type sigma", "est égale à : ", str(sigma))
        """
        for d in range(20):
            if val_inf < b[d] < val_sup:
                s.append(b[d])
        sigma_prime = stdev(s)
        print("Ecart-type (sans valeurs abérrantes) des résistance pristine est : ", str(sigma_prime))

    #Selection des lignes
        Rt=80
        Rp_inf=Rt-Rt/2
        Rp_sup=Rt+Rt/2
        b=np.zeros((20,2))
        j=0

        for i in range(20):

            b[i][1]=l[i]
            b[i][0]=a[i]
            if Rp_inf<b[i][1]<Rp_sup: 
                    j=j+1
                    print('ligne=', str(b[i][0]), 'Rp=',str(b[i][1]),'K ohm' )
                    f.write(str(b[i][0]))
                    f.write('\t')
                    f.write('\t')
                    f.write('\t')
                    f.write('\t')
                    f.write(str(b[i][1]))
                    f.write('\n')
        print(j, "lignes sont exploitables sur la puce",name)
    with open("Résistance pristine/"+name+".txt", "a") as f: 
            f.write(text2)
            f.write(str(k))
            f.write('\n')
            f.write(text4)
            f.write(str(q))
            f.write('\n')
            f.write(text3)
            f.write(str(sigma))
            f.write('\n')
            f.write(text5)
            f.write(str(sigma_prime))
            f.write('\n')
"""

    plt.figure()
    for t in range(20):
        plt.plot(X[t], Rp[t][1], 'r.', markersize=dot_plot)
        plt.plot(X[t], r_th_B, 'c_', markersize=dot_plot)
        plt.title(name, fontsize="20")
        plt.xlabel("N° ligne", fontsize="16")
        plt.ylabel("Résistance (K ohm)", fontsize="16")
    plt.xlim([0, 21])
    plt.ylim([0, 100])
    plt.grid()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], fontsize="14")
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize="14")
    plt.savefig("20240925/" + name + ".svg")
    # plt.show()
# Taille C
if via == 3:
    val_inf = r_th_C - (r_th_C / 2)
    val_sup = r_th_C + (r_th_C / 2)
    g = list();
    s = list()
    print("La moyenne des résistance pristine est : ", k)
    """
        for h in range(20):
            if val_inf < b[h] < val_sup:
                g.append(b[h])
        q = mean(g)
        print("La moyenne (sans valeurs abérrantes) des résistances pristines est : ", str(q))
        """
    # Calcul de l'écart type
    with open("Résistance pristine/" + name + ".txt", "a") as f:
        f.write(text1)
        f.write('\n')
        f.write('\n')
        f.write(text)
        f.write('\n')
        sigma = stdev(b)
        print("L", "'", "écart-type sigma", "est égale à : ", str(sigma))
        """
        for d in range(20):
            if val_inf < b[d] < val_sup:
                s.append(b[d])
        sigma_prime = stdev(s)
        print("Ecart-type (sans valeurs abérrantes) des résistance pristine est : ", str(sigma_prime))

    #Selection des lignes
        Rt=80
        Rp_inf=Rt-Rt/2
        Rp_sup=Rt+Rt/2
        b=np.zeros((20,2))
        j=0

        for i in range(20):

            b[i][1]=l[i]
            b[i][0]=a[i]
            if Rp_inf<b[i][1]<Rp_sup: 
                    j=j+1
                    print('ligne=', str(b[i][0]), 'Rp=',str(b[i][1]),'K ohm' )
                    f.write(str(b[i][0]))
                    f.write('\t')
                    f.write('\t')
                    f.write('\t')
                    f.write('\t')
                    f.write(str(b[i][1]))
                    f.write('\n')
        print(j, "lignes sont exploitables sur la puce",name)
    with open("Résistance pristine/"+name+".txt", "a") as f: 
            f.write(text2)
            f.write(str(k))
            f.write('\n')
            f.write(text4)
            f.write(str(q))
            f.write('\n')
            f.write(text3)
            f.write(str(sigma))
            f.write('\n')
            f.write(text5)
            f.write(str(sigma_prime))
            f.write('\n')
    """

    plt.figure()
    for t in range(20):
        plt.plot(X[t], Rp[t][1], 'r.', markersize=dot_plot)
        plt.plot(X[t], Rp[t][2], 'b.', markersize=dot_plot)
        plt.plot(X[t], r_th_C, 'c_', markersize=dot_plot)
        plt.title(name, fontsize="18")
        plt.xlabel("N° ligne", fontsize="16")
        plt.ylabel("Résistance (K ohm)", fontsize="16")
    plt.xlim([0, 21])
    plt.ylim([0, 150])
    plt.grid()
    plt.xticks([0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
                10, 11, 12, 13,
                14, 15, 16, 17, 18, 19, 20], fontsize="11")
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                110, 120, 130, 140, 150], fontsize="11")
    plt.savefig("20240930/" + name + ".svg")
    plt.show()
# Taille D
if via == 4:
    val_inf = r_th_D - (r_th_D / 2)
    val_sup = r_th_D + (r_th_D / 2)
    g = list();
    s = list()
    print("La moyenne des résistance pristine est : ", k)
    for h in range(20):
        if val_inf < b[h] < val_sup:
            g.append(b[h])
    q = mean(g)
    print("La moyenne (sans valeurs abérrantes) des résistances pristines est : ", str(q))
    # Calcul de l'écart type
    with open("Résistance pristine/" + name + ".txt", "a") as f:
        f.write(text1)
        f.write('\n')
        f.write('\n')
        f.write(text)
        f.write('\n')
        sigma = stdev(b)
        print("L", "'", "écart-type sigma", "est égale à : ", str(sigma))
        for d in range(20):
            if val_inf < b[d] < val_sup:
                s.append(b[d])
        sigma_prime = stdev(s)
        print("Ecart-type (sans valeurs abérrantes) des résistance pristine est : ", str(sigma_prime))
    """
    #Selection des lignes
        Rt=80
        Rp_inf=Rt-Rt/2
        Rp_sup=Rt+Rt/2
        b=np.zeros((20,2))
        j=0

        for i in range(20):

            b[i][1]=l[i]
            b[i][0]=a[i]
            if Rp_inf<b[i][1]<Rp_sup: 
                    j=j+1
                    print('ligne=', str(b[i][0]), 'Rp=',str(b[i][1]),'K ohm' )
                    f.write(str(b[i][0]))
                    f.write('\t')
                    f.write('\t')
                    f.write('\t')
                    f.write('\t')
                    f.write(str(b[i][1]))
                    f.write('\n')
        print(j, "lignes sont exploitables sur la puce",name)
    with open("Résistance pristine/"+name+".txt", "a") as f: 
            f.write(text2)
            f.write(str(k))
            f.write('\n')
            f.write(text4)
            f.write(str(q))
            f.write('\n')
            f.write(text3)
            f.write(str(sigma))
            f.write('\n')
            f.write(text5)
            f.write(str(sigma_prime))
            f.write('\n')
"""

    plt.figure()
    for t in range(20):
        plt.plot(X[t], Rp[t][1], 'r.', markersize=dot_plot)
        plt.plot(X[t], r_th_D, 'c_', markersize=dot_plot)
        plt.title(name)
        plt.xlabel("N° ligne")
        plt.ylabel("Résistance (K ohm)")
    plt.xlim([0, 21])
    plt.ylim([0, 500])
    plt.grid()
    plt.xticks([0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
                10, 11, 12, 13,
                14, 15, 16, 17, 19, 20])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
                310, 320, 330, 340, 350, 360, 370, 380, 390, 400])