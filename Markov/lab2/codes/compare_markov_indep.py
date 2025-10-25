import numpy as np
from gaussian_mixture import *
from markov_chain import *
from tools import bruit_gauss, calc_erreur
import matplotlib.pyplot as plt

# param
n = 2000
cl = [0,1]
m1, m2, sig1, sig2 = 0,3,1,2
p_ind = (0.25, 0.75)
p_markov = (0.25, 0.75)
A = np.array([[0.8, 0.2], [0.07, 0.93]])


# Générer signal 
signal_indep = simu_gm(n, cl, p_ind)

# Bruiter le signal
signal_indep_noisy = bruit_gauss(signal_indep, cl, m1, sig1, m2, sig2)

# Estimer les paramètres
p_mc_est, A_mc_est = calc_probaprio_mc(signal_indep, cl)

# Restaurer le signal par MPM
signal_indep_segmented_gm = mpm_gm(signal_indep_noisy, cl, p_ind, m1, sig1, m2, sig2)
signal_indep_segmented_mc = mpm_mc(signal_indep_noisy, cl, p_mc_est, A_mc_est, m1, sig1, m2, sig2)

# Taux d'erreur
erreur_indep_gm = calc_erreur(signal_indep, signal_indep_segmented_gm)
erreur_indep_mc = calc_erreur(signal_indep, signal_indep_segmented_mc)

# Générer un signal par une chaîne de Markov
signal_mc = simu_mc(n, cl, p_markov, A)

# Bruiter le signal
signal_mc_noisy = bruit_gauss(signal_mc, cl, m1, sig1, m2, sig2)

# Estimer les param
p_indep_est = calc_probaprio_gm(signal_mc, cl)

# Restaurer le signal par MPM
signal_mc_segmented_gm = mpm_gm(signal_mc_noisy, cl, p_indep_est, m1, sig1, m2, sig2)
signal_mc_segmented_mc = mpm_mc(signal_mc_noisy, cl, p_markov, A, m1, sig1, m2, sig2)

# taux d'erreur
erreur_mc_gm = calc_erreur(signal_mc, signal_mc_segmented_gm)
erreur_mc_mc = calc_erreur(signal_mc, signal_mc_segmented_mc)

print("Signal généré par le modèle indépendant")
print(f"Taux d'erreur avec MPM (modèle indépendant) : {erreur_indep_gm:.2f}%")
print(f"Taux d'erreur avec MPM (chaîne de Markov) : {erreur_indep_mc:.2f}%")

print("Signal généré par la chaîne de Markov")
print(f"Taux d'erreur avec MPM (modèle indépendant) : {erreur_mc_gm:.2f}%")
print(f"Taux d'erreur avec MPM (chaîne de Markov) : {erreur_mc_mc:.2f}%")
