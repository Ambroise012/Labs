import numpy as np
from tools import bruit_gauss, taux_erreur, calc_probaprio2, MAP_MPM2, classif_gauss2, simul2
import matplotlib.pyplot as plt
from config import cfg

param_bruit = cfg["param_bruit"]
signaux = [np.load(path) for path in cfg["signaux"]]

p1_values = [0.1, 0.3, 0.5, 0.7, 0.9]
p2_values = [0.9, 0.7, 0.5, 0.3, 0.1]

n = 1000

erreurs_moyennes_MV = np.zeros((5, 5, 5))
erreurs_moyennes_MAP = np.zeros((5, 5, 5))

for k, params in enumerate(param_bruit):
    for i, p1 in enumerate(p1_values):
        p2 = p2_values[i]
        for j in range(5):
            X = simul2(n, 0, 1, p1, p2)
            cl1, cl2 = np.unique(X)

            Y = bruit_gauss(X, cl1, cl2, params["m1"], params["sig1"], params["m2"], params["sig2"])

            # MV
            X_est_MV = classif_gauss2(Y, cl1, cl2, params["m1"], params["sig1"], params["m2"], params["sig2"])
            erreurs_moyennes_MV[i, k, j] = taux_erreur(X, X_est_MV)

            # MAP
            X_est_MAP = MAP_MPM2(Y, cl1, cl2, p1, p2, params["m1"], params["sig1"], params["m2"], params["sig2"])
            erreurs_moyennes_MAP[i, k, j] = taux_erreur(X, X_est_MAP)

erreurs_moyennes_MV_moy = np.mean(erreurs_moyennes_MV, axis=2)
erreurs_moyennes_MAP_moy = np.mean(erreurs_moyennes_MAP, axis=2)

print("\nMV :")
for k in range(5):
    print(f"\nParamètres de bruit {k+1} : {param_bruit[k]}")
    for i in range(5):
        print(f"p1/p2 = {p1_values[i]}/{p2_values[i]}: {erreurs_moyennes_MV_moy[i, k]:.2f}")

print("\nMAP :")
for k in range(5):
    print(f"\nParamètres de bruit {k+1} : {param_bruit[k]}")
    for i in range(5):
        print(f"p1/p2 = {p1_values[i]}/{p2_values[i]}: {erreurs_moyennes_MAP_moy[i, k]:.2f}")