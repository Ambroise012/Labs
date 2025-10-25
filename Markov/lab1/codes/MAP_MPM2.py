import numpy as np
from tools import bruit_gauss, taux_erreur, calc_probaprio2, MAP_MPM2, classif_gauss2
import matplotlib.pyplot as plt
from config import cfg

param_bruit = cfg["param_bruit"]
signaux = [np.load(path) for path in cfg["signaux"]]

erreurs_moyennes_MV = np.zeros((6, 5))
erreurs_moyennes_MAP = np.zeros((6, 5))

for i, X in enumerate(signaux):
    cl1, cl2 = np.unique(X)
    p1, p2 = calc_probaprio2(X, cl1, cl2)
    for j, params in enumerate(param_bruit):
        # Maximum de Vraisemblance
        erreurs_MV = []
        for _ in range(100):
            Y = bruit_gauss(X, cl1, cl2, params["m1"], params["sig1"], params["m2"], params["sig2"])
            X_est_MV = classif_gauss2(Y, cl1, cl2, params["m1"], params["sig1"], params["m2"], params["sig2"])
            erreurs_MV.append(taux_erreur(X, X_est_MV))
        erreurs_moyennes_MV[i, j] = np.mean(erreurs_MV)

        # Maximum A Posteriori
        erreurs_MAP = []
        for _ in range(100):
            Y = bruit_gauss(X, cl1, cl2, params["m1"], params["sig1"], params["m2"], params["sig2"])
            X_est_MAP = MAP_MPM2(Y, cl1, cl2, p1, p2, params["m1"], params["sig1"], params["m2"], params["sig2"])
            erreurs_MAP.append(taux_erreur(X, X_est_MAP))
        erreurs_moyennes_MAP[i, j] = np.mean(erreurs_MAP)

print("\nMaximum de Vraisemblance :")
for i in range(6):
    print(f"Signal {i+1}: {erreurs_moyennes_MV[i, :].round(2)}")

print("\nMaximum A Posteriori :")
for i in range(6):
    print(f"Signal {i+1}: {erreurs_moyennes_MAP[i, :].round(2)}")
