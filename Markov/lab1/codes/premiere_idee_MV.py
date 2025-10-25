import os

import numpy as np
import matplotlib.pyplot as plt
from tools import bruit_gauss, classif_gauss2, taux_erreur
from config import cfg

X = np.load("signaux/signal.npy")

cl1, cl2 = np.unique(X)
m1, m2, sig1, sig2 = 120, 130, 1, 2

Y = bruit_gauss(X, cl1, cl2, m1, sig1, m2, sig2)

X_est = classif_gauss2(Y, cl1, cl2, m1, sig1, m2, sig2)

tau = taux_erreur(X,X_est)
    
print(f"Taux d'erreur de segmentation {tau}%")

plt.figure(figsize=(10, 6))
plt.plot(X, 'b--', label='Signal original')
plt.plot(Y, 'r-', label='Signal bruité', linewidth=2)
plt.plot(X_est, 'g:', label='Signal segmenté', linewidth=2)
plt.title("Signal original, bruité et segmenté")
plt.xlabel("Échantillons")
plt.ylabel("Labels")
plt.legend()
os.makedirs("figures", exist_ok=True)

plt.savefig("figures/original_bruit_segment.png")
plt.show()



# ####################
# Erreur pour T variant de 1 à 100
# ####################
T_max = 100
erreurs_moyennes = np.zeros(T_max)

for T in range(1, T_max + 1):
    erreurs = []
    for _ in range(T):
        Y = bruit_gauss(X, cl1, cl2, m1, sig1, m2, sig2)
        X_est = classif_gauss2(Y, cl1, cl2, m1, sig1, m2, sig2)
        erreurs.append(taux_erreur(X, X_est))
    erreurs_moyennes[T-1] = np.mean(erreurs)

plt.figure(figsize=(10, 6))
plt.plot(range(1, T_max + 1), erreurs_moyennes, 'b-', label='Erreur moyenne')
plt.title("Évolution de l'erreur moyenne en fonction de T")
plt.xlabel("Nombre de simulations T")
plt.ylabel("Erreur moyenne (%)")
plt.legend()
plt.savefig("figures/erreur_moyenne.png")
plt.show()


# ####################
# Pour les 6 signaux
# ####################

param_bruit = cfg["param_bruit"]
signaux = [np.load(path) for path in cfg["signaux"]]

erreurs_moyennes = np.zeros((6, 5))

for i, X in enumerate(signaux):
    cl1, cl2 = np.unique(X)
    for j, params in enumerate(param_bruit):
        erreurs = []
        for _ in range(T_max):  # T = 100
            Y = bruit_gauss(X, cl1, cl2, params["m1"], params["sig1"], params["m2"], params["sig2"])
            X_est = classif_gauss2(Y, cl1, cl2, params["m1"], params["sig1"], params["m2"], params["sig2"])
            erreurs.append(taux_erreur(X, X_est))
        erreurs_moyennes[i, j] = np.mean(erreurs)

print("--------- Erreurs moyennes ---------")
for i in range(6):
    print(f"Signal {i+1}: {erreurs_moyennes[i, :].round(2)}")


# visualisation de qlq cas interessants

cas_interessants = [1, 3, 5]
n_echan = 300

X = np.load("signaux/signal.npy")
cl1, cl2 = np.unique(X)

os.makedirs("figures", exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

for idx, k in enumerate(cas_interessants):
    params = cfg["param_bruit"][k - 1]
    
    # Génération du signal bruité et segmenté
    Y = bruit_gauss(X, cl1, cl2, params["m1"], params["sig1"], params["m2"], params["sig2"])
    X_est = classif_gauss2(Y, cl1, cl2, params["m1"], params["sig1"], params["m2"], params["sig2"])
    
    # Calcul du taux d’erreur
    tau = taux_erreur(X, X_est)
    
    # Tracé sur le subplot correspondant
    ax = axes[idx]
    ax.plot(X[:n_echan], 'b--', label='Original')
    ax.plot(Y[:n_echan], 'r-', alpha=0.7, label='Bruit')
    ax.plot(X_est[:n_echan], 'g:', linewidth=2, label='Segmenté')
    
    ax.set_title(f"Cas {k} — m₁={params['m1']}, m₂={params['m2']}, σ₁={params['sig1']}, σ₂={params['sig2']}  |  Erreur = {tau:.2f}%", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    if idx == 0:
        ax.legend(loc='upper right')

axes[-1].set_xlabel("Échantillons")
fig.suptitle("Comparaison des signaux : original, bruité et segmenté", fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("figures/comparaison_signaux.png", dpi=300)
plt.show()