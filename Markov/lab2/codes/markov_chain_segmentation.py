import numpy as np
from tools import bruit_gauss, calc_erreur
from markov_chain import *
import matplotlib.pyplot as plt

# parameters
m1, m2, sig1, sig2 = 0, 3, 1, 2
n = 200
A = np.array([[0.8, 0.2], [0.3, 0.7]])
cl = [0, 1]  # Valeurs des classes
p = [0.3, 0.7]

signal = simu_mc(n, cl, p, A)
signal_noisy = bruit_gauss(signal, cl, m1, sig1, m2, sig2)
signal_segmented = mpm_mc(signal_noisy, cl, p, A, m1, sig1, m2, sig2)

# erreur
erreur = calc_erreur(signal, signal_segmented)

# signal discret
plt.figure(figsize=(12, 6))
plt.plot(signal, 'b-', label='Signal discret')
plt.plot(signal_noisy, 'r-', label='Signal bruité')
plt.plot(signal_segmented, 'g-', label='Signal segmenté')
plt.title('Signal discret généré')
plt.legend()

plt.savefig("figures/original_bruit_segment.png")


# taux d'erreur
print(f"Taux d'erreur entre le signal original et le signal segmenté : {erreur:.2f}%")

##########################
# 3 bruits
##########################
gaussian_noises = [
    {"m1": 0, "m2": 3, "sig1": 1, "sig2": 2},
    {"m1": 1, "m2": 1, "sig1": 1, "sig2": 5},
    {"m1": 0, "m2": 1, "sig1": 1, "sig2": 1}
]

transition_matrices = [
    np.array([[0.9, 0.1], [0.4, 0.6]]),
    np.array([[0.6, 0.4], [0.2, 0.8]]),
    np.array([[0.1, 0.9], [0.5, 0.5]]) 
]

n = 1000
cl=[0,1]
p = [0.3, 0.7]

results = []

for idx_A, A in enumerate(transition_matrices):
    signal = simu_mc(n, cl, p, A)

    for noise in gaussian_noises:
        m1, m2, sig1, sig2 = noise["m1"], noise["m2"], noise["sig1"], noise["sig2"]

        signal_noisy = bruit_gauss(signal, cl, m1, sig1, m2, sig2)
        signal_segmented = mpm_mc(signal_noisy, cl, p, A, m1, sig1, m2, sig2)
        erreur = calc_erreur(signal, signal_segmented)

        results.append({
            "Matrice": f"A{idx_A + 1}",
            "Bruit": f"({m1}, {m2}, {sig1}, {sig2})",
            "Taux erreur (%)": f"{erreur:.2f}"
        })

print(results)
