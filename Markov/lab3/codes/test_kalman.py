import numpy as np
import matplotlib.pyplot as plt
from utils import calc_MSE, creer_trajectoire, creer_observations, filtre_de_kalman

Te = 1
N = 100
sigma_Q = 1
sigma_px = 30
sigma_py = 30

x_init = np.array([3, 40, -4, 20])

F = np.array([
    [1, Te, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, Te],
    [0, 0, 0, 1]
])

H = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

R = np.array([
    [sigma_px**2, 0],
    [0, sigma_py**2]
])

Q = sigma_Q * np.array([
    [Te**3/3, Te**2/2, 0, 0],
    [Te**2/2, Te, 0, 0],
    [0, 0, Te**3/3, Te**2/2],
    [0, 0, Te**2/2, Te]
])


x = creer_trajectoire(F, Q, N, x_init)
y = creer_observations(H, R, x)


plt.figure(figsize=(8, 6))
plt.plot(x[:, 0], x[:, 2], 'b-', label="Trajectoire vraie")
plt.scatter(y[:, 0], y[:, 1], color='r', alpha=0.6, label="Observations bruitées")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis("equal")
plt.savefig("lab3/figures/trajectoire_bruit.png")
