import numpy as np
import matplotlib.pyplot as plt
from utils import calc_MSE, filtre_de_kalman

x_ligne = np.load("lab3/data/vecteur_x_avion_ligne.npy")
y_ligne = np.load("lab3/data/vecteur_y_avion_ligne.npy")

x_voltige = np.load("lab3/data/vecteur_x_avion_voltige.npy")
y_voltige = np.load("lab3/data/vecteur_y_avion_voltige.npy")

Te = 1
N = 100
sigma_Q = 1
sigma_px = 30
sigma_py = 30

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

P_init = np.identity(4)

x_est_ligne, P_est_ligne = filtre_de_kalman(F, Q, H, R, y_ligne, x_ligne[0, :], P_init)
x_est_voltige, P_est_voltige = filtre_de_kalman(F, Q, H, R, y_voltige, x_voltige[0, :], P_init)

mse_ligne = calc_MSE(x_ligne, x_est_ligne)
mse_voltige = calc_MSE(x_voltige, x_est_voltige)
print(f"Erreur quadratique moyenne (avion de ligne) : {mse_ligne:.2f}")
print(f"Erreur quadratique moyenne (avion de voltige) : {mse_voltige:.2f}")

t = np.arange(len(x_ligne))


plt.figure(figsize=(7, 6))
plt.plot(x_ligne[:, 0], x_ligne[:, 2], 'b-', label="Trajectoire vraie")
plt.plot(x_est_ligne[:, 0], x_est_ligne[:, 2], 'g--', label="Trajectoire estimée (Kalman)")
plt.scatter(y_ligne[:, 0], y_ligne[:, 1], c='r', s=10, alpha=0.5, label="Observations")
plt.xlabel("Position x")
plt.ylabel("Position y")
plt.title("Avion de ligne")
plt.legend()
plt.savefig("lab3/figures/traj_ligne.png")


plt.figure(figsize=(7, 6))
plt.plot(x_voltige[:, 0], x_voltige[:, 2], 'b-', label="Trajectoire vraie")
plt.plot(x_est_voltige[:, 0], x_est_voltige[:, 2], 'g--', label="Trajectoire estimée (Kalman)")
plt.scatter(y_voltige[:, 0], y_voltige[:, 1], c='r', s=10, alpha=0.5, label="Observations")
plt.xlabel("Position x")
plt.ylabel("Position y")
plt.title("Avion de voltige")
plt.legend()
plt.savefig("lab3/figures/traj_voltige.png")
