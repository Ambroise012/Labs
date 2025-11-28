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

P_init = np.identity(4)

x_true = creer_trajectoire(F, Q, N, x_init)
y_obs = creer_observations(H, R, x_true)

x_est, P_est = filtre_de_kalman(F, Q, H, R, y_obs, x_init, P_init)

# MSE
MSE = calc_MSE(x_true, x_est)
print("MSE : ", MSE)

# mean mse : 
MSE_mean = []
for i in range(1000):
    x_true = creer_trajectoire(F, Q, N, x_init)
    y_obs = creer_observations(H, R, x_true)

    x_est, P_est = filtre_de_kalman(F, Q, H, R, y_obs, x_init, P_init)
    mse = calc_MSE(x_true, x_est)
    MSE_mean.append(mse)
print("Mean MSE: ", np.mean(MSE_mean))


t = np.arange(N)

plt.figure(figsize=(10, 5))
plt.plot(t, x_true[:, 0], 'b-', label='Position vraie (x)')
plt.plot(t, y_obs[:, 0], 'r.', label='Position observée (x)')
plt.plot(t, x_est[:, 0], 'g--', label='Position estimée (x)')
plt.xlabel("Temps (s)")
plt.ylabel("Position x")
plt.title("Position en abscisse (x)")
plt.legend()
plt.savefig("lab3/figures/pos_absci.png")

plt.figure(figsize=(10, 5))
plt.plot(t, x_true[:, 2], 'b-', label='Position vraie (y)')
plt.plot(t, y_obs[:, 1], 'r.', label='Position observée (y)')
plt.plot(t, x_est[:, 2], 'g--', label='Position estimée (y)')
plt.xlabel("Temps (s)")
plt.ylabel("Position y")
plt.title("Position en ordonnée (y)")
plt.legend()
plt.savefig("lab3/figures/pos_ordo.png")

plt.figure(figsize=(10, 6))
plt.plot(x_true[:, 0], x_true[:, 2], 'b-', label="Trajectoire vraie")
plt.scatter(y_obs[:, 0], y_obs[:, 1], color='r', alpha=0.6, label="Observations bruitées")
plt.plot(x_est[:, 0], x_est[:, 2], 'g--', label="Trajectoire estimée (Kalman)")
plt.xlabel("Position x")
plt.ylabel("Position y")
plt.legend()
plt.title("Trajectoire de la cible")
plt.savefig("lab3/figures/traj_cible.png")
