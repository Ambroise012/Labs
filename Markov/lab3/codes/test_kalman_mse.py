import numpy as np
import matplotlib.pyplot as plt
from utils import creer_trajectoire, creer_observations, filtre_de_kalman, calc_MSE

Te = 1.0
N = 100
x_init = np.array([3, 40, -4, 20])

# param à tester
sigma_Q_vals = np.array([0.0, 0.01, 0.1, 1.0, 10.0])

F = np.array([
    [1, Te, 0, 0],
    [0, 1,  0, 0],
    [0, 0,  1, Te],
    [0, 0,  0, 1]
])
H = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])
P_init = np.identity(4)

# -------------------------
# MSE en fonction de sigma_Q
# -------------------------
# sigma_p_x
plt.figure(figsize=(8,6))
for sigma_p_x in [1.0, 10.0, 30.0, 100.0]:
    mse_means = []
    for sigma_Q in sigma_Q_vals:
        mse_runs = []
        for run in range(100):
            Q = (sigma_Q**2) * np.array([
                [Te**3/3, Te**2/2, 0, 0],
                [Te**2/2, Te, 0, 0],
                [0, 0, Te**3/3, Te**2/2],
                [0, 0, Te**2/2, Te]
            ])
            sigma_p_y = 30
            R = np.diag([sigma_p_x**2, sigma_p_y**2])

            x_true = creer_trajectoire(F, Q, N, x_init)
            y_obs = creer_observations(H, R, x_true)

            x_est, P_est = filtre_de_kalman(F, Q, H, R, y_obs, x_init, P_init)

            # mse
            mse = calc_MSE(x_true, x_est)
            mse_runs.append(mse)
        mse_means.append(np.mean(mse_runs))
    plt.plot(sigma_Q_vals, mse_means, marker='o', label=f'sigma_p_x={sigma_p_x}')
plt.xscale('log') # log pour plus de visibilité
plt.yscale('log')
plt.xlabel('sigma_Q (écart-type bruit de processus)')
plt.ylabel('MSE (moyenne)')
plt.title('MSE vs sigma_Q pour différents sigma_p_x')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig("lab3/figures/MSE_vs_sigQ_sig_p_x.png")

# sigma_p_y
plt.figure(figsize=(8,6))
for sigma_p_y in [1.0, 10.0, 30.0, 100.0]:
    mse_means = []
    for sigma_Q in sigma_Q_vals:
        mse_runs = []
        for run in range(100):
            Q = (sigma_Q**2) * np.array([
                [Te**3/3, Te**2/2, 0, 0],
                [Te**2/2, Te, 0, 0],
                [0, 0, Te**3/3, Te**2/2],
                [0, 0, Te**2/2, Te]
            ])
            sigma_p_x = 30
            R = np.diag([sigma_p_x**2, sigma_p_y**2])

            x_true = creer_trajectoire(F, Q, N, x_init)
            y_obs = creer_observations(H, R, x_true)

            x_est, P_est = filtre_de_kalman(F, Q, H, R, y_obs, x_init, P_init)

            # mse
            mse = calc_MSE(x_true, x_est)
            mse_runs.append(mse)
        mse_means.append(np.mean(mse_runs))
    plt.plot(sigma_Q_vals, mse_means, marker='o', label=f'sigma_p_y={sigma_p_y}')
plt.xscale('log') # log pour plus de visibilité
plt.yscale('log')
plt.xlabel('sigma_Q (écart-type bruit de processus)')
plt.ylabel('MSE (moyenne)')
plt.title('MSE vs sigma_Q pour différents sigma_p_y')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.savefig("lab3/figures/MSE_vs_sigQ_sig_p_y.png")
