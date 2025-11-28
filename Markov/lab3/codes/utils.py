import numpy as np
from scipy.stats import multivariate_normal


def creer_trajectoire(F, Q, N, x_init):
   """
   Cette fonction permet de générer une trajectoire a peu près rectiligne uniforme
   (à un bruit près)
   :param F: la matrice de passage de X_n à X_(n+1)
   :param Q: la matrice de covariance du bruit gaussien centré U_(n+1)
   :param N: la taille de la trajectoire à générer
   :param x_init: Le premier point de la trajectoire
   :return: la trajectoire générée (numpy array N*4  de float)
   """
   # 𝑋𝑛+1 = 𝐹𝑛 * 𝑋𝑛 + 𝑈𝑛+1
   x = np.zeros((N, len(x_init)))
   x[0] = x_init

   for n in range(1, N):
      bruit = np.random.multivariate_normal(np.zeros(len(x_init)), Q)
      x[n] = F @ x[n-1] + bruit
   return x


def creer_observations(H, R, x):
   """
   Cette fonction permet de bruiter une trajectoire x, pour
   :param H: la matrice de passage de X_(n+1) à Y_(n+1)
   :param R: la matrice de covariance du bruit gaussien centré V_(n+1)
   :param x: la trajectoire à bruiter
   :return: la trajectoire bruitée (numpy array N*2  de float)
   """
   # 𝑌𝑛+1 = 𝐻𝑛+1 * 𝑋𝑛+1 + 𝑉𝑛+1
   N = x.shape[0]
   y = np.zeros((N, H.shape[0]))

   for n in range(N):
      bruit = np.random.multivariate_normal(np.zeros(H.shape[0]), R)
      y[n] = H @ x[n] + bruit
   return y


def filtre_de_kalman_iter(F, Q, H, R, y_n, x_kalm_prec, P_kalm_prec):
   """
   Cette fonction permet de calculer l'état estimé et sa variance à l'instant n
   à partir de l'état estimé et sa variance à l'instant n-1
   :param F: la matrice de passage de X_n à X_(n+1)
   :param Q: la matrice de covariance du bruit gaussien centré U_(n+1)
   :param H: la matrice de passage de X_(n+1) à Y_(n+1)
   :param R: la matrice de covariance du bruit gaussien centré V_(n+1)
   :param y_n: la n-ième observation associée à la trajectoire
   :param x_kalm_prec: l'état estimé à l'instant n-1
   :param P_kalm_prec: la variance de l'état estimé à l'instant n-1
   :return: l'état estimé à l'instant n (numpy array de taille 4  de float)
   et la variance de l'état estimé à l'instant n (numpy array de taille 4*4  de float)
   """
   x_pred = F @ x_kalm_prec
   P_pred = F @ P_kalm_prec @ F.T + Q

   innovation = y_n - H @ x_pred

   # Gain de Kalman
   S = H @ P_pred @ H.T + R
   K = P_pred @ H.T @ np.linalg.inv(S)

   # Mise à jour de l’état et de la covariance
   x_kalm_n = x_pred + K @ innovation
   P_kalm_n = (np.eye(len(x_kalm_prec)) - K @ H) @ P_pred
   return x_kalm_n, P_kalm_n


def filtre_de_kalman(F, Q, H, R, y, x_init, P_init):
   """
   Cette fonction permet de mettre dans une boucle for la fonction
   filtre_de_kalman_iter pour calculer successivement tous les états et les variances
   estimés par le filtre de kalman
   :param F: la matrice de passage de X_n à X_(n+1)
   :param Q: la matrice de covariance du bruit gaussien centré U_(n+1)
   :param H: la matrice de passage de X_(n+1) à Y_(n+1)
   :param R: la matrice de covariance du bruit gaussien centré V_(n+1)
   :param y: l'ensemble des observations associées à la trajectoire
   :param x_init: le premier point de la trajectoire
   :return: un tableau contenant tous les états estimés (numpy array
   de taille N*4  de float) et un tableau contenant les variances (numpy array
   de taille N*4*4  de float)
   """
   N = y.shape[0]
   dim_x = len(x_init)
   x_est = np.zeros((N,dim_x))
   P_est = np.zeros((N,dim_x,dim_x))
   x_kalm_prec = x_init
   P_kalm_prec = P_init

   for n in range(N):
      y_n = y[n]
      if np.isnan(y_n).any():
            # Étape de prédiction seulement
            x_pred = F @ x_kalm_prec
            P_pred = F @ P_kalm_prec @ F.T + Q

            x_kalm_n = x_pred
            P_kalm_n = P_pred
      else:
         # Étape complète du filtre de Kalman
         x_kalm_n, P_kalm_n = filtre_de_kalman_iter(F, Q, H, R, y_n, x_kalm_prec, P_kalm_prec)


      x_est[n] = x_kalm_n
      P_est[n] = P_kalm_n

      x_kalm_prec = x_kalm_n
      P_kalm_prec = P_kalm_n

   return x_est, P_est


def calc_MSE(x, x_est):
   """
   Cette fonction permet de calculer l'erreur quadratique moyenne entre
   les trajectoires x et y
   :param x: la vraie trajectoire
   :param x_est: la trajectoire estimée
   :return: l'erreur quadratique moyenne entre
   la vraie trajectoire et la trajectoire estimée (float)
   """
   erreur_pos = x[:, [0, 2]] - x_est[:, [0, 2]]
   MSE = np.mean(np.sum(erreur_pos**2, axis=1))

   return MSE



