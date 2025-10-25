import numpy as np
from tools import gauss


def forward(A, p, gauss):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs forward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param gauss: numpy array 2D (longeur de signal_noisy,2) qui correspond aux valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: numpy array 2D (longeur de signal_noisy,2) contenant tous les forward (de 1 à longeur de signal_noisy) pour chaque classe
    """
    n = gauss.shape[0]
    alpha = np.zeros((n, 2))
    alpha[0, :] = p * gauss[0, :]
    alpha[0, :] /= np.sum(alpha[0, :])  # rescaling

    for t in range(1, n):
        alpha[t, :] = np.dot(alpha[t - 1, :], A) * gauss[t, :]
        s = np.sum(alpha[t, :])
        if s == 0:
            s = 1e-12
        alpha[t, :] /= s 

    return alpha


def backward(A, gauss):
    """
    Cette fonction calcule récursivement (mais ce n'est pas une fonction récursive!) les valeurs backward de la chaîne
    :param A: Matrice (2*2) de transition de la chaîne
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param gauss: numpy array (longeur de signal_noisy)*2 qui correspond aux valeurs des densité gaussiennes pour chaque élément du signal bruité
    :return: numpy array 2D (longeur de signal_noisy,2) contenant tous les backward (de 1 à longeur de signal_noisy) pour chaque classe.
    Attention, si on calcule les backward en partant de la fin de la chaine, je conseille quand même d'ordonner le vecteur backward du début à la fin
    """
    n = gauss.shape[0]
    beta = np.ones((n,2))
    beta[-1,:] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t, :] = np.dot(A, (gauss[t + 1, :] * beta[t + 1, :]))
        s = np.sum(beta[t, :])
        if s == 0:
            s = 1e-12
        beta[t, :] /= s

    return beta


def mpm_mc(signal_noisy, cl, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction permet d'appliquer la méthode mpm pour retrouver notre signal d'origine à partir de sa version bruité et des paramètres du model.
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param cl: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    alpha = forward(A,p,gausses)
    beta = backward(A,gausses)

    # proba a posteriori
    gamma = alpha * beta
    sum_gamma = gamma.sum(axis=1, keepdims=True)
    sum_gamma[sum_gamma == 0] = 1.0  # Évite la division par zéro
    gamma = gamma / sum_gamma

    # choix classe qui maximise
    signal_discret = np.array([cl[i] for i in np.argmax(gamma, axis=1)])

    return signal_discret.astype(int)


def calc_probaprio_mc(signal, cl):
    """
    Cete fonction permet de calculer les probabilité a priori des classes w1 et w2 et les transitions a priori d'une classe à l'autre,
    en observant notre signal non bruité
    :param signal: Signal discret non bruité à deux classes (numpy array 1D d'int)
    :param cl: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :return: une liste contenant: un vecteur p de taille 2 avec la probailité d'apparition a priori pour chaque classe et une matrice A de taille (2,2) correspondant aux probas de transitions a priori
    """
    n = len(signal)
    n_w1 = np.sum(signal == cl[0])
    n_w2 = np.sum(signal == cl[1])
    p = np.array([n_w1/n, n_w2/n])
    A = np.zeros((2,2))

    for i in range(n-1):
        current_cl = signal[i]
        next_cl = signal[i+1]
        if current_cl == cl[0]:
            if next_cl == cl[0]:
                A[0,0]+=1
            else:
                A[0,1]+=1
        else:
            if next_cl == cl[0]:
                A[1,0]+=1
            else:
                A[1,1]+=1
    A[0,:] = A[0,:]/np.sum(A[0,:]) 
    A[1,:] = A[1,:]/np.sum(A[1,:]) 

    return [p,A]


def simu_mc(n, cl, p, A):
    """
    Cette fonction permet de simuler un signal discret à 2 classe de taille n à partir des probabilités d'apparition des deux classes et de la Matrice de transition
    :param n: taille du signal
    :param cl: vecteur dont la première composante est la valeur de la classe w1 et la deuxième est la valeur de la classe w2
    :param p: vecteur de taille 2 avec la probailité d'apparition pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :return: Un signal discret à 2 classe (numpy array 1D d'int)
    """
    x = np.zeros(n,dtype=int)
    x[0] = np.random.choice(cl, p=p)

    for i in range(1,n):
        # indice de la classe précédente
        x_index = np.where(cl == x[i-1])[0][0]
        x[i] = np.random.choice(cl, p=A[x_index,:])
    return x


def calc_param_SEM_mc(signal_noisy, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction permet de calculer les nouveaux paramètres estimé pour une itération de SEM
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: vecteur de taille 2 avec la probailité d'apparition a priori pour chaque classe
    :param A: Matrice (2*2) de transition de la chaîne
    :param m1: La moyenne de la première gaussienne
    :param sig1: La variance de la première gaussienne
    :param m2: La moyenne de la deuxième gaussienne
    :param sig2: La variance de la deuxième gaussienne
    :return: tous les paramètres réestimés donc p, A, m1, sig1, m2, sig2
    """
    gausses = gauss(signal_noisy, m1, sig1, m2, sig2)
    alpha = forward(A, p, gausses)
    beta = backward(A, gausses)

    # proba a posteriori
    gamma = alpha * beta
    sum_gamma = gamma.sum(axis=1, keepdims=True)
    sum_gamma[sum_gamma == 0] = 1.0  # Évite la division par zéro
    gamma = gamma / sum_gamma

    n = len(signal_noisy)

    # proba conjointe
    xi = np.zeros((n-1,2,2))
    for t in range(n - 1):
        for i in range(2):
            for j in range(2):
                xi[t, i, j] = alpha[t, i] * A[i, j] * gausses[t + 1, j] * beta[t + 1, j]
        s = np.sum(xi[t])
        if s > 0:
            xi[t] /= s
        else:
            xi[t] = np.full((2, 2), 0.25)


    p_new = gamma[0,:]

    A_new = np.zeros((2,2))
    for i in range(2):
        denom = np.sum(gamma[:-1, i])
        if denom > 0:
            A_new[i, :] = np.sum(xi[:, i, :], axis=0) / denom
        else:
            A_new[i, :] = 0.5  # ligne uniforme

    # Normalisation
    A_new = np.nan_to_num(A_new)
    A_new /= A_new.sum(axis=1, keepdims=True) + 1e-12

    # moyennes et écarts types
    sum_gamma1 = np.sum(gamma[:, 0])
    sum_gamma2 = np.sum(gamma[:, 1])

    if sum_gamma1 > 0:
        m1_new = np.sum(gamma[:, 0] * signal_noisy) / sum_gamma1
        sig1_new = np.sqrt(np.sum(gamma[:, 0] * (signal_noisy - m1_new) ** 2) / sum_gamma1)
    else:
        m1_new, sig1_new = m1, sig1

    if sum_gamma2 > 0:
        m2_new = np.sum(gamma[:, 1] * signal_noisy) / sum_gamma2
        sig2_new = np.sqrt(np.sum(gamma[:, 1] * (signal_noisy - m2_new) ** 2) / sum_gamma2)
    else:
        m2_new, sig2_new = m2, sig2

    # pourla stabilité numérique 
    sig1_new = max(sig1_new, 1e-3)
    sig2_new = max(sig2_new, 1e-3)
    p_new = np.clip(p_new, 1e-3, 1 - 1e-3)
    p_new /= np.sum(p_new)
    
    return p_new, A_new, m1_new, sig1_new, m2_new, sig2_new


def estim_param_SEM_mc(iter, signal_noisy, p, A, m1, sig1, m2, sig2):
    """
    Cette fonction est l'implémentation de l'algorithme SEM pour le modèle en question
    :param iter: Nombre d'itération choisie
    :param signal_noisy: Signal bruité (numpy array 1D de float)
    :param p: la valeur d'initialisation du vecteur de proba
    :param A: la valeur d'initialisation de la matrice de transition de la chaîne
    :param m1: la valeur d'initialisation de la moyenne de la première gaussienne
    :param sig1: la valeur d'initialisation de la variance de la première gaussienne
    :param m2: la valeur d'initialisation de la moyenne de la deuxième gaussienne
    :param sig2: la valeur d'initialisation de la variance de la deuxième gaussienne
    :return: Tous les paramètres réestimés à la fin de l'algorithme SEM donc p, A, m1, sig1, m2, sig2
    """
    p_est = p
    A_est = A
    m1_est = m1
    sig1_est = sig1
    m2_est = m2
    sig2_est = sig2
    for i in range(iter):
        p_est, A_est, m1_est, sig1_est, m2_est, sig2_est = calc_param_SEM_mc(signal_noisy, p_est, A_est, m1_est, sig1_est, m2_est, sig2_est)
        print({'p':p_est, 'A':A_est, 'm1':m1_est, 'sig1':sig1_est, 'm2':m2_est, 'sig2':sig2_est})
    return p_est, A_est, m1_est, sig1_est, m2_est, sig2_est

