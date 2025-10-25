
import numpy as np
from scipy.stats import norm

def bruit_gauss(X, cl1, cl2, m1, sig1, m2, sig2):
    """
    Cette fonction permet de bruiter un signal discret à deux classes avec deux gaussiennes
    :param X: le signal a bruiter (un numpy array d'int)
    :param cl1: la valeur de la classe 1 du signal X
    :param cl2: la valeur de la classe 2 du signal X
    :param m1: la moyenne de la première gaussienne
    :param sig1: l'écart type de la première gaussienne
    :param m2: la moyenne de la deuxième gaussienne
    :param sig2: l'écart type de la deuxième gaussienne
    :return: le signal bruité (numpy array de float)
    """
    # init signal bruité Y
    Y = X.astype(float)
    bruit1 =np.random.normal(m1, sig1, size = X.shape)
    bruit2 = np.random.normal(m2, sig2, size = X.shape)

    Y[X==cl1] = bruit1[X==cl1]
    Y[X==cl2] = bruit2[X==cl2]
    return Y


def classif_gauss2(Y, cl1, cl2, m1, sig1, m2, sig2):
    """
    Cette fonction permet de construire le signal segmenté X_est en classant
    les données du signal bruité Y dans les classes cl1 et cl2 suivant le
    critère du maximum de vraisemblance
    :param Y: le signal bruité (un numpy array de float)
    :param cl1: la valeur de la classe 1 du signal X
    :param cl2: la valeur de la classe 2 du signal X
    :param m1: la moyenne de la première gaussienne
    :param sig1: l'écart type de la première gaussienne
    :param m2: la moyenne de la deuxième gaussienne
    :param sig2: l'écart type de la deuxième gaussienne
    :return: une estimation du signal discret X (numpy array d'int)
    """
    # maximum de vraissemblance pour la classe1 et 2
    MV_1 = norm.pdf(Y, loc=m1, scale=sig1)
    MV_2 = norm.pdf(Y, loc=m2, scale=sig2)
    X_est = np.where(MV_1 > MV_2, cl1, cl2)

    return X_est


def taux_erreur(A, B):
    """
    Cette fonction permet de mesurer la difference entre deux signaux discret (de même taille) à deux classes
    :param A: le premier signal, un numpy array
    :param B: le deuxième signal, un numpy array
    :return: le pourcentage de différence entre les deux signaux (un float)
    """

    diff = np.sum(A != B)
    tau = diff / len(A)*100

    return tau


def calc_probaprio2(X, cl1, cl2):
    """
    Fonction qui calcule la loi du processus X a priori à partir du signal d'origine X
    :param X: le signal discret (un numpy array d'int)
    :param cl1: la valeur de la classe 1 du signal X
    :param cl1: la valeur de la classe 2 du signal X
    :return: Deux float correspondant aux probabilité à priori estimées de X
    """
    n_cl1 = np.sum(X==cl1)
    n_cl2 = np.sum(X==cl2)

    p1 = n_cl1 / len(X)
    p2 = n_cl2 / len(X)
    return p1, p2


def MAP_MPM2(Y, cl1, cl2, p1, p2, m1, sig1, m2, sig2):
    """
    Cette fonction permet de construire le signal segmenté X_est en classant
    les données du signal bruité Y dans les classes cl1 et cl2 suivant le
    critère du maximum a posteriori
    :param Y: Le signal bruité (un numpy array de float)
    :param cl1: la valeur de la classe 1 du signal X
    :param cl2: la valeur de la classe 2 du signal X
    :param p1: probabilité a priori de la classe 1
    :param p2: probabilité a priori de la classe 2
    :param m1: la moyenne de la première gaussienne
    :param sig1: l'écart type de la première gaussienne
    :param m2: la moyenne de la deuxième gaussienne
    :param sig2: l'écart type de la deuxième gaussienne
    :return: une estimation du signal discret X (numpy array d'int)
    """
    MV_1 = norm.pdf(Y, loc=m1, scale=sig1)
    MV_2 = norm.pdf(Y, loc=m2, scale=sig2)
    
    # à posteriori
    post_1 = MV_1 * p1
    post_2 = MV_2 * p2

    X_est = np.where(post_1 > post_2, cl1, cl2)

    return X_est


def simul2(n, cl1, cl2, p1, p2):
    """
    Cette fonction simule un signal de taille n dont les composantes
    sont indépendantes et prennent les valeurs
    cl1 et cl2 avec les probabilités respectives p1 et p2.
    :param n: taille du signal à simuler
    :param cl1: la valeur de la classe 1 du signal à simuler
    :param cl2: la valeur de la classe 2 du signal à simuler
    :param p1: probabilité a priori de la classe 1
    :param p2: probabilité a priori de la classe 2
    :return: Un signal discret X (numpy array d'int)
    """
    x = np.random.rand(n)
    X_simu = np.where(x <= p1, cl1, cl2)

    return X_simu


