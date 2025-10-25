import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

from tools import bruit_gauss, calc_erreur, peano_transform_img, transform_peano_in_img
from gaussian_mixture import estim_param_SEM_gm, mpm_gm
from markov_chain import estim_param_SEM_mc, mpm_mc

imgs = [
    "images/alfa2.bmp",
    "images/veau2.bmp",
    "images/zebre2.bmp"
]

gaussian_noises = [
    {"m1": 0, "m2": 3, "sig1": 1, "sig2": 2},
    {"m1": 1, "m2": 1, "sig1": 1, "sig2": 5},
    {"m1": 0, "m2": 1, "sig1": 1, "sig2": 1}
]

cl = [0, 1]
iterations = 10
results = []

# 3 figures pour les 3 bruits 
for j, noise in enumerate(gaussian_noises):
    m1, m2, sig1, sig2 = noise["m1"], noise["m2"], noise["sig1"], noise["sig2"]

    n_rows, n_cols = len(imgs), 4  # Subplot : original, noisy, markov, indep
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    axes = np.atleast_2d(axes)

    for i, img_path in enumerate(imgs):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        min_dim = min(img.shape)
        img = img[:min_dim, :min_dim]
        signal = peano_transform_img(img)

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Original\n{img_path.split('/')[-1]}", fontsize=10)
        axes[i, 0].axis('off')

        signal_noisy = bruit_gauss(signal, cl, m1, sig1, m2, sig2)
        img_noisy = transform_peano_in_img(signal_noisy, img.shape[0])

        Y = signal_noisy.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(Y)
        labels = kmeans.labels_

        p = np.array([np.mean(labels == 0), np.mean(labels == 1)])
        A = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                transitions = np.sum((labels[:-1] == a) & (labels[1:] == b))
                total = np.sum(labels[:-1] == a)
                A[a, b] = transitions / total if total > 0 else 0

        m1_init = np.mean(signal_noisy[labels == 0])
        m2_init = np.mean(signal_noisy[labels == 1])
        sig1_init = np.sqrt(np.var(signal_noisy[labels == 0]))
        sig2_init = np.sqrt(np.var(signal_noisy[labels == 1]))

        # markov
        p_mc, A_mc, m1_mc, sig1_mc, m2_mc, sig2_mc = estim_param_SEM_mc(
            iterations, signal_noisy, p, A, m1_init, sig1_init, m2_init, sig2_init
        )
        signal_segmented_mc = mpm_mc(signal_noisy, cl, p_mc, A_mc, m1_mc, sig1_mc, m2_mc, sig2_mc)

        # indep
        p_gm, m1_gm, sig1_gm, m2_gm, sig2_gm = estim_param_SEM_gm(
            iterations, signal_noisy, p, m1_init, sig1_init, m2_init, sig2_init
        )
        signal_segmented_gm = mpm_gm(signal_noisy, cl, p_gm, m1_gm, sig1_gm, m2_gm, sig2_gm)

        erreur_mc = calc_erreur(labels, signal_segmented_mc)
        erreur_gm = calc_erreur(labels, signal_segmented_gm)

        img_mc = transform_peano_in_img(signal_segmented_mc, img.shape[0])
        img_gm = transform_peano_in_img(signal_segmented_gm, img.shape[0])

        imgs_to_show = [
            (img_noisy, f"Bruit\n ({m1}, {m2}, {sig1}, {sig2}"),
            (img_mc, f"Markov\nErr={erreur_mc:.2f}%"),
            (img_gm, f"Indép.\nErr={erreur_gm:.2f}%")
        ]
        for k, (im, title) in enumerate(imgs_to_show):
            ax = axes[i, k + 1]
            ax.imshow(im, cmap='gray')
            ax.set_title(title, fontsize=9)
            ax.axis('off')

        results.append({
            "Image": img_path.split('/')[-1],
            "m1": m1, "sig1": sig1,
            "m2": m2, "sig2": sig2,
            "Erreur Markov": erreur_mc,
            "Erreur Indépendant": erreur_gm,
            "m1_MC": m1_mc, "m2_MC": m2_mc,
            "m1_GM": m1_gm, "m2_GM": m2_gm
        })

    plt.suptitle(f"Bruit {j+1}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"figures/resultats_bruit_{j+1}.png")
    plt.show()

df = pd.DataFrame(results)
print(df)

df.to_csv("figures/resultats_segmentation.csv", index=False)
