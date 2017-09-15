#!/usr/bin/env python3.6
import numpy as np
import p1 as gs
import matplotlib.pyplot as plt
from PIL import Image
from sys import argv
import util

N = 2
ALPHA = 0.5
BETA = 0.5
GAMMA = 0.0

def piecewise_histogram_transform(im, n, alpha, beta, gamma):
    def dividir_histogramas(histograma, n):
        for desde, hasta in rangos:
            zeros = np.zeros(util.L)
            zeros[desde:hasta] += histograma[desde:hasta]
            yield zeros

    def Wk(H_k, k):
        desde, hasta = rangos[k]
        ha, hb = desde, hasta - 1
        vk = abs(ha - hb) / 2
        I_m = np.average(H_k[desde:hasta], weights=list(range(desde,hasta)))
        sigmath = abs(128 - I_m)
        sigmak = max(sigmath, vk)
        uk = (ha + hb) / 2
        W_k = np.empty(util.L)
        for i in range(util.L):
            W_k[i] = np.exp(- ((i-uk) ** 2) / (2 * (sigmak ** 2)))
        return W_k


    def Hu_k(H_k, k):
        desde, hasta = rangos[k]
        W_k = Wk(H_k, k)
        hu_k = np.empty(util.L)
        for i in range(len(hu_k)):
            hu_k[i] = W_k[i] if i in range(desde,hasta) else 1
        return hu_k

    im2 = util.to_hsi(im)

    histogram_i = gs.histograma(np.vectorize(lambda x: x * 255,
                                             otypes = [np.uint8])(im2[:,:,2]))
    r = np.linspace(0, util.L, dtype = int)
    rangos = np.column_stack((r[:-1], r[1:]))

    histogramas_divididos = dividir_histogramas(histogram_i, n)
    Ht_ks = []
    W_ks = []

    for k, H_k in enumerate(histogramas_divididos):
        l = len(H_k)
        Hu = Hu_k(H_k, k)
        D = (-1) * np.eye(l-1, l) + np.eye(l-1, l, k=1)
        Ht_k = np.dot(np.linalg.inv((alpha + beta) * np.eye(l) + gamma * np.dot(np.transpose(D), D)),
                      alpha * H_k + beta * Hu)
        Ht_ks.append(Ht_k)
        W_ks.append(Wk(H_k, k))

    sumW_k = sum(W_ks)
    w_k = np.empty((util.L, n))
    for i, k in np.ndindex(np.shape(w_k)):
        w_k[(i, k)] = W_ks[k][i] / sumW_k[i]
    Hs = np.empty(util.L)
    for i in range(len(Hs)):
        Hs[i] = sum(w_k[i][j] * Ht_ks[j][i] for j in range(n))
    normHs = [x / sum(Hs) for x in Hs]
    eq = gs.transformada(im2[:,:,2], normHs)
    im2[:,:,2] = np.vectorize(lambda x: x / 255, otypes = [float])(eq)
    imRes = util.to_rgb(im2)

    #print(eq)
    plt.figure(figsize=(16, 10))
    plt.subplot(2,2,1).imshow(im)
    plt.subplot(2,2,2).imshow(imRes)
    plt.show()
    # print(Hs)

im = np.asarray(Image.open(argv[1]).convert('RGB'))
piecewise_histogram_transform(im, N, ALPHA, BETA, GAMMA)

# im4 = util.to_rgb(im2)
#side_by_side.sbys_histogram([im1, im2, im3, im4], ['rgb', 'hsi', 'hsi', 'rgb'],
#                                argv=argv[2] if len(argv)>2 else None)
