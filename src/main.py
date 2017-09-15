#!/usr/bin/env python3.6
import numpy as np
import p1 as gs
import p2
import matplotlib.pyplot as plt
from PIL import Image
from sys import argv
import util

N = 2
ALPHA = 0.5
BETA = 0.5
GAMMA = 0.0

def piecewise_histogram_transform(I, n, alpha, beta, gamma):

    histograma = gs.histograma(I)
    r = np.linspace(0, util.L, n + 1, dtype = int)
    rangos = np.column_stack((r[:-1], r[1:]))

    H0 = np.zeros((n, util.L), dtype = int)
    for k, (desde, hasta) in enumerate(rangos):
        H0[k][desde:hasta] += histograma[desde:hasta]

    W = np.empty((n, util.L), dtype = float)
    for k in range(len(W)):
        desde, hasta = rangos[k]
        ha, hb = desde, hasta - 1
        vk = abs(ha - hb) / 2
        I_m = np.average(H0[k][desde:hasta], weights=list(range(desde,hasta)))
        sigmath = abs(128 - I_m)
        sigmak = max(sigmath, vk)
        uk = (ha + hb) / 2
        for i in range(len(W[k])):
            W[k][i] = np.exp(- ((i-uk) ** 2) / (2 * (sigmak ** 2)))

    Hu = np.empty((n, util.L), dtype = float)
    for k in range(len(Hu)):
        desde, hasta = rangos[k]
        for i in range(len(Hu[k])):
            Hu[k][i] = W[k][i] if i in range(desde,hasta) else 1

    Ht = np.empty((n, util.L), dtype = float)
    for k, H in enumerate(H0):
        D = (-1) * np.eye(util.L-1, util.L) + np.eye(util.L-1, util.L, k=1)
        Ht[k] = np.dot(np.linalg.inv((alpha + beta) * np.eye(util.L) + gamma *
                                    np.dot(np.transpose(D), D)),
                      alpha * H + beta * Hu[k])

    sumW = sum(W)
    w = np.empty((n, util.L))
    for k, i in np.ndindex(np.shape(w)):
        w[(k, i)] = W[k][i] / sumW[i]
    Hs = np.empty(util.L)
    for i in range(len(Hs)):
        Hs[i] = sum(w[j][i] * Ht[j][i] for j in range(n))
    normHs = np.vectorize(lambda x: x / np.sum(Hs))(Hs)
    eq = gs.transformada(I, normHs)
    return np.vectorize(lambda x: x / 255, otypes = [float])(eq)

im = p2.toHSI(np.asarray(Image.open(argv[1]).convert('RGB')))
eqI = piecewise_histogram_transform(np.vectorize(lambda x: x * 255,
                                                 otypes = [np.uint8])(im[2]),
                                    N, ALPHA, BETA, GAMMA)

Image.fromarray(p2.toRGB(im[0], im[1], eqI), mode = 'RGB').show()
