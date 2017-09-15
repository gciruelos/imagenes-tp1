#!/usr/bin/env python
import numpy as np
import p1 as gs
import p2
import matplotlib.pyplot as plt
from PIL import Image
from sys import argv
import util

LARGOS = [1,5,1]
ALPHA = 0.0000001
BETA = 1 - ALPHA
GAMMA = 0.

def piecewise_histogram_transform(I, particiones, alpha, beta, gamma):
    largo_particiones = list(map(lambda x: round(256.0 * x / sum(particiones)), particiones))
    # Arreglo el ultimo para que sumen 256
    largo_particiones[-1] += 256 - sum(largo_particiones)
    rangos = []
    for i in range(len(particiones)):
        desde = sum(largo_particiones[:i])
        rangos.append((desde, desde+largo_particiones[i]))
    
    n = len(particiones)
    histograma = gs.histograma(I)

    H0 = np.zeros((n, util.L), dtype = int)
    for k, (desde, hasta) in enumerate(rangos):
        H0[k][desde:hasta] += histograma[desde:hasta]

    W = np.empty((n, util.L), dtype = float)
    for k in range(n):
        desde, hasta = rangos[k]
        ha, hb = desde, hasta - 1
        vk = abs(ha - hb) / 2
        pesos = H0[k][desde:hasta]
        if sum(pesos) != 0:
            pesos = np.vectorize(lambda x: x / sum(pesos))(pesos)
        I_m = np.average(np.dot(list(range(desde,hasta)), pesos))
        sigmath = abs(128 - I_m)
        sigmak = max(sigmath, vk)
        uk = (ha + hb) / 2
        for i in range(len(W[k])):
            W[k][i] = np.exp(-(np.power(i-uk, 2) / (2 * np.power(sigmak, 2))))

    Hu = np.empty((n, util.L), dtype = float)
    for k in range(n):
        desde, hasta = rangos[k]
        for i in range(len(Hu[k])):
            Hu[k][i] = 1 if i in range(desde,hasta) else W[k][i]

    Ht = np.empty((n, util.L), dtype = float)
    for k in range(n):
        D = (-1) * np.eye(util.L-1, util.L) + np.eye(util.L-1, util.L, k=1)
        Ht[k] = np.dot(np.linalg.inv((alpha + beta) * np.eye(util.L) + gamma *
                                    np.dot(np.transpose(D), D)),
                      alpha * H0[k] + beta * Hu[k])

    sumW = sum(W)
    w = np.empty((n, util.L))
    for k, i in np.ndindex(np.shape(w)):
        w[(k, i)] = W[(k, i)] / sumW[i]
    Hs = np.empty(util.L)
    for i in range(len(Hs)):
        Hs[i] = sum(w[(j, i)] * Ht[(j, i)] for j in range(n))
    normHs = np.vectorize(lambda x: x / np.sum(Hs))(Hs)
    eq = gs.transformada(I, normHs)

    plt.subplot(3, 1, 1)
    for k in range(n):
        plt.plot(range(util.L), H0[k])
    plt.subplot(3, 1, 2)
    for k in range(n):
        plt.plot(range(util.L), Ht[k])
    plt.subplot(3, 1, 3)
    plt.plot(range(util.L), normHs)
    plt.show()

    return np.vectorize(lambda x: x / 255, otypes = [float])(eq)

im = p2.toHSI(np.asarray(Image.open(argv[1]).convert('RGB')))
eqI = piecewise_histogram_transform(np.vectorize(lambda x: x * 255,
                                                 otypes = [np.uint8])(im[2]),
                                    LARGOS, ALPHA, BETA, GAMMA)

Image.fromarray(p2.toRGB(im[0], im[1], eqI), mode = 'RGB').show()#.save("../Resultados/wom.png")
