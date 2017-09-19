#!/usr/bin/env python
import numpy as np
import utils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image
from sys import argv

L = 256

LARGOS = [4,5,3,2,1]
ALPHA = 0.0000001
BETA = 1 - ALPHA
GAMMA = 0.

title = "alpha = "+str(ALPHA)+"; beta = "+str(BETA)+"; gamma = "+str(GAMMA)+"; n = "+str(len(LARGOS))

plt.figure(figsize=(10,10))
gridsp = gridspec.GridSpec(12, 3)
# Histogramas
ax1 = plt.subplot(gridsp[0:4, 0:2])
ax1.set_title("Separación de histogramas: H⁰_k")
ax1.get_xaxis().set_visible(False)
ax2 = plt.subplot(gridsp[4:8, 0:2])
ax2.set_title("Histogramas ecualizados: Hᵘ_k")
ax2.get_xaxis().set_visible(False)
ax3 = plt.subplot(gridsp[8:12, 0:2])
ax3.set_title("Integración de los histogramas: Hˢ")
# Imagenes
ax4 = plt.subplot(gridsp[0:3, 2])
ax4.set_title("Imagen original")
ax4.get_xaxis().set_visible(False)
ax4.get_yaxis().set_visible(False)
ax5 = plt.subplot(gridsp[3:6, 2])
ax5.set_title("Canal I original")
ax5.get_xaxis().set_visible(False)
ax5.get_yaxis().set_visible(False)
ax6 = plt.subplot(gridsp[6:9, 2])
ax6.set_title("Canal I post-algoritmo")
ax6.get_xaxis().set_visible(False)
ax6.get_yaxis().set_visible(False)
ax7 = plt.subplot(gridsp[9:12, 2])
ax7.set_title("Imagen post-algoritmo")
ax7.get_xaxis().set_visible(False)
ax7.get_yaxis().set_visible(False)

def piecewise_histogram_transform(I, particiones, alpha, beta, gamma):
    largo_particiones = list(map(lambda x: 256.0 * x / sum(particiones), particiones))
    # Hacemos dithering para distribuir los largos equitativamente
    deuda = 0.0
    for i in range(len(largo_particiones)):
        nuevo_valor = round(largo_particiones[i] + deuda)
        deuda += largo_particiones[i] - float(nuevo_valor)
        largo_particiones[i] = nuevo_valor
    # Arreglo el ultimo para que sumen 256
    largo_particiones[-1] += 256 - sum(largo_particiones)
    rangos = []
    for i in range(len(particiones)):
        desde = sum(largo_particiones[:i])
        rangos.append((desde, desde+largo_particiones[i]))

    n = len(particiones)
    histograma = utils.histograma(I)

    H0 = np.zeros((n, L), dtype = int)
    for k, (desde, hasta) in enumerate(rangos):
        H0[k][desde:hasta] += histograma[desde:hasta]

    W = np.empty((n, L), dtype = float)
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

    Hu = np.empty((n, L), dtype = float)
    for k in range(n):
        desde, hasta = rangos[k]
        for i in range(len(Hu[k])):
            Hu[k][i] = 1 if i in range(desde,hasta) else W[k][i]

    Ht = np.empty((n, L), dtype = float)
    for k in range(n):
        D = (-1) * np.eye(L-1, L) + np.eye(L-1, L, k=1)
        Ht[k] = np.dot(np.linalg.inv((alpha + beta) * np.eye(L) + gamma *
                                    np.dot(np.transpose(D), D)),
                      alpha * H0[k] + beta * Hu[k])

    sumW = sum(W)
    w = np.empty((n, L))
    for k, i in np.ndindex(np.shape(w)):
        w[(k, i)] = W[(k, i)] / sumW[i]
    Hs = np.empty(L)
    for i in range(len(Hs)):
        Hs[i] = sum(w[(j, i)] * Ht[(j, i)] for j in range(n))
    normHs = np.vectorize(lambda x: x / np.sum(Hs))(Hs)
    eq = utils.transformada(I, normHs)

    for k in range(n):
        ax1.plot(range(L), H0[k])
    for k in range(n):
        ax2.plot(range(L), Ht[k])
    ax3.plot(range(L), normHs)

    return np.vectorize(lambda x: x / 255, otypes = [float])(eq)

imrgb = np.asarray(Image.open(argv[1]).convert('RGB'))
im = utils.toHSI(imrgb)
eqI = piecewise_histogram_transform(np.vectorize(lambda x: x * 255,
                                                 otypes = [np.uint8])(im[2]),
                                    LARGOS, ALPHA, BETA, GAMMA)
im2 = utils.toRGB(im[0], im[1], eqI)

ax4.imshow(imrgb)
ax5.imshow(im[2], cmap='gray', vmin=0.0, vmax=1.0)
ax6.imshow(eqI, cmap='gray', vmin=0.0, vmax=1.0)
ax7.imshow(im2)
plt.tight_layout()

if len(argv) > 2:
    plt.savefig(argv[2])
else:
    plt.show()

