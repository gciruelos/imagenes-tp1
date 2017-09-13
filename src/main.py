import numpy as np
from PIL import Image
from sys import argv
import util

N = 3
ALPHA = 0.5
BETA = 0.5
GAMMA = 0.0

def piecewise_histogram_transform(im, n, alpha, beta, gamma):
    def dividir_histogramas(histograma, n):
        for i in range(n):
            desde = i*largo_histogramas
            hasta = (i+1)*largo_histogramas

            zeros = np.zeros(len(histograma))
            zeros[desde:hasta] += histograma[desde:hasta]
            yield (i, list(zeros))

    def Wk(H_k, k):
        desde = k*largo_histogramas
        hasta = (k+1)*largo_histogramas
        ha = desde
        hb = (hasta - 1)
        vk = abs(ha - hb) / 2.0
        I_m = np.average(H_k[desde:hasta], weights=list(range(desde,hasta)))
        sigmath = abs(128 - I_m) 
        sigmak = max(sigmath, vk)
        uk = (ha + hb) / 2.0
        Wk = lambda i: np.exp(- np.power(i-uk, 2.0) / (2.0 * np.power(sigmak, 2.0)))
        return Wk
        
    
    def Hu_k(H_k, k):
        desde = k*largo_histogramas
        hasta = (k+1)*largo_histogramas
        W_k = Wk(H_k, k)
        return [W_k(i) if i in range(desde,hasta) else 1 for i in range(util.L)]


    im2 = util.to_hsi(im)

    histogram_h, accum_h, histogram_s, accum_s, histogram_i, accum_i = util.histogram_hsi(im2)
    largo_histogramas = len(histogram_i) // n
    
    histogramas_divididos = list(dividir_histogramas(histogram_i, n))
    Ht_ks = []
    W_ks = []

    for k, H_k in histogramas_divididos:
        l = len(H_k)
        H_k = np.array(H_k)
        Hu = np.array(Hu_k(H_k, k))
        D = (-1) * np.eye(l-1, l) + np.eye(l-1, l, k=1)
        Ht_k = np.dot(np.linalg.inv((alpha + beta) * np.eye(l) + gamma * np.dot(np.transpose(D), D)),
                      alpha * H_k + beta * Hu)
        Ht_ks.append(Ht_k)
        W_ks.append(Wk(H_k, k))

    w_k = [lambda i: W_ks[k](i) / sum(W_ks[j](i) for j in range(n)) for k in range(n)]
    Hs = [sum(w_k[j](i) * Ht_ks[j][i] for j in range(n)) for i in range(l)]
    print(Hs)

im = np.asarray(Image.open(argv[1]).convert('RGB'))
piecewise_histogram_transform(im, N, ALPHA, BETA, GAMMA)

# im4 = util.to_rgb(im2)
#side_by_side.sbys_histogram([im1, im2, im3, im4], ['rgb', 'hsi', 'hsi', 'rgb'],
#                                argv=argv[2] if len(argv)>2 else None)
