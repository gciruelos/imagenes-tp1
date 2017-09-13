import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
from PIL import Image

L = 256
EPS = 0.000001


'''
CALCULO DE HISTORGRAMAS RGB
'''
def histogram_rgb(img):
    histogram_r = [0 for i in range(L)]
    histogram_g = [0 for i in range(L)]
    histogram_b = [0 for i in range(L)]
    nm = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p = img[i, j]
            histogram_r[p[0]] += 1
            histogram_g[p[1]] += 1
            histogram_b[p[2]] += 1
            nm += 1

    for i in range(L):
        histogram_r[i] = float(histogram_r[i]) / nm
        histogram_g[i] = float(histogram_g[i]) / nm
        histogram_b[i] = float(histogram_b[i]) / nm

    accum_r = [0 for i in range(L)]
    accum_g = [0 for i in range(L)]
    accum_b = [0 for i in range(L)]
    for i in range(256):
        for j in range(i):
            accum_r[i] += histogram_r[j]
            accum_g[i] += histogram_g[j]
            accum_b[i] += histogram_b[j]
            if accum_r[i] > 1.0: accum_r[i] = 1.0
            if accum_g[i] > 1.0: accum_g[i] = 1.0
            if accum_b[i] > 1.0: accum_b[i] = 1.0
    return histogram_r, accum_r, histogram_g, accum_g, histogram_b, accum_b

'''
CALCULO DE HISTORGRAMAS HSI
'''
SAMPLES = 256
dom_h = [2*np.pi * float(i) / float(SAMPLES) for i in range(SAMPLES)]
dom_s = [float(i) / float(SAMPLES) for i in  range(SAMPLES)]
dom_i = [float(i) / float(SAMPLES) for i in  range(SAMPLES)]

def search_not_exact(needle, haystack):
    lo = 0
    hi = len(haystack)
    while lo + 1 < hi and lo + (hi - lo) // 2 < len(haystack) - 1:
        mid = lo + (hi - lo) // 2 
        if haystack[mid] <= needle < haystack[mid+1]:
            return mid
        elif haystack[mid] > needle:
            hi = mid
        else:
            lo = mid
    return lo


def histogram_hsi(img):
    histogram_h = [0 for i in range(SAMPLES)]
    histogram_s = [0 for i in range(SAMPLES)]
    histogram_i = [0 for i in range(SAMPLES)]
    nm = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p = img[i, j]
            histogram_h[search_not_exact(p[0], dom_h)] += 1
            histogram_s[search_not_exact(p[1], dom_s)] += 1
            histogram_i[search_not_exact(p[2], dom_i)] += 1
            nm += 1

    for i in range(SAMPLES):
        histogram_h[i] = float(histogram_h[i]) / nm
        histogram_s[i] = float(histogram_s[i]) / nm
        histogram_i[i] = float(histogram_i[i]) / nm

    accum_h = [0 for i in range(SAMPLES)]
    accum_s = [0 for i in range(SAMPLES)]
    accum_i = [0 for i in range(SAMPLES)]
    for i in range(SAMPLES):
        for j in range(i):
            accum_h[i] += histogram_h[j]
            accum_s[i] += histogram_s[j]
            accum_i[i] += histogram_i[j]
            if accum_h[i] > 1.0: accum_h[i] = 1.0
            if accum_s[i] > 1.0: accum_s[i] = 1.0
            if accum_i[i] > 1.0: accum_i[i] = 1.0
    return histogram_h, accum_h, histogram_s, accum_s, histogram_i, accum_i

'''
CONVERSION HSI <-> RGB
'''
def to_hsi(img):
    imgr = img.copy().astype(np.float32)
    for i_ in range(imgr.shape[0]):
        for j_ in range(imgr.shape[1]):
            p = imgr[i_, j_]
            r = float(p[0]) / L
            g = float(p[1]) / L
            b = float(p[2]) / L
            cuenta_h = np.arccos(
                (0.5 * ((r - g) + (r - b))) / (EPS + np.power(np.power(r-g, 2.0) + (r-b)*(g-b), 0.5)))
            if b <= g:
                h = cuenta_h
            else:
                h = 2*np.pi - cuenta_h
            s = 1.0 - 3.0 * float(min(r,g,b)) / (EPS + float(r + g + b))
            i = float(r + g + b) / 3.0
            imgr[i_, j_] = [h, s, i]
    return imgr


def hsi_to_rgb(hsi):
    h = hsi[0]
    s = hsi[1]
    i = hsi[2]
    # arreglo de potenciales problemas (coordenadas fuera del cono).
    if i > 0.5 and s > 2 - 0.5 * i:
        s = 2 - 0.5 * i
    if i < 0.5 and s > 2 * i:
        s = 2 * i

    if h < 2.0 * np.pi / 3.0:
        r = i * (1 + s * np.cos(h)                 / np.cos(np.pi/3.0 - h))
        b = i * (1 - s)
        g = 3 * i - (r + b)
    elif h < 4.0 * np.pi / 3.0:
        g = i * (1 + s * np.cos(h - 2.0*np.pi/3.0) / np.cos(np.pi/3.0 - (h - 2.0*np.pi/3.0)))
        r = i * (1 - s)
        b = 3 * i - (r + g)
    else:
        b = i * (1 + s * np.cos(h - 4.0*np.pi/3.0) / np.cos(np.pi/3.0 - (h - 4.0*np.pi/3.0)))
        g = i * (1 - s)
        r = 3 * i - (g + b)
    if r > 1.0: print(hsi, 'r', r)
    if g > 1.0: print(hsi, 'g', g)
    if b > 1.0: print(hsi, 'b', b)
    r = min(int(256 * r - 0.5), 255)
    g = min(int(256 * g - 0.5), 255)
    b = min(int(256 * b - 0.5), 255)
    return [r, g, b]
    

def to_rgb(img):
    imgr = img.copy().astype(np.uint8)
    for i_ in range(imgr.shape[0]):
        for j_ in range(imgr.shape[1]):
            imgr[i_, j_] = hsi_to_rgb(img[i_, j_])
    return imgr

'''
GRAFICAR
'''

def sbys_histogram(im1, ims_hsi, hists_i, accum_i, hists_i_nuevos, accum_i_nuevos, im2):
    plt.figure(figsize=(16, 10))
    if argv == '--end':
        gs = gridspec.GridSpec(1, 2)
        plt.subplot(gs[0]).imshow(im1)
        plt.subplot(gs[1]).imshow(im2)
        if len(sys.argv) > sys.argv.index('--end') + 1:
            plt.savefig(sys.argv[sys.argv.index('--end') + 1], bbox_inches='tight')
        else:
            plt.show()
        return
    n = len(hists_i) + 1
    gs = gridspec.GridSpec(
        8, n, width_ratios=[1 for _ in range(n)], height_ratios=[1, 1, 1, 1])

    plt.subplot(gs[0:2, 0:(n+1)]).imshow(im1)
    plt.subplot(gs[2:4, 0]).imshow(ims_hsi[0][:,:,2], cmap='gray', vmin=0.0, vmax=2*np.pi)
    for i in range(n):
        plt.subplot(gs[2, i+1]).plot(range(L), hists_i, 'k-')
        plt.subplot(gs[3, i+1]).plot(range(L), accum_i, 'k-')
    plt.subplot(gs[4:6, 0]).imshow(ims_hsi[1][:,:,2], cmap='gray', vmin=0.0, vmax=2*np.pi)
    for i in range(n):
        plt.subplot(gs[4, i+1]).plot(range(L), hists_i_nuevo, 'k-')
        plt.subplot(gs[5, i+1]).plot(range(L), accum_i_nuevo, 'k-')
    plt.subplot(gs[6:8, 0:(n+1)]).imshow(im2)
    if argv is not None:
        plt.savefig(argv, bbox_inches='tight')
    else:
        plt.show()
