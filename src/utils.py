#!/usr/bin/env python3.6
from PIL import Image
import numpy as np
from sys import argv

def histograma(im):
    h = np.zeros(256, dtype = float)
    s = 0
    for v in np.nditer(im):
        s += 1
    for v in np.nditer(im):
        h[v] += 1./s
    return h

#Función general para la transformada
#Target es la p.m.f. deseada
#h es el histograma que se manipulará
def transformada(im, target, h = None, L1 = 256, tipo = np.uint8):
    L2 = 256
    #Si no doy un histograma al llamar a la función, calculo el de la imagen
    if h is None:
        h = histograma(im)
    #Normalizo para tener una p.m.f
    pR = h / np.sum(h)
    #Calculo la c.d.f. de la variable de entrada
    W = [pR[0]] * L1
    for k, v in enumerate(pR[1:]):
        W[k + 1] = W[k] + v
    #Calculo la c.d.f. de la variable de salida
    Wm = [target[0]] * L2
    for k, v in enumerate(target[1:]):
        Wm[k + 1] = Wm[k] + v
    #Calculo el valor de salida para cada valor de entrada
    #--
    #El valor inicial debería sobreescribirse,
    #ya que el último valor de ambas aculumaladas debería ser 1,
    #(entonces siempre debería ∃w ∈ Wm / w - W[i] ≽ 0, ∀i)
    #Sin embargo, al trabajar con valores de punto flotante pueden introducirse pequeños errores
    #Por ende, el último valor de la acumulada de salida podría ser ligeramente menor a 1,
    #por lo que el valor inicial se mantendrá; como quiero que sea el último valor posible del rango,
    #inicializo con L2 - 1
    Y = np.ones(L1, dtype = int) * (L2 - 1)
    #Al ser c.d.f.s, están ordenados, por lo que puedo hacer una sola pasada de cada array
    k = 0
    for i in range(len(Y)):
        for j, w in enumerate(Wm[k:]):
            if w - W[i] >= 0:
                Y[i], k = j + k, j + k
                break
    #Busco en Y el valor de salida para cada celda de la imagen original
    return np.vectorize(lambda x: Y[x], otypes = [tipo])(im)

def RGBtoHSI(R, G, B):
    R /= 255
    G /= 255
    B /= 255
    n = (R - G) + (R - B)
    n /= 2
    d = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    θ = np.arccos(n/d)
    if B <= G:
        H = θ
    else:
        H = 2 * np.pi - θ
    if R + G + B != 0:
        S = 1 - 3 * min(R, G, B) / (R + G + B)
    else:
        S = 0
    I = (R + G + B) / 3
    return H, S, I

def HSItoRGB(h, s, i):
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
    r = min(int(256 * r - 0.5), 255)
    g = min(int(256 * g - 0.5), 255)
    b = min(int(256 * b - 0.5), 255)
    return r, g, b

def toHSI(im):
    Hs = np.empty(np.shape(im)[:2], dtype = float)
    Ss = np.empty(np.shape(im)[:2], dtype = float)
    Is = np.empty(np.shape(im)[:2], dtype = float)
    for i in range(len(im)):
        if not i % 25:
            print(i)
        for j in range(len(im[0])):
            H, S, I = RGBtoHSI(*im[(i, j)])
            Hs[(i, j)] = H
            Ss[(i, j)] = S
            Is[(i, j)] = I
    return Hs, Ss, Is

def toRGB(H, S, I):
    res = np.empty((*np.shape(H), 3), dtype = np.uint8)
    for i, j in np.ndindex(np.shape(H)):
        R, G, B = HSItoRGB(H[(i, j)], S[(i, j)], I[(i, j)])
        res[(i, j, 0)] = R
        res[(i, j, 1)] = G
        res[(i, j, 2)] = B
    return res
