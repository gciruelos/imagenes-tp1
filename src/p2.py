#!/usr/bin/env python3.6
from PIL import Image
import numpy as np
from scipy.stats import norm
import p1 as gs
from sys import argv
import argparse as ap

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

def histRGB(im):
    return (gs.histograma(C) for C in im)

def equalPorCanal(im):
    return (gs.equal(C) for C in im)

def despHSI(im):
    H, S, I = toHSI(im)
    Hp = np.empty_like(im)
    for k, h in np.ndenumerate(H):
        Hp[k] = HSItoRGB(h, 1, 0.5)
    S = np.vectorize(lambda s: np.floor(s * 255), otypes = [np.uint8])(S)
    I = np.vectorize(lambda i: np.floor(i * 255), otypes = [np.uint8])(I)
    return Hp, S, I

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

def transfHSI(im, f):
    return toRGB(*f(*toHSI(im)))

def equalI(im):
    def eqI(H, S, I):
        I = np.vectorize(lambda i: np.floor(i * 255), otypes = [np.uint8])(I)
        I = gs.equal(I)
        I = np.vectorize(lambda i: i / 255, otypes = [float])(I)
        return H, S, I
    return transfHSI(im, eqI)

def logI(im):
    def lI(H, S, I):
        I = np.vectorize(lambda i: np.log(i + 1))(I)
        return H, S, I
    return transfHSI(im, lI)

def rotH(im, c):
    def lH(H, S, I):
        H = np.vectorize(lambda h: (h + c) % (2 * np.pi))(H)
        return H, S, I
    return transfHSI(im, lH)

def reflH(im, c):
    def rH(H, S, I):
        H = np.vectorize(lambda h: (2 * c - h) % (2 * np.pi))(H)
        return H, S, I
    return transfHSI(im, rH)

def reflHS(im, cH, cS):
    def rHS(H, S, I):
        H = np.vectorize(lambda h: (2 * cH - h) % (2 * np.pi))(H)
        S = np.vectorize(lambda s: (2 * cS - s) % 1)(S)
        return H, S, I
    return transfHSI(im, rHS)

def reflSLim(im, c):
    def rSL(H, S, I):
        for k in np.ndindex(np.shape(S)):
            SLim = 1 - 2 * np.abs(I[k] - 0.5)
            S[k] = 2 * c * SLim - S[k]
            if S[k] > SLim:
                S[k] = SLim
            if S[k] < 0:
                S[k] = 0
        return H, S, I
    return transfHSI(im, rSL)

def partH(im, N):
    def partN(H, S, I):
        p = np.linspace(0, 2 * np.pi, N + 1)
        for k in np.ndindex(np.shape(H)):
            H[k] = p[np.argmin([np.abs(pP - H[k]) for pP in p]) % N]
        return H, S, I
    return transfHSI(im, partN)

def partS(im, N):
    def partSN(H, S, I):
        for k in np.ndindex(np.shape(I)):
            SLim = 1 - 2 * np.abs(I[k] - 0.5)
            p = np.linspace(0, SLim, N)
            S[k] = p[np.argmin([np.abs(pP - S[k]) for pP in p])]
        return H, S, I
    return transfHSI(im, partSN)

def grain(im, sel, mu, sigma):
    def gr(H, S, I):
        for b, c, n in zip(sel, (H, S, I), range(3)):
            if b:
                vals = norm.rvs(loc = mu, scale = sigma, size = np.shape(c))
                for k in np.ndindex(np.shape(c)):
                    c[k] += vals[k]
                    if c[k] < 0:
                        c[k] = 0
                    if n == 0:
                        if c[k] > 2 * np.pi:
                            c[k] %= 2 * np.pi
                    else:
                        if c[k] > 1:
                            c[k] = 1
        return H, S, I
    return transfHSI(im, gr)

def blur(im, sel, k):
    def bl(H, S, I):
        for b, c in zip(sel, (H, S, I)):
            if b:
                for i in range(k, len(c) - k):
                    for j in range(k, len(c[0]) - k):
                        c[(i, j)] = np.mean(c[i - k:i + k + 1, j - k:j + k + 1])
        return H, S, I
    return transfHSI(im, bl)

def multS(im, c):
    def mS(H, S, I):
        S = np.vectorize(lambda s: min(1, s * c))(S)
        return H, S, I
    return transfHSI(im, mS)

def chan(im):
    return (im[:,:,i].view() for i in range(3))

def unChan(chans):
    return np.dstack(chans)

if __name__ == "__main__":
    p = ap.ArgumentParser(prog = './p2.py')
    p.add_argument('entrada', help='Imagen de entrada.')
    p.add_argument('--show', dest = 'mostrar', action = 'store_true',
                   help='Mostrar la imagen de salida por pantalla en vez de guardarla.')
    p.add_argument('--showOG', dest = 'mostrarOG', action = 'store_true',
                   help='Mostrar la imagen de entrada por pantalla.')
    sp = p.add_subparsers(dest = 'ej', help = 'Ejercicio a realizar.')

    sp.add_parser('hist', help='Computar el histograma RGB de una imagen.')
    sp2 = sp.add_parser('eqRGB', help='Ecualizar los canales R, G, B.')
    sp2.add_argument('--unChan', dest = 'unChan', action = 'store_true',
                     help = 'Juntar los canales ecualizados en una imagen a color \
                     en vez de mostrarlos por separado como imágenes en escala de gris')

    sp.add_parser('despHSI', help='Desplegar los canales H, S, I.')
    sp.add_parser('eqI', help='Ecualizar el canal I.')

    sp1 = sp.add_parser('transfHSI',
                        help='Realizar transformaciones sobre los canales H, S, I.')
    spSub1 = sp1.add_subparsers(dest = 'transf',
                                help = 'Transformación a realizar.')

    spSub1.add_parser('logI', help='Aplicar T(r) = log(r + 1) sobre el canal I.')
    spSb1Sb1 = spSub1.add_parser('multS',
                                help='Multiplicar cada valor del canal S por una constante.')
    spSb1Sb1.add_argument('c', type = float, help = 'Constante por la que multiplicar.')
    spSb1Sb2 = spSub1.add_parser('addH',
                                help='Sumar una constante a cada valor del canal H.')
    spSb1Sb2.add_argument('c', type = float, help = 'Constante a sumar.')
    spSb1Sb3 = spSub1.add_parser('reflSLim',
                                help='Reflejar S.')
    spSb1Sb3.add_argument('c', type = float,
                          help = 'Valor respecto del cual reflejar.')
    spSb1Sb3 = spSub1.add_parser('partH',
                                help='Asignar cada H al más cercano de N valores equidistribuidos.')
    spSb1Sb3.add_argument('N', type = int,
                          help = 'Cantidad de valores.')
    spSb1Sb3 = spSub1.add_parser('partS',
                                help='Asignar cada S al más cercano de N valores equidistribuidos.')
    spSb1Sb3.add_argument('N', type = int,
                          help = 'Cantidad de valores.')
    spSb1Sb4 = spSub1.add_parser('blur',
                                help='Realizar blur sobre los canales H, S o I.')
    spSb1Sb4.add_argument('k', type = int,
                          help = 'Cantidad de pixeles a cada lado a promediar.')
    spSb1Sb4.add_argument('-H', dest = 'Hb', action = 'store_true',
                          help = 'Realizar blur sobre el canal H.')
    spSb1Sb4.add_argument('-S', dest = 'Sb', action = 'store_true',
                          help = 'Realizar blur sobre el canal S.')
    spSb1Sb4.add_argument('-I', dest = 'Ib', action = 'store_true',
                          help = 'Realizar blur sobre el canal I.')
    spSb1Sb5 = spSub1.add_parser('grain',
                                help='Aplicar un grano Gaussiano sobre los canales H, S o I.')
    spSb1Sb5.add_argument('mu', type = float,
                          help = 'Media de la Gaussiana.')
    spSb1Sb5.add_argument('sigma', type = float,
                          help = 'Desviación estándar de la Gaussiana.')
    spSb1Sb5.add_argument('-H', dest = 'Hb', action = 'store_true',
                          help = 'Aplicar grano sobre el canal H.')
    spSb1Sb5.add_argument('-S', dest = 'Sb', action = 'store_true',
                          help = 'Aplicar grano sobre el canal S.')
    spSb1Sb5.add_argument('-I', dest = 'Ib', action = 'store_true',
                          help = 'Aplicar grano sobre el canal I.')

    rgb = ['R', 'G', 'B']
    hsi = ['H', 'S', 'I']

    vargs = vars(p.parse_args())
    entrada = vargs['entrada']
    try:
        inp = Image.open("Imagenes/" + entrada + ".png").convert(mode = "RGB")
    except FileNotFoundError:
        inp = Image.open("Resultados/" + entrada + ".png").convert(mode = "RGB")
    im = np.array(inp)
    if vargs['mostrarOG']:
        inp.show()

    if vargs['ej'] == 'hist':
        for h, c in zip(histRGB(im), rgb):
            print("%s: %s" % (c, h))
    elif vargs['ej'] == 'eqRGB':
        if vargs['unChan']:
            imRes = Image.fromarray(unChan(equalPorCanal(chan(im))))
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s.png" % (entrada, vargs['ej']))
        else:
            for res, c in zip(equalPorCanal(chan(im)), rgb):
                imRes = Image.fromarray(res, mode = "L")
                if vargs['mostrar']:
                    imRes.show()
                else:
                    imRes.save("Resultados/%s-%s-%s.png" % (entrada, vargs['ej'], c))
    elif vargs['ej'] == 'despHSI':
        for res, c in zip(despHSI(im), hsi):
            if c == 'H':
                imRes = Image.fromarray(res, mode = "RGB")
            else:
                imRes = Image.fromarray(res, mode = "L")
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s-%s.png" % (entrada, vargs['ej'], c))
    elif vargs['ej'] == 'eqI':
        imRes = Image.fromarray(equalI(im), mode = "RGB")
        if vargs['mostrar']:
            imRes.show()
        else:
            imRes.save("Resultados/%s-%s.png" % (entrada, vargs['ej']))
    elif vargs['ej'] == 'transfHSI':
        if vargs['transf'] == 'logI':
            imRes = Image.fromarray(logI(im), mode = "RGB")
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s.png" % (entrada, vargs['transf']))
        elif vargs['transf'] == 'multS':
            imRes = Image.fromarray(multS(im, vargs['c']), mode = "RGB")
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s-%.2f.png" % (entrada, vargs['transf'], vargs['c']))
        elif vargs['transf'] == 'addH':
            imRes = Image.fromarray(rotH(im, vargs['c']), mode = "RGB")
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s-%.2f.png" % (entrada, vargs['transf'], vargs['c']))
        elif vargs['transf'] == 'reflSLim':
            imRes = Image.fromarray(reflSLim(im, vargs['c']), mode = "RGB")
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s-%.2f.png" % (entrada, vargs['transf'], vargs['c']))
        elif vargs['transf'] == 'partH':
            imRes = Image.fromarray(partH(im, vargs['N']), mode = "RGB")
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s-%d.png" % (entrada, vargs['transf'], vargs['N']))
        elif vargs['transf'] == 'partS':
            imRes = Image.fromarray(partS(im, vargs['N']), mode = "RGB")
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s-%d.png" % (entrada, vargs['transf'], vargs['N']))
        elif vargs['transf'] == 'blur':
            sel = [vargs['Hb'], vargs['Sb'], vargs['Ib']]
            if not any(sel):
                print("No se especificó ningún canal. Ver ./p2.py inp transfHSI blur -h.")
                exit()
            imRes = Image.fromarray(blur(im, sel, vargs['k']), mode = "RGB")
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s-%d-%d%d%d.png" % (entrada, vargs['transf'], vargs['k'], *sel))
        elif vargs['transf'] == 'grain':
            sel = [vargs['Hb'], vargs['Sb'], vargs['Ib']]
            if not any(sel):
                print("No se especificó ningún canal. Ver ./p2.py inp transfHSI grain -h.")
                exit()
            imRes = Image.fromarray(grain(im, sel, vargs['mu'], vargs['sigma']), mode = "RGB")
            if vargs['mostrar']:
                imRes.show()
            else:
                imRes.save("Resultados/%s-%s-%.2f-%.2f-%d%d%d.png" %
                           (entrada, vargs['transf'], vargs['mu'], vargs['sigma'], *sel))
        else:
            print("Mal pasados los parámetros. Ver ./p2.py inp transfHSI --help")
    else:
        print("Mal pasados los parámetros. Ver ./p2.py --help")
