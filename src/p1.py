from PIL import Image
import numpy as np
from scipy.stats import norm
from sys import argv

#Uso int para tener precisión arbitraria
#1.a)
def suma(im1, im2, tipo = int):
    #Aplico la suma celda a celda
    return np.vectorize(lambda x, y: x + y, otypes = [tipo])(im1, im2)

#1.a)
def resta(im1, im2, tipo = int):
    return np.vectorize(lambda x, y: x - y, otypes = [tipo])(im1, im2)

#1.a)
#No está claro en la consigna si es celda por celda o es el producto matricial
#Hago celda por celda porque me parece que tiene más sentido en el contexto de imágenes
def producto(im1, im2, tipo = int):
    return np.vectorize(lambda x, y: x * y, otypes = [tipo])(im1, im2)

#1.b)
def escalar(im, k, tipo = int):
    return np.vectorize(lambda x: x * k, otypes = [tipo])(im)

#1.c)
def crd(im, R, L = 256, tipo = int):
    return np.vectorize(lambda x: np.log10(x + 1) *
                         ((L - 1) / np.log10(R + 1)),
                         otypes = [tipo])(im)

#2)
def neg(im, L = 256, tipo = int):
    return np.vectorize(lambda x: L - 1 - x, otypes = [tipo])(im)

#3)
def threshold(im, u, L = 256, tipo = int):
    return np.vectorize(lambda x: 0 if x < u else L - 1, otypes = [tipo])(im)

#4)
#b es la cantidad de bits por celda de la imagen
def bitplane(im, b, tipo = np.bool):
    #inicializo un array por bit
    imRes = [np.empty_like(im, dtype = tipo) for k in range(b)]
    for k, v in np.ndenumerate(im):
        #El formato de bin(x) es "0b(0|1)*"
        #Elimino el "0b"
        binario = bin(v)[2:]
        #Paddeo con 0s para que el string tenga tamaño exactamente b
        binario = "0" * (b - len(binario)) + binario
        #Notar que el string está al revés, con binario[0] siendo el bit más significativo
        #[-(kR + 1)] recupera el orden deseado
        #e.g., con kR == 0, binario[-(kR + 1)] es el bit menos significativo
        for kR, vR in enumerate(imRes):
            #Tengo que pasar a int para que bool de el resultado esperado para cada caracter
            vR[k] = tipo(int(binario[-(kR + 1)]))
    return imRes

#5)
def histograma(im):
    h = np.zeros(256, dtype = int)
    for v in np.nditer(im):
        h[v] += 1
    return h

#Función general para la transformada
#Target es la p.m.f. deseada
#h es el histograma que se manipulará
def transformada(im, target, h = None, L1 = 256, tipo = np.uint8):
    L2 = 256
    im = np.vectorize(lambda x: x * 255, otypes = [np.uint8])(im)
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

#6)
#Defino el histograma por partes deseado y realizo la transformada
def aumContr(im, r1, r2, f1, f2, f3, L = 256):
    target = np.empty(L)
    for i in range(r1):
        target[i] = f1(i)
    for i in range(r1, r2):
        target[i] = f2(i)
    for i in range(r2, L):
        target[i] = f3(i)
    return transformada(im, target, L1 = L)

#Ejemplo de aumento de contraste
def aumContrEj1(im):
    return aumContr(im,
                    256 // 8,
                    256 * 7 // 8,
                    lambda x: 2 / 256,
                    lambda x: 4 / (256 * 6),
                    lambda x: 2 / 256)

#Ejemplo de aumento de contraste
def aumContrEj2(im):
    return aumContr(im,
                    256 // 16,
                    256 * 15 // 16,
                    lambda x: 4 / 256,
                    lambda x: 8 / (256 * 14),
                    lambda x: 4 / 256)
#7)
#Transformo con distribución uniforme como target
def equal(im, h = None, L = 256, tipo = np.uint8):
    return transformada(im, np.ones(L) / L, h, L1 = L, tipo = tipo)

#9)
#Transformo con distribución normal como target
def gaussTransf(im, mu, sigma, tipo = np.uint8):
    target = norm.pdf(range(256), mu, sigma)
    #Normalizo ya que no va a sumar exactamente 1 en range(256)
    target /= np.sum(target)
    return transformada(im, target, tipo = np.uint8)

#10)
#Modifico el histograma de la imagen y luego ecualizo con dicho histograma
#Tomo los defaults de λ y γ de forma tal que equal(im) == equalMoño(im)
def equalMoño(im, Lambda = 0, Gamma = 0, L = 256, tipo = np.uint8):
    uL = np.ones(L) * Lambda / L
    h0 = histograma(im)
    D = np.eye(L - 1, L) * (-1) + np.eye(L - 1, L, k = 1)
    hm = np.dot(np.linalg.inv((1 + Lambda) * np.eye(L, L) +
                       Gamma * np.dot(D.transpose(), D)),
                np.add(h0, uL))
    return equal(im, hm, L = L, tipo = np.uint8)

if __name__ == "__main__":
    entrada = argv[1]
    inp = np.array(Image.open(entrada).convert(mode = "L"))
    ej = argv[2]
    if "1.a" in ej:
        entrada2 = argv[3]
        salida = argv[4]
        inp2 = np.array(Image.open(entrada2).convert(mode = "L"))
        if "suma" in ej:
            Image.fromarray(suma(inp, inp2, tipo = np.uint8)).save("%s.png" % salida)
        elif "resta" in ej:
            Image.fromarray(resta(inp, inp2, tipo = np.uint8)).save("%s.png" % salida)
        elif "producto" in ej:
            Image.fromarray(producto(inp, inp2, tipo = np.uint8)).save("%s.png" % salida)
    elif "1.b" == ej:
        k = argv[3]
        salida = argv[4]
        Image.fromarray(escalar(inp, float(k), tipo = np.uint8)).save("%s.png" % salida)
    elif "1.c" == ej:
        R = argv[3]
        salida = argv[4]
        Image.fromarray(crd(inp, int(k), tipo = np.uint8)).save("%s.png" % salida)
    elif "2" == ej:
        salida = argv[3]
        Image.fromarray(neg(inp, tipo = np.uint8)).save("%s.png" % salida)
    elif "3" == ej:
        u = argv[3]
        salida = argv[4]
        Image.fromarray(threshold(inp, int(u), tipo = np.uint8)).save("%s.png" % salida)
    elif "4" == ej:
        salida = argv[3]
        out = bitplane(inp, 8)
        for k, v in enumerate(out):
            Image.fromarray(threshold(v, 1, tipo = np.uint8), mode = "L").save("%s-bit:%d.png" % (salida, k))
    elif "5" == ej:
        print(histograma(inp))
    elif "6.1" == ej:
        salida = argv[3]
        Image.fromarray(aumContrEj1(inp)).save("%s.png" % salida)
    elif "6.2" == ej:
        salida = argv[3]
        Image.fromarray(aumContrEj2(inp)).save("%s.png" % salida)
    elif "7" == ej:
        salida = argv[3]
        Image.fromarray(equal(inp)).save("%s.png" % salida)
    elif "8" == ej:
        salida = argv[3]
        Image.fromarray(equal(equal(inp))).save("%s.png" % salida)
    elif "9" == ej:
        mu = argv[3]
        sigma = argv[4]
        salida = argv[5]
        Image.fromarray(gaussTransf(inp, float(mu), float(sigma))).save("%s.png" % salida)
    elif "10" == ej:
        Lambda = argv[3]
        Gamma = argv[4]
        salida = argv[5]
        Image.fromarray(equalMoño(inp, float(Lambda), float(Gamma))).save("%s.png" % salida)
