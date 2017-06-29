"""
Este programa genera un vector de Caudales tipo onda cuadrada entre Qmin y Qmax, de longitud n
"""

'''
Los parametros de entrada son:

Qmin: Caudal minimo
Qmax: Caudal maximo
longitud: Longitud del vector de Caudales
T: Periodo
paso: Paso de intervalos de tiempo

El parametro de salida es:
vector_caudal
'''

# Definimos la funcion

def vector_caudal_triangular(Qmin, Qmax, T, paso, longitud):

    import numpy as np
    import matplotlib.pyplot as plt

    vector_tiempo = np.arange(0.0, longitud, paso)

    vector_caudal = np.array([Qmin])

    contador = 0
    n = 1
    m_neg = (Qmin - Qmax)/T
    m_pos = (Qmax - Qmin)/T
    T = round(T,0)
    while contador < len(vector_tiempo) - 1:
        posicion = vector_tiempo[contador]
        nperiodo = n*T
        if posicion == nperiodo:
            n += 1
        elif n%2 == 0:
            arista_negativa = m_neg*posicion - m_neg*(n - 1)*T + Qmax
            vector_caudal = np.append(vector_caudal, arista_negativa)
            contador += 1
        elif n%2 != 0:
            arista_positiva = m_pos*posicion - m_pos*(n - 1)*T + Qmin
            vector_caudal = np.append(vector_caudal, arista_positiva)
            contador += 1
    return vector_caudal