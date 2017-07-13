"""
Programa que genera un vector de caudales
"""

'''

Parametros de entrada

    set_point: Valor que tendra el caudal a lo largo del vector
    longitud: Longitud del vector
    paso: Paso de integracion
    ruido: De ser True, el vector tendra ruido gausiano.

Parametros de salida

    Q: Vector de caudales

'''


def caudal(set_point, longitud, paso, ruido=False):

    import numpy as np

    Q = np.ones(longitud/paso)*set_point

    if ruido:
        np.random.seed(0)
        Q += set_point*0.01*np.random.normal(0, 1, longitud)

    return Q
