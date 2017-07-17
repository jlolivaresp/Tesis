"""
Programa que genera un vector de caudales
"""

'''

Parametros de entrada

    set_point: Valor que tendra el caudal a lo largo del vector
    longitud: Longitud del vector
    paso: Paso de integracion
    factpr_ruido: Amplitud del ruido gausiano
    ruido: De ser True, el vector tendra ruido gausiano.

Parametros de salida

    Q: Vector de caudales

'''


def caudal(set_point, longitud, paso, factor_ruido=0.01, ruido=False):

    import numpy as np

    if int(longitud/paso) - longitud/paso == 0:
        Q = np.ones(longitud/paso)*set_point

        if ruido:
            np.random.seed(0)
            Q += set_point*factor_ruido*np.random.normal(0, 1, longitud/paso)

    else:
        print('longitud/paso debe ser un entero')
        Q=[]

    return Q
