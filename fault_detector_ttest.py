"""
Funcion que detecta cambios en la media en un grupo de muestras de tamano N en base a la prueba estadistica t-test.
Para ello se va comparando una ventana movil de tamano N con el conjunto de valores que anteceden temporalmente dicha
ventana.

Parametros de entrada:

lista: Vector de dimensiones 1 x N sobre el cual se hara la prueba
stand_dv: Desviacion estandar de los valores del vector lista
conf_lev: Nivel de confiabilidad del 0 - 1 que se quiere tener para la clasificacion de los resultados
delta_mean: Variacion de la media que se quisiera detectar como limite
N: Numero de muestras a ser usadas como ventana de prueba. Si es 'auto', la funcion calcula automaticamente el numero de
    muestras necesarias para dectear dicho cambio en base a conf_lev y stand_dv.

Parametros de salida:

lista_falla: valores del vector lista para los cuales se detecto variacion en la media
falla_bool: Vector de longitud 1 x N de 0 y 1 que coincide con lista de modo que para cada posicion se asigana un 1 si
            para la misma posicion en lista el valor es considerado anomalo o un 0 en caso contrario
fault_counter: Numero de valores anomalos detectados
N: Tamano de la ventana de muestras sobre los cuales se aplico t-test. En caso de que se requiera saber cual fue el
    valor calculado en modo N = 'auto'

"""

def fault_detector_ttest(lista, stand_dv, conf_lev, delta_mean, N = 'auto'):
    from math import ceil
    import scipy.stats
    import scipy as sp
    import math
    import numpy as np

    if N == 'auto':
        N = ceil((stand_dv*sp.stats.norm.ppf((1+conf_lev)/2)/delta_mean)**2)
    falla_bool = np.zeros(len(lista))
    if 2*N <= len(lista):
        cont = 0
        while N+cont <= len(lista) - N:
            ttest_issermann = (lista[:N+cont].mean()-lista[N+cont:2*N+cont].mean())*\
                                math.sqrt(N*N*(2*N - 2)/(2*N))/math.sqrt((2*N-2)*(stand_dv**2))
            if abs(ttest_issermann) > sp.stats.t.ppf(conf_lev, 2*N-2):
                falla_bool[N+cont:2*N+cont] = np.ones(N)
            cont += 1
    else:
        print('N debe ser menor a la longitud del vector lista')
    fault_counter = len(falla_bool[falla_bool == 1])
    lista_falla = lista[falla_bool == 1]
    return lista_falla, falla_bool, fault_counter, N

