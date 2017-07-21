"""
Funcion que detecta cambios en la varianza en un grupo de muestras de tamano N en base a la prueba estadistica F-test.
Para ello se va comparando una ventana movil de tamano N con el conjunto de valores que anteceden temporalmente dicha
ventana.

Parametros de entrada:

lista: Vector de dimensiones 1 x N sobre el cual se hara la prueba
delta_var: Variacion de la varianza que se quisiera detectar como limite
conf_lev: Nivel de confiabilidad del 0 - 1 que se quiere tener para la clasificacion de los resultados
N: Numero de muestras a ser usadas como ventana de prueba. Si es 'auto', la funcion calcula automaticamente el numero de
    muestras necesarias para dectear dicho cambio en base a conf_lev y stand_dv.

Parametros de salida:

lista_falla: valores del vector lista para los cuales se detecto variacion en la varianza
falla_bool: Vector de longitud 1 x N de 0 y 1 que coincide con lista de modo que para cada posicion se asigana un 1 si
            para la misma posicion en lista el valor es considerado anomalo o un 0 en caso contrario
fault_counter: Numero de valores anomalos detectados
N: Tamano de la ventana de muestras sobre los cuales se aplico F-test. En caso de que se requiera saber cual fue el
    valor calculado en modo N = 'auto'

"""


def fault_detector_F(lista, delta_var, conf_lev, N = 'auto'):

    import scipy.stats
    import scipy as sp
    import numpy as np

    falla_bool = np.zeros(len(lista))
    if np.std(lista) > 0.0001:
        if N == 'auto':
            N = 1
            X_2_table = sp.stats.chi2.ppf(q=conf_lev,df=N)
            X_2 = ((lista.std()+delta_var)/np.std(lista))**2

            while X_2 < X_2_table:
                N += 1
                X_2_table = sp.stats.chi2.ppf(q=conf_lev,df=N)
                X_2 = (N - 1)*(((lista.std()+delta_var)/np.std(lista))**2)
                if N > len(lista)/2:
                    print('No se cuenta con suficientes valores para detectar un '
                          'cambio en la varianza de {}, pruebe indicando el valor de N'.format(delta_var))
                    break

        cont = 0
        while N+cont <= len(lista) - N:
            dfn = len(lista[:N+cont])
            if np.std(lista[N+cont:2*N+cont]) > 0.0001:
                F_N1_N2 = (np.std(lista[:N+cont])/np.std(lista[N+cont:2*N+cont]))**2
                F_table = sp.stats.f.ppf(q=conf_lev, dfn=dfn, dfd=N)

                if F_N1_N2 < F_table:
                    falla_bool[N+cont:2*N+cont] = np.ones(N)
            cont += 1
    fault_counter = len(falla_bool[falla_bool == 1])
    lista_falla = lista[falla_bool == 1]
    return lista_falla, falla_bool, fault_counter, N