"""
Este programa genera un vector de Caudales tipo onda cuadrada entre Qmin y Qmax, de longitud n
"""

'''
Los parametros de entrada son:

Qmin: Caudal minimo
Qmax: Caudal maximo
T: Periodo
longitud: Longitud del vector de Caudales

El parametro de salida es:
vector_caudal
'''

# Definimos la funcion

def vector_caudal_cuadrado(Qmin, Qmax, T, paso, longitud):

    import numpy as np                                              # Importamos numpy para trabajar con arreglos

    contador = 0                                                    # Predefinimos un contador para el ciclo
    vector_caudal = np.array([Qmin])                                # Predefinimos el vector vector_caudal empezando
                                                                    # por Qmin en la posicion 0
    vector_tiempo = np.arange(0.0, longitud, paso)

    n = 1

# Iniciamos el ciclo que ira agregando Qmin (en posiciones impares) y Qmax (en posiciones pares) al vector hasta que se
# alcance la longitud deseada

    while contador < len(vector_tiempo) - 1:
        posicion = vector_tiempo[contador]
        nperiodo = n * T
        if posicion == nperiodo:
            n += 1
        elif n%2 != 0:
            vector_caudal = np.append(vector_caudal, Qmax)
            contador += 1
        elif n%2 == 0:
            vector_caudal = np.append(vector_caudal, Qmin)
            contador += 1

    return vector_caudal
