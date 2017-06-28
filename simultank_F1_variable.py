"""
Creamos una funcion llamada Simultank que resuelve la ecuacion diferencial que describe el comportamiento
de la altura del tanque en funcion del tiempo.

Los parametros de entrada de nuestra funcion son:

A: Area transversal del tanque
Ho: Altura inicial del contenido del tanque
R: Factor de descarga
F1: Caudal de entrada
ti: Tiempo inicial desde el cual se quiere simular el comportamiento
tf: Tiempo final desde el cual se quiere simular el comportamiento
paso: El incremento de tiempo entre ti y tf

Los parametros de salida son:
vector_H: vector de alturas H en funcion del tiempo
"""
import numpy as np
def simul_tank_caudal_variable(A, Ho, R, F1, ti, tf, paso):         # Creamos nuestra funcion con los parametros de entrada


    vector_tiempo = np.arange(ti, tf, paso)         # Generamos el vector de intervalos de tiempo
    vector_H = np.array([Ho])                       # Creamos el vector de altura que estara en funcion del tiempo
    vector_K1 = np.array([0])                       # K1 y K2 son los parametros del metodo RK2
    vector_K2 = np.array([0])

    iteraciones = len(vector_tiempo)
    contador = 0

    while contador < iteraciones - 1:               # Este ciclo ira iterando para cada valor de tiempo

        ''' Aplicamos RK2
        '''
        vector_K1_mas_1 = paso*((F1[contador]/A) - (Ho /(A*R[contador])))
        vector_K2_mas_1 = paso*((F1[contador]/A) - (Ho + vector_H[contador] + vector_K1[contador])/(A*R[contador]))
        H_mas_1 = vector_H[contador] + 0.5*(vector_K1[contador] + vector_K2[contador])

        contador += 1

        vector_H = np.append(vector_H, H_mas_1)     # El vector de altura va creciendo para cada t
        vector_K1 = np.append(vector_K1,vector_K1_mas_1 )
        vector_K2 = np.append(vector_K2, vector_K2_mas_1)

    return vector_H