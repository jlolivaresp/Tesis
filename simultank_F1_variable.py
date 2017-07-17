"""
Creamos una funcion llamada simultank que resuelve la ecuacion diferencial que describe el comportamiento
de la altura del tanque en funcion del tiempo.

Los parametros de entrada de nuestra funcion son:

area: Area transversal del tanque
nivel_inicial: Altura inicial del contenido del tanque
resist_hidraulica: Resistencia hidraulica de la valvula de salida
caudal_entrada: Caudal de entrada
tiempo_inicial: Tiempo inicial desde el cual se quiere simular el comportamiento
tiempo_final: Tiempo final desde el cual se quiere simular el comportamiento
paso: El incremento de tiempo entre ti y tf

Los parametros de salida son:
vector_H: vector de alturas H en funcion del tiempo
"""

def simultank(area, nivel_inicial, resist_hidraulica, caudal_entrada, tiempo_inicial, tiempo_final, paso):

    import numpy as np

    vector_tiempo = np.arange(tiempo_inicial, tiempo_final, paso)

    K1 = paso*(caudal_entrada[0]/area - nivel_inicial/(area*resist_hidraulica[0]))
    K2 = paso*(caudal_entrada[0]/area - (nivel_inicial + K1/2)/(area*resist_hidraulica[0]))
    K3 = paso*(caudal_entrada[0]/area - (nivel_inicial + K2/2)/(area*resist_hidraulica[0]))
    K4 = paso*(caudal_entrada[0]/area - (nivel_inicial + K3)/(area*resist_hidraulica[0]))

    H_mas_1 = nivel_inicial + (1/6)*(K1 + 2*K2 + 2*K3 + K4)
    vector_H = np.array([nivel_inicial, H_mas_1])

    iteraciones = len(vector_tiempo)
    contador = 1

    while contador < iteraciones - 1:

        ''' Aplicamos RK4
        '''

        K1 = paso*(caudal_entrada[contador]/area - H_mas_1/(area*resist_hidraulica[contador]))
        K2 = paso*(caudal_entrada[contador]/area - (H_mas_1 + K1/2)/(area*resist_hidraulica[contador]))
        K3 = paso*(caudal_entrada[contador]/area - (H_mas_1 + K2/2)/(area*resist_hidraulica[contador]))
        K4 = paso*(caudal_entrada[contador]/area - (H_mas_1 + K3)/(area*resist_hidraulica[contador]))

        H_mas_1 = np.array([nivel_inicial + (1/6)*(K1 + 2*K2 + 2*K3 + K4)])

        contador += 1

        vector_H = np.append(vector_H, H_mas_1)

    return vector_H