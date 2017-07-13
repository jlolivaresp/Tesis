"""
Funcion que genera un vector de valores para resistencia hidraulica de una valvula con un incremento de r_inicial a
r_final desde un tiempo inicial t_inicial hasta un tiempo final t_final
"""

'''

Parametros de entrada

    r_inicial: Valor inicial de la resistencia hidraulica
    r_final: Valor final de la resistencia hidraulica
    longitud: Longitud del vector
    t_inicial: Tiempo en que inicia el cambio gradual
    t_final: Tiempo en que finaliza el cambio gradual
    paso: Paso de integracion

Parametros de salida

    R: vector de resistencias hidraulicas

'''


def aumento_gradual_r(r_inicial, r_final, longitud, t_inicial, t_final, paso):

    import numpy as np

    R = np.ones(longitud/paso)*r_inicial
    R[t_inicial:t_final] = np.arange(r_inicial, r_final, (r_final-r_inicial)/((t_final-t_inicial)/paso))
    R[t_final:] = np.ones(longitud-t_final)*r_final

    return R
