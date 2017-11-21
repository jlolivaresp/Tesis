"""Este script genera los datos de simulacion del tanque sin falla y con falla"""

import numpy as np

"""__________________________________________ Variables Generales ___________________________________________________"""

'''Caracteristicas del Tanque'''

q = 10                                                  # Caudal de entrada [m^3/h]
radio = 0.5                                             # Radio del tanque [m]
area = np.pi * radio**2                                 # Area transversal del tanque [m^2]
r = 0.2                                                 # Valor inicial de la resistencia hidraulica
nivel_inicial = 0                                       # Nivel inicial del tanque [m]

'''Datos de la Simulacion'''

tiempo = 24                                             # Tiempo de simulacion [h] (240h -> 24d)
paso = 0.1                                              # Paso de integracion [h]
decimal = str(paso)[::-1].find('.')                     # Numero de decimales del paso
t_i = 0                                                 # Tiempo inciial de la simulacion [h]
t_f = tiempo                                            # Tiempo final de la simulacion [h]
tss_2 = int(np.round(4*area*r, decimal)/paso)           # Tiempo de establecimiento del nivel
t_sim = np.arange(t_i, t_f, paso)                       # Vector de rangos de tiempo de la simulacion
t_sim_ee = t_sim[tss_2/paso:]                           # Vector de tiempo de simulacion en estado estacionario
q_sim = np.ones(tiempo/paso)*q                          # Vector de caudal para cada tiempo
r_sim = np.ones(tiempo/paso)*r                          # Vector de resistencia hidraulica para cada tiempo

"""___________________________________________ Variables de las Fallas ______________________________________________"""

'''Falla de Deriva'''

t_i_falla_deriva = 8                                        # Tiempo de inicio de la falla [h]
t_f_falla_deriva = [8.2, 12.1, 16.1, 24]                  # Tiempos de finalizacion de la falla [h]
delta_r = [0.0001, 0.0005, 0.001]

'''Falla de Pulso'''

t_i_falla_pulso = 8                                         # Tiempo de inicio de la falla [h]
t_f_falla_pulso = 24                                        # Tiempos de finalizacion de la falla [h]
N_pulsos = [16, 40, 80, 100]                                # Numero de muestras con pulso
amplitud_pulso = [0.02, 0.05, 0.1]                          # Desviacion estandar de los pulsos
random_seed = 0                                             # Valor semilla del generador de numeros aleatorios
modo = 'random'                                             # Amplitud de pulsos de distribucion normal aleatoria
