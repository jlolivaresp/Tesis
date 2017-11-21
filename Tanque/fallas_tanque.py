"""Este script genera los data sets de fallas del tanque"""

from simulador_nivel_tanque import simultank
import fault_2 as f
import datos_tanque as datos
import numpy as np
import pandas as pd
from exploracion_datos_nivel_sin_falla import df_tanque

np.random.seed(0)

# Predefinimos el DataFrame al que iremos anexando los valores con falla para distintas condiciones

# Las columnas de caracteristica_1 y caracteristica_2 representan para los casos:
# Deriva -> Pendiente de deriva y tiempo en que finaliza la deriva
# Pulso -> Desviacion estandar del pulso y numero de muestras con pulso
# Varianza -> Desviacion estandar de la varianza y tiempo en que finaliza la varianza
columnas = ['tipo_falla', 'tiempo', 'caracateristica_1', 'caracateristica_2', 'nivel', 'condicion_falla']
df_tanque_falla = pd.DataFrame(columns=columnas)

'''____________________________________________ Falla de deriva _____________________________________________________'''

# Columna de tipo de falla
tipo = np.array(['deriva' for i in datos.t_sim_ee])

# Iteracion para cada valor de resistencia hidraulica y tiempo de finalizacion de falla que queremos probar
for r in datos.delta_r:
    # Generamos la columna de caracteristica_1
    caracteristica_1 = np.ones((len(datos.t_sim_ee)))*r

    for t_f in datos.t_f_falla_deriva:
        # Generamos la columna de caracteristica_2
        caracteristica_2 = np.ones((len(datos.t_sim_ee)))*t_f
        # Generamos la falla
        r_falla, pendiente_deriva, r_falla_bool = f.fault_generator(datos.r_sim).drift(start=datos.t_i_falla_deriva,
                                                                                       stop=t_f, step=datos.paso,
                                                                                       change=r)
        nivel_falla_deriva = simultank(area=datos.area, nivel_inicial=datos.nivel_inicial, resist_hidraulica=r_falla,
                                       caudal_entrada=datos.q_sim, tiempo_inicial=datos.t_i, tiempo_final=datos.t_f,
                                       paso=datos.paso, analitic_sol=False) + np.random.normal(0, 0.02, len(r_falla))

        # Consideramos unicamente los datos de tiempo mayor al tiempo de establecimiento al 2%
        nivel_falla_drift_ee = nivel_falla_deriva[datos.tss_2/datos.paso:]
        r_falla_bool_ee = r_falla_bool[datos.tss_2/datos.paso:]

        # Creamos un DataFrame con los resultados
        df_tanque_falla_r = pd.DataFrame({'tiempo': datos.t_sim_ee, 'tipo_falla': tipo,
                                          'caracateristica_1': caracteristica_1, 'caracateristica_2': caracteristica_2,
                                          'nivel': nivel_falla_drift_ee, 'condicion_falla': r_falla_bool_ee},
                                         columns=columnas)

        # Anexamos los datos de falla nuevos al DataFrame general
        df_tanque_falla = df_tanque_falla.append(df_tanque_falla_r, ignore_index=True)

'''______________________________________________ Falla de pulso ____________________________________________________'''

# Columna de tipo de falla
tipo = ['pulso' for j in datos.t_sim_ee]

# Nivel del tanque a condiciones normales
nivel = simultank(area=datos.area, nivel_inicial=datos.nivel_inicial, resist_hidraulica=datos.r_sim,
                  caudal_entrada=datos.q_sim, tiempo_inicial=datos.t_i, tiempo_final=datos.t_f,
                  paso=datos.paso, analitic_sol=False) + np.random.normal(0, 0.02, len(datos.r_sim))
# Consideramos unicamente los datos de tiempo mayor al tiempo de establecimiento al 2%
nivel_pulso_ee = nivel[datos.tss_2/datos.paso:]

# Iteracion para cada amplitud y numero de muestras con falla que queremos probar
for amp in datos.amplitud_pulso:
    # Generamos la columna de caracteristica_1
    caracteristica_1 = np.ones((len(datos.t_sim_ee)))*amp

    for N in datos.N_pulsos:
        # Generamos la columna de caracteristica_2
        caracteristica_2 = np.ones((len(datos.t_sim_ee)))*N

        # Generamos la falla sobre el conjunto de valores de nivel sin falla
        nivel_falla_pulso, frac_muestras_falla, pulso_bool = \
            f.fault_generator(nivel_pulso_ee).random_pulse(start=datos.t_i_falla_pulso, stop=datos.t_f_falla_pulso,
                                                           step=datos.paso, N=N, amplitude=amp,
                                                           random_seed=datos.random_seed, mode=datos.modo)
        # Creamos un DataFrame con los resultados
        df_tanque_falla_pulso = pd.DataFrame({'tiempo': datos.t_sim_ee, 'tipo_falla': tipo,
                                             'caracateristica_1': caracteristica_1,
                                              'caracateristica_2': caracteristica_2, 'nivel': nivel_falla_pulso,
                                              'condicion_falla': pulso_bool}, columns=columnas)

        # Anexamos los datos de falla nuevos al DataFrame general
        df_tanque_falla = df_tanque_falla.append(df_tanque_falla_pulso, ignore_index=True)

print(df_tanque_falla)
