"""Este script genera los data sets de fallas del tanque"""

from simultank_F1_variable import simultank
import fault_2 as f
import datos_tanque as datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

# Predefinimos el DataFrame al que iremos anexando los valores con falla para distintas condiciones

# Las columnas de caracteristica_1 y caracteristica_2 representan para los casos:
# Deriva -> Pendiente de deriva y tiempo en que finaliza la deriva
# Pulso -> Desviacion estandar del pulso y numero de muestras con pulso
# Varianza -> Desviacion estandar de la varianza y tiempo en que finaliza la varianza
columnas = ['tipo_falla', 'caracteristica_1', 'caracteristica_2', 'intensidad_falla', 'tiempo',
            'nivel', 'caudal', 'nivel_sin_falla', 'condicion_falla']
df_tanque_falla = pd.DataFrame(columns=columnas)
# Nivel del tanque a condiciones normales
nivel = simultank(area=datos.area, nivel_inicial=datos.nivel_inicial, resist_hidraulica=datos.r_sim,
                  caudal_entrada=datos.q_sim, tiempo_inicial=datos.t_i, tiempo_final=datos.t_f,
                  paso=datos.paso, analitic_sol=False) + np.random.normal(0, 0.005, len(datos.t_sim))
nivel_ee = nivel[int(datos.tss_2):]
print(datos.tss_2)
'''____________________________________________ Falla de deriva _____________________________________________________'''

# Columna de tipo de falla
tipo = np.array(['deriva' for i in datos.t_sim_ee])

# Iteracion para cada valor de resistencia hidraulica y tiempo de finalizacion de falla que queremos probar
for h in datos.delta_h:
    # Generamos la columna de caracteristica_1
    caracteristica_1 = np.ones((len(datos.t_sim_ee)))*h
    for t_f in datos.t_f_falla_deriva:
        # Generamos la columna de caracteristica_2
        caracteristica_2 = np.ones((len(datos.t_sim_ee)))*t_f
        # Generamos la falla
        nivel_falla_deriva, pendiente_deriva, h_falla_bool = f.fault_generator(nivel).drift(start=datos.t_i_falla_deriva,
                                                                                            stop=t_f, step=datos.paso,
                                                                                            change=h)
        intensidad_falla = np.ones((len(datos.t_sim_ee)))*pendiente_deriva

        # Consideramos unicamente los datos de tiempo mayor al tiempo de establecimiento al 2%
        nivel_falla_drift_ee = nivel_falla_deriva[int(datos.tss_2):]
        h_falla_bool_ee = h_falla_bool[int(datos.tss_2):]
        q_sim_ee = datos.q_sim[int(datos.tss_2):]
        # Creamos un DataFrame con los resultados
        df_tanque_falla_h = pd.DataFrame({'tiempo': datos.t_sim_ee, 'tipo_falla': tipo,
                                          'caracteristica_1': caracteristica_1, 'caracteristica_2': caracteristica_2,
                                          'intensidad_falla':intensidad_falla, 'nivel': nivel_falla_drift_ee,
                                          'caudal': q_sim_ee,
                                          'nivel_sin_falla': nivel_ee, 'condicion_falla': h_falla_bool_ee},
                                         columns=columnas)

        # Anexamos los datos de falla nuevos al DataFrame general
        df_tanque_falla = df_tanque_falla.append(df_tanque_falla_h, ignore_index=True)

'''______________________________________________ Falla de pulso ____________________________________________________'''

# Columna de tipo de falla
tipo = ['pulso' for j in datos.t_sim_ee]

# Consideramos unicamente los datos de tiempo mayor al tiempo de establecimiento al 2%
nivel_pulso_ee = nivel[int(datos.tss_2):]

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
        intensidad_falla = np.ones((len(datos.t_sim_ee)))*frac_muestras_falla*amp
        q_sim_ee = datos.q_sim[int(datos.tss_2):]
        # Creamos un DataFrame con los resultados
        df_tanque_falla_pulso = pd.DataFrame({'tiempo': datos.t_sim_ee, 'tipo_falla': tipo,
                                             'caracteristica_1': caracteristica_1, 'caracteristica_2': caracteristica_2,
                                              'intensidad_falla':intensidad_falla, 'nivel': nivel_falla_pulso,
                                              'caudal': q_sim_ee,
                                              'nivel_sin_falla': nivel_ee, 'condicion_falla': pulso_bool},
                                             columns=columnas)

        # Anexamos los datos de falla nuevos al DataFrame general
        df_tanque_falla = df_tanque_falla.append(df_tanque_falla_pulso, ignore_index=True)

'''_____________________________________________ Falla de varianza __________________________________________________'''

# Columna de tipo de falla
tipo = ['varianza' for k in datos.t_sim_ee]

# Usamos los mismo datos de nivel a condiciones normales que usamos en la seccion de pulso
nivel_var_ee = np.copy(nivel_pulso_ee)

# Iteracion para cada amplitud y tiempo de finalizacion de falla que queremos probar
for amp in datos.amplitud_var:
    # Generamos la columna de caracteristica_1
    caracteristica_1 = np.ones((len(datos.t_sim_ee)))*amp

    for t_f in datos.t_f_falla_var:
        # Generamos la columna de caracteristica_2
        caracteristica_2 = np.ones((len(datos.t_sim_ee)))*t_f

        # Generamos la falla sobre el conjunto de valores de nivel sin falla
        nivel_falla_var, frac_muestras_falla, var_bool = \
            f.fault_generator(nivel_var_ee).variance(start=datos.t_i_falla_var, stop=t_f, step=datos.paso,
                                                     stand_dev=amp, random_seed=datos.random_seed)
        intensidad_falla = np.ones((len(datos.t_sim_ee)))*frac_muestras_falla*amp
        q_sim_ee = datos.q_sim[int(datos.tss_2):]
        # Creamos un DataFrame con los resultados
        df_tanque_falla_var = pd.DataFrame({'tiempo': datos.t_sim_ee, 'tipo_falla': tipo,
                                           'caracteristica_1': caracteristica_1,
                                            'caracteristica_2': caracteristica_2, 'intensidad_falla': intensidad_falla,
                                            'nivel': nivel_falla_var, 'nivel_sin_falla': nivel_ee,
                                            'caudal': q_sim_ee,
                                            'condicion_falla': var_bool}, columns=columnas)

        # Anexamos los datos de falla nuevos al DataFrame general
        df_tanque_falla = df_tanque_falla.append(df_tanque_falla_var, ignore_index=True)

print(df_tanque_falla.describe())