"""Este script genera los data sets de fallas del tanque"""

from simulador_nivel_tanque import simultank
import fault_2 as f
import datos_tanque as datos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from exploracion_datos_nivel_sin_falla import df_tanque
import seaborn as sbn
from sklearn.metrics import confusion_matrix

np.random.seed(0)

'''Nivel del tanque con falla de deriva'''

# Predefinimos el DataFrame al que iremos anexando los valores con falla para distintas condiciones
df_tanque_falla = pd.DataFrame(columns=['tiempo', 'nivel', 'falla', 'falla_pred'])
fig_detect_deriva = plt.figure(figsize=(10, 3))
true_positives = np.zeros([1, 3])
false_positives = np.zeros([1, 3])
pendiente = np.array([])

# Iteracion para cada valor de resistencia hidraulica que queremos probar
for r, i in zip(datos.delta_r, range(0, 3)):
    r_falla, pendiente_deriva, r_falla_bool = f.fault_generator(datos.r_sim).drift(start=datos.t_i_falla_deriva,
                                                                                   stop=datos.t_f_falla_deriva[-1],
                                                                                   step=datos.paso, change=r)
    pendiente = np.append(pendiente,[pendiente_deriva])
    nivel_falla_drift = simultank(area=datos.area, nivel_inicial=datos.nivel_inicial, resist_hidraulica=r_falla,
                                  caudal_entrada=datos.q_sim, tiempo_inicial=datos.t_i, tiempo_final=datos.t_f,
                                  paso=datos.paso, analitic_sol=False) + np.random.normal(0, 0.02, len(r_falla))

    # Consideramos unicamente los datos de tiempo mayor al tiempo de establecimiento al 2%
    nivel_falla_drift_ee = nivel_falla_drift[datos.tss_2/datos.paso:]
    r_falla_bool_ee = r_falla_bool[datos.tss_2/datos.paso:]

    # Creamos un DataFrame con los resultados
    df_tanque_falla_r = pd.DataFrame({'tiempo': datos.t_sim_ee, 'nivel': nivel_falla_drift_ee,
                                      'falla': r_falla_bool_ee}, columns=['tiempo', 'nivel', 'falla'])

    # Detectamos la falla
    vector_detected, falla_bool, fault_counter, N = f.fault_detector(df_tanque_falla_r.nivel). \
        t_test(non_faulty_data=df_tanque.nivel, stand_dev=df_tanque_falla_r.nivel.std(),
               conf_lev=datos.conf_lev, delta_mean=0.05, N='auto')

    # Agregamos la columna de prediccion de falla
    df_tanque_falla_r['falla_pred'] = falla_bool

    # Anexamos los datos de falla nuevos al DataFrame general
    df_tanque_falla = df_tanque_falla.append(df_tanque_falla_r, ignore_index=True)

    # Anadimos la grafica de la deteccion de fallas a fig
    ax = fig_detect_deriva.add_subplot(1, 3, i+1)
    ax.plot(datos.t_sim, nivel_falla_drift)
    ax.scatter(datos.t_sim_ee[falla_bool == 1], vector_detected, c='r', marker='o',
               alpha=0.2, label='Falla detectada', s=12)

    # Creamos los vectores de FDR y FAR para ser usados en el heatmap
    tn, fp, fn, tp = confusion_matrix(y_true=r_falla_bool_ee, y_pred=falla_bool).ravel()
    true_positives[0, i] = tp/(tp+fn)
    false_positives[0, i] = fp/(fp+tn)

# Generamos un Heatmap para visualizar el FDR y FAR
fig, (ax1, ax2, ax_cbar) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]}, figsize=(12,4))
ax1.get_shared_y_axes().join(ax2)
g1 = sbn.heatmap(true_positives, annot=True, xticklabels=pendiente,
                 yticklabels=datos.delta_r, ax=ax1, vmin=0, vmax=1, cbar=False)
g2 = sbn.heatmap(false_positives, annot=True, xticklabels=pendiente,
                 yticklabels=datos.delta_r, ax=ax2, vmin=0, vmax=1, cbar_ax=ax_cbar)

# Mostramos las graficas
plt.show()
