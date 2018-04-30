""" Este script identifica los parametros de un modelo ARMAX de orden (1,0) con variable exogena para el nivel del
tanque en funcion del caudal Q"""

import numpy as np
import matplotlib.pyplot as plt
from simulador_nivel_tanque import simultank
import datos_tanque
from statsmodels.tsa.arima_model import ARIMA
import fault_2 as f
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot

np.random.seed(0)

# Generamos un vector caudal como senal cuadrada con media mean, desviacion estandar std y periodo T


def senal_cuadrada_normal_dist(size, mean, std, T):
    periods_number = int(np.ceil(size/T))
    normal_periods = np.random.normal(mean, std, periods_number)
    senal_cuadrada = [np.ones(T)*i for i in normal_periods]
    return np.concatenate(senal_cuadrada).ravel()[0:size]

q_modelaje = senal_cuadrada_normal_dist(len(datos_tanque.t_sim), 0, 1, 5)

# Simulamos el nivel del tanque para este caudal

nivel_modelaje = simultank(datos_tanque.area, datos_tanque.nivel_inicial, datos_tanque.r_sim, q_modelaje,
                           datos_tanque.t_i, datos_tanque.t_f, datos_tanque.paso)[datos_tanque.tss_2:]

# Normalizamos los datos

norm_q_modelaje = q_modelaje/np.linalg.norm(q_modelaje)
norm_nivel_modelaje = nivel_modelaje/np.linalg.norm(nivel_modelaje)

# Aplicamos el modelo ARMAX y obtenemos los parametros

model = ARIMA(endog=nivel_modelaje, order=(1, 0, 0), exog=q_modelaje[datos_tanque.tss_2:])
model_fit = model.fit(trend='nc', disp=False)

parametros = model_fit.params

# Probamos el modelo

# Falla Deriva

q_test = np.copy(datos_tanque.q_sim) + np.random.normal(0, 0.01, len(datos_tanque.q_sim))
nivel_test = simultank(datos_tanque.area, datos_tanque.nivel_inicial, datos_tanque.r_sim, q_test,
                       datos_tanque.t_i, datos_tanque.t_f, datos_tanque.paso)
nivel_falla, _, _ = f.fault_generator(nivel_test).drift(start=datos_tanque.t_i_falla_deriva,
                                                        stop=datos_tanque.t_f_falla_deriva[2], step=datos_tanque.paso,
                                                        change=datos_tanque.delta_h[2])
q_test = q_test[datos_tanque.tss_2:]
nivel_test = nivel_test[datos_tanque.tss_2:]
nivel_pred = np.zeros(len(q_test))
nivel_falla = nivel_falla[datos_tanque.tss_2:]

for i in range(1, len(q_test)):
    nivel_pred[i] = parametros[1]*nivel_falla[i-1] + parametros[0]*q_test[i-1]

resid = nivel_pred[1:] - nivel_falla[1:]
plt.figure()
plt.plot(resid)

plt.figure()
plt.plot(nivel_falla[1:], c='b', label='NIvel con Falla')
plt.plot(nivel_pred[1:], c='g', label='Nivel ARMAX')
plt.plot(nivel_test[1:], c='k', label='Nivel Teorico')
plt.legend()
#plt.show()

# Falla Pulso

q_test = np.copy(datos_tanque.q_sim) + np.random.normal(0, 0.01, len(datos_tanque.q_sim))
nivel_test = simultank(datos_tanque.area, datos_tanque.nivel_inicial, datos_tanque.r_sim, q_test,
                       datos_tanque.t_i, datos_tanque.t_f, datos_tanque.paso)
nivel_falla, _, _ = f.fault_generator(nivel_test).random_pulse(start=datos_tanque.t_i_falla_pulso,
                                                               stop=datos_tanque.t_f_falla_pulso,
                                                               step=datos_tanque.paso, N=datos_tanque.N_pulsos[2],
                                                               amplitude=datos_tanque.amplitud_pulso[2],
                                                               random_seed=datos_tanque.random_seed,
                                                               mode=datos_tanque.modo)
q_test = q_test[datos_tanque.tss_2:]
nivel_test = nivel_test[datos_tanque.tss_2:]
nivel_pred = np.zeros(len(q_test))
nivel_falla = nivel_falla[datos_tanque.tss_2:]

for i in range(1, len(q_test)):
    nivel_pred[i] = parametros[1]*nivel_falla[i-1] + parametros[0]*q_test[i-1]

resid = nivel_pred[1:] - nivel_falla[1:]
plt.figure()
plt.plot(resid)

plt.figure()
plt.plot(nivel_falla[1:], c='b', label='NIvel con Falla')
plt.plot(nivel_pred[1:], c='g', label='Nivel ARMAX')
plt.plot(nivel_test[1:], c='k', label='Nivel Teorico')
plt.legend()
plt.show()