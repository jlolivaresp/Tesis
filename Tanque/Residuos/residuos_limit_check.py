import numpy as np
import matplotlib.pyplot as plt
from simulador_nivel_tanque import simultank
import datos_tanque
from statsmodels.tsa.arima_model import ARIMA
import fault_2 as f
import fallas_tanque
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

# print(model_fit.summary())

# Construimos el DataFrame al que anexaremos las columnas de residuos y de FDR y FAR

grupos = fallas_tanque.df_tanque_falla.groupby(['tipo_falla', 'caracteristica_1', 'caracteristica_2'])

nivel_pred_todos = np.zeros([])
nivel_pred_sin_falla_todos = np.zeros([])

for grupo in grupos:
    nivel_pred = np.zeros(len(grupo[1]))
    nivel_pred_sin_falla = np.zeros(len(grupo[1]))
    for i in range(1, len(nivel_pred)):
        nivel_pred[i] = parametros[1]*grupo[1].nivel.values[i-1] + parametros[0]*grupo[1].nivel.values[i-1]
        nivel_pred_sin_falla[i] = parametros[1]*grupo[1].nivel_sin_falla.values[i-1] + parametros[0]*grupo[1].nivel_sin_falla.values[i-1]
    nivel_pred_todos = np.append(nivel_pred_todos, nivel_pred)
    nivel_pred_sin_falla_todos = np.append(nivel_pred_sin_falla_todos, nivel_pred_sin_falla)
nivel_pred_todos = np.delete(nivel_pred_todos, 0)
nivel_pred_sin_falla_todos = np.delete(nivel_pred_sin_falla_todos, 0)

df_tanque_falla_residuos = fallas_tanque.df_tanque_falla
df_tanque_falla_residuos['nivel_armax'] = nivel_pred_todos
df_tanque_falla_residuos['residuos'] = df_tanque_falla_residuos.nivel - df_tanque_falla_residuos.nivel_armax
df_tanque_falla_residuos['nivel_armax_sin_falla'] = nivel_pred_sin_falla_todos
df_tanque_falla_residuos['residuos_sin_falla'] = df_tanque_falla_residuos.nivel_sin_falla - \
                                                 df_tanque_falla_residuos.nivel_armax_sin_falla

df_tanque_falla_residuos = df_tanque_falla_residuos.groupby(['tipo_falla', 'caracteristica_1', 'caracteristica_2']).\
    apply(lambda group: group.iloc[1:])
df_tanque_falla_residuos.drop(['tipo_falla', 'caracteristica_1', 'caracteristica_2'], axis=1, inplace=True)
df_tanque_falla_residuos.reset_index(inplace=True)
print(df_tanque_falla_residuos)
df_tanque_falla_residuos.nivel.plot()
df_tanque_falla_residuos.nivel_armax_sin_falla.plot()
df_tanque_falla_residuos.residuos.plot()
plt.show()