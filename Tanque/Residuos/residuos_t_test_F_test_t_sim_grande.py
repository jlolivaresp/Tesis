import numpy as np
import matplotlib.pyplot as plt
from simulador_nivel_tanque import simultank
import datos_tanque
from statsmodels.tsa.arima_model import ARIMA
import fault_2 as f
import fallas_tanque_extend
import pandas as pd
from sklearn.metrics import confusion_matrix
import scipy.stats
import scipy as sp

# Generamos un vector caudal como senal cuadrada con media mean, desviacion estandar std y periodo T


def senal_cuadrada_normal_dist(size, mean, std, T):
    periods_number = int(np.ceil(size/T))
    normal_periods = np.random.normal(mean, std, periods_number)
    senal_cuadrada = [np.ones(T)*i for i in normal_periods]
    return np.concatenate(senal_cuadrada).ravel()[0:size]

q_modelaje = senal_cuadrada_normal_dist(len(datos_tanque.t_sim_extend), 0, 1, 5)

# Simulamos el nivel del tanque para este caudal

nivel_modelaje = simultank(datos_tanque.area, datos_tanque.nivel_inicial, datos_tanque.r_sim_extend, q_modelaje,
                           datos_tanque.t_i, datos_tanque.t_f_extend, datos_tanque.paso)[datos_tanque.tss_2:]

# Normalizamos los datos

norm_q_modelaje = q_modelaje/np.linalg.norm(q_modelaje)
norm_nivel_modelaje = nivel_modelaje/np.linalg.norm(nivel_modelaje)

# Aplicamos el modelo ARMAX y obtenemos los parametros

model = ARIMA(endog=nivel_modelaje, order=(1, 0, 0), exog=q_modelaje[datos_tanque.tss_2:])
model_fit = model.fit(trend='nc', disp=False)

parametros = model_fit.params

# print(model_fit.summary())

# Construimos el DataFrame al que anexaremos las columnas de residuos y de FDR y FAR

grupos = fallas_tanque_extend.df_tanque_falla.groupby(['tipo_falla', 'caracteristica_1', 'caracteristica_2'])

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

df_tanque_falla_residuos = fallas_tanque_extend.df_tanque_falla
df_tanque_falla_residuos['nivel_armax'] = nivel_pred_todos
df_tanque_falla_residuos['residuos'] = df_tanque_falla_residuos.nivel - df_tanque_falla_residuos.nivel_armax
df_tanque_falla_residuos['nivel_armax_sin_falla'] = nivel_pred_sin_falla_todos
df_tanque_falla_residuos['residuos_sin_falla'] = df_tanque_falla_residuos.nivel_sin_falla - \
                                                 df_tanque_falla_residuos.nivel_armax_sin_falla

# Detectamos las fallas con t-test y F-test

groups = df_tanque_falla_residuos.groupby(['tipo_falla', 'caracteristica_1', 'caracteristica_2'])
tp_mat = np.array([])
tn_mat = np.array([])
fp_mat = np.array([])
fn_mat = np.array([])
delta = np.array([])
N_vector_ttest = np.array([])

for j in datos_tanque.detect_delta_media_residuos_extend:
    for group in groups:
        _, t_test_pred, _, N = f.fault_detector(group[1].residuos[1:]).t_test(group[1].residuos_sin_falla,
                                                                              group[1].residuos_sin_falla.std(), 0.95, j)
        a = [1 if i is True else 0 for i in group[1].condicion_falla[1:].values]

        tn, fp, fn, tp = confusion_matrix(y_true=a, y_pred=t_test_pred).ravel()
        tp_mat = np.append(tp_mat, tp)
        tn_mat = np.append(tn_mat, tn)
        fp_mat = np.append(fp_mat, fp)
        fn_mat = np.append(fn_mat, fn)
        delta = np.append(delta, j)
        N_vector_ttest = np.append(N_vector_ttest, N)
FDR_ttest = tp_mat/(tp_mat+fn_mat)
FAR_ttest = fp_mat/(fp_mat+tn_mat)

tp_mat = np.array([])
tn_mat = np.array([])
fp_mat = np.array([])
fn_mat = np.array([])
intensidad_falla = np.array([])
tipo_falla = np.array([])
N_vector_ftest = np.array([])

for j in datos_tanque.detect_delta_var_residuos:
    for group in groups:
        _, F_test_pred, _, N = f.fault_detector(group[1].residuos[1:]).f_test(group[1].residuos_sin_falla,
                                                                              group[1].residuos_sin_falla.std(), j, 0.95)
        a = [1 if i is True else 0 for i in group[1].condicion_falla[1:].values]

        tn, fp, fn, tp = confusion_matrix(y_true=a, y_pred=F_test_pred).ravel()
        tp_mat = np.append(tp_mat, tp)
        tn_mat = np.append(tn_mat, tn)
        fp_mat = np.append(fp_mat, fp)
        fn_mat = np.append(fn_mat, fn)

        intensidad_falla = np.append(intensidad_falla, group[1].intensidad_falla.unique()[0])
        tipo_falla = np.append(tipo_falla, group[1].tipo_falla.unique()[0])
        N_vector_ftest = np.append(N_vector_ftest, N)
FDR_Ftest = tp_mat/(tp_mat+fn_mat)
FAR_Ftest = fp_mat/(fp_mat+tn_mat)

FDR_FAR_fallas_tanque_residuos = pd.DataFrame(data={'ttest_FDR': FDR_ttest, 'ttest_FAR': FAR_ttest,
                                                    'ftest_FDR': FDR_Ftest, 'ftest_FAR': FAR_Ftest,
                                                    'tipo_falla': tipo_falla, 'delta': delta,
                                                    'intensidad_falla': intensidad_falla, 'N_ttest': N_vector_ttest,
                                                    'N_ftest': N_vector_ftest},
                                              columns=['tipo_falla', 'delta', 'ttest_FDR', 'ttest_FAR',
                                                       'ftest_FDR', 'ftest_FAR', 'intensidad_falla', 'N_ttest',
                                                       'N_ftest'])

FDR_FAR_fallas_tanque_residuos.set_index(['delta', 'tipo_falla'], inplace=True)