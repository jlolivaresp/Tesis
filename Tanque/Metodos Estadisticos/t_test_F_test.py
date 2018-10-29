"""Este programa realiza las pruebas de t_test y F_test sobre los datos falla_tanque"""

import fallas_tanque
import fault_2 as f
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import datos_tanque

'''
A = np.array([2.00, 2.02, 1.99, 2.01, 1.96, 2.00, 2.07, 2.08, 2.01, 1.98, 2.00,
     2.02, 1.99, 2.01, 1.96, 2.00, 2.07, 2.08, 2.01, 1.98])
B = np.array([2.10, 2.15, 2.08, 1.98, 1.95, 1.95, 2.01, 2.07, 2.03, 1.99, 2.10,
     2.15, 2.08, 1.98, 1.95, 1.95, 2.01, 2.07, 2.03, 1.99])

_, t_test_pred, _, N = f.fault_detector(B).t_test(A, np.std(A), 0.95, 0.02)
print(t_test_pred)
print('end')
'''

print(fallas_tanque.df_tanque_falla.describe())
groups = fallas_tanque.df_tanque_falla.groupby(['tipo_falla', 'caracteristica_1', 'caracteristica_2'])
tp_mat = np.array([])
tn_mat = np.array([])
fp_mat = np.array([])
fn_mat = np.array([])
delta = np.array([])
N_vector_ttest = np.array([])

for j in datos_tanque.detect_delta_media:
    print(j)
    for group in groups:
        _, t_test_pred, _, N = f.fault_detector(group[1].nivel).t_test(group[1].nivel_sin_falla,
                                                                       group[1].nivel_sin_falla.std(), 0.95, j)
        a = [1 if i is True else 0 for i in group[1].condicion_falla.values]

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

for j in datos_tanque.detect_delta_var:
    print(j)
    for group in groups:
        _, F_test_pred, _, N = f.fault_detector(group[1].nivel).f_test(group[1].nivel_sin_falla,
                                                                       group[1].nivel_sin_falla.std(), j, 0.95)
        a = [1 if i is True else 0 for i in group[1].condicion_falla.values]

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

FDR_FAR_fallas_tanque = pd.DataFrame(data={'ttest_FDR': FDR_ttest, 'ttest_FAR': FAR_ttest,
                                           'ftest_FDR': FDR_Ftest, 'ftest_FAR': FAR_Ftest,
                                           'tipo_falla': tipo_falla, 'delta': delta,
                                           'intensidad_falla': intensidad_falla, 'N_ttest': N_vector_ttest,
                                           'N_ftest': N_vector_ftest},
                                     columns=['tipo_falla', 'delta', 'ttest_FDR', 'ttest_FAR',
                                              'ftest_FDR', 'ftest_FAR', 'intensidad_falla', 'N_ttest', 'N_ftest'])

FDR_FAR_fallas_tanque.set_index(['delta', 'tipo_falla'], inplace=True)
print(FDR_FAR_fallas_tanque.describe())
