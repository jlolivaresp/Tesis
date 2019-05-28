import matplotlib.pyplot as plt
import numpy as np
import t_test_F_test
#import residuos_t_test_F_test
import pandas as pd
import t_test_F_test_ML
#import t_test_F_test_ML_residuos

ttest_FDR_avg = t_test_F_test.FDR_FAR_fallas_tanque.ttest_FDR.mean()
ttest_FAR_avg = t_test_F_test.FDR_FAR_fallas_tanque.ttest_FAR.mean()
ftest_FDR_avg = t_test_F_test.FDR_FAR_fallas_tanque.ftest_FDR.mean()
ftest_FAR_avg = t_test_F_test.FDR_FAR_fallas_tanque.ftest_FAR.mean()
ttest_FDR_complemento = 1-ttest_FDR_avg
ttest_FAR_complemento = 1-ttest_FAR_avg
ftest_FDR_complemento = 1-ftest_FDR_avg
ftest_FAR_complemento = 1-ftest_FAR_avg

fig, ax = plt.subplots()

FDR_FAR = [ttest_FDR_avg, ttest_FDR_complemento, ttest_FAR_avg, ttest_FAR_complemento,
           ftest_FDR_avg, ftest_FDR_complemento, ftest_FAR_avg, ftest_FAR_complemento]

size = 0.3
vals = np.array(FDR_FAR)
labels = ['t-test: FDR', '', 't-test: FAR', '', 'F-test: FDR', '', 'F-test: FAR', '']

_, tx, autotexts = ax.pie(vals[0:4], radius=1, colors=['#108282', '#cfcac8', '#ad190f', '#cfcac8'], labels=labels[0:4],
                          wedgeprops=dict(width=size, edgecolor='w'), autopct='', pctdistance=0.85)
for i, a in enumerate(autotexts):
    a.set_text("{:.2f}".format(FDR_FAR[0:4][i]))
    if i==1 or i==3:
        a.set_text('')
    tx[i].set_fontsize(8)

_, tx, autotexts = ax.pie(vals[4:8], radius=1-size, colors=['#5becff', '#f2ece9', '#ed6145', '#f2ece9'], labels=labels[4:8],
                          wedgeprops=dict(width=size, edgecolor='w'), autopct='', pctdistance=0.80, labeldistance=0.3)
for i, a in enumerate(autotexts):
    a.set_text("{:.2f}".format(FDR_FAR[4:8][i]))
    if i==1 or i==3:
        a.set_text('')
    tx[i].set_fontsize(8)
ax.set(aspect="equal", title='Promedios de FDR y FAR Globales (Nivel)')

'''Para el vector de residuos'''
'''
ttest_FDR_avg_resid = residuos_t_test_F_test.FDR_FAR_fallas_tanque_residuos.ttest_FDR.mean()
ttest_FAR_avg_resid = residuos_t_test_F_test.FDR_FAR_fallas_tanque_residuos.ttest_FAR.mean()
ftest_FDR_avg_resid = residuos_t_test_F_test.FDR_FAR_fallas_tanque_residuos.ftest_FDR.mean()
ftest_FAR_avg_resid = residuos_t_test_F_test.FDR_FAR_fallas_tanque_residuos.ftest_FAR.mean()

ttest_FDR_resid_complemento = 1-ttest_FDR_avg_resid
ttest_FAR_resid_complemento = 1-ttest_FAR_avg_resid
ftest_FDR_resid_complemento = 1-ftest_FDR_avg_resid
ftest_FAR_resid_complemento = 1-ftest_FAR_avg_resid

fig, ax = plt.subplots()

FDR_FAR_resid = [ttest_FDR_avg_resid, ttest_FDR_resid_complemento, ttest_FAR_avg_resid, ttest_FAR_resid_complemento,
                 ftest_FDR_avg_resid, ftest_FDR_resid_complemento, ftest_FAR_avg_resid, ftest_FAR_resid_complemento]
FDR_FAR_resid *= 1
size = 0.3
vals = np.array(FDR_FAR_resid)
labels = ['t-test: FDR', '', 't-test: FAR', '', 'F-test: FDR', '', 'F-test: FAR', '']

_, tx, autotexts = ax.pie(vals[0:4], radius=1, colors=['#108282', '#cfcac8', '#ad190f', '#cfcac8'], labels=labels[0:4],
                         wedgeprops=dict(width=size, edgecolor='w'), autopct='', pctdistance=0.85)
for i, a in enumerate(autotexts):
    a.set_text("{:.2f}".format(FDR_FAR_resid[0:4][i]))
    if i==1 or i==3:
        a.set_text('')
    tx[i].set_fontsize(8)

_, tx, autotexts = ax.pie(vals[4:8], radius=1-size, colors=['#5becff', '#f2ece9', '#ed6145', '#f2ece9'], labels=labels[4:8],
                         wedgeprops=dict(width=size, edgecolor='w'), autopct='', pctdistance=0.80, labeldistance=0.05)
for i, a in enumerate(autotexts):
    a.set_text("{:.2f}".format(FDR_FAR_resid[4:8][i]))
    if i==1 or i==3:
        a.set_text('')
    tx[i].set_fontsize(8)
ax.set(aspect="equal", title='Promedios de FDR y FAR Globales (Residuos)')
'''
plt.show()

ttest_FDR_avg_ML = t_test_F_test_ML.FDR_0
ttest_FAR_avg_ML = t_test_F_test_ML.FAR_0
ttest_FDR_avg_ML_residuos = t_test_F_test_ML_residuos.FDR_0
ttest_FAR_avg_ML_residuos = t_test_F_test_ML_residuos.FAR_0

avg = [[ttest_FDR_avg, ttest_FAR_avg, ftest_FDR_avg, ftest_FAR_avg, ttest_FDR_avg_ML, ttest_FAR_avg_ML],
       [ttest_FDR_avg_resid, ttest_FAR_avg_resid,
        ftest_FDR_avg_resid, ftest_FAR_avg_resid,
        ttest_FDR_avg_ML_residuos, ttest_FAR_avg_ML_residuos],
       ['t-test', 't-test', 'F-test', 'F-test', 'kNN', 'kNN'],
       ['FDR', 'FAR', 'FDR', 'FAR', 'FDR', 'FAR']]
df_resultados_globales = pd.DataFrame(data={'Nivel': avg[0], 'Residuos': avg[1], 'Metodo':avg[2], 'Metric': avg[3]})
df_resultados_globales.set_index(['Metodo', 'Metric'], inplace=True)
print(df_resultados_globales)
df_resultados_globales.to_excel('C:/Users/User/Documents/USB/Control de Procesos/Tesis/Libro, '
                                'Bitácora e Imágenes/resultados_globales.xlsx')