import t_test_F_test
import matplotlib.pyplot as plt
import numpy as np
import datos_tanque
import pandas as pd
import fallas_tanque
import fault_2 as f
import matplotlib.gridspec as gridspec

a = t_test_F_test.FDR_FAR_fallas_tanque.groupby(level=['tipo_falla']).ttest_FDR.apply(lambda x: np.average(x))
b = t_test_F_test.FDR_FAR_fallas_tanque.groupby(level=['tipo_falla']).ttest_FAR.apply(lambda x: np.average(x))
c = t_test_F_test.FDR_FAR_fallas_tanque.groupby(level=['tipo_falla']).ftest_FDR.apply(lambda x: np.average(x))
d = t_test_F_test.FDR_FAR_fallas_tanque.groupby(level=['tipo_falla']).ftest_FAR.apply(lambda x: np.average(x))


pos = [2, 1, 0]
width = 0.25
fig, ax = plt.subplots(ncols=2, sharey=True)
ax[0].set_ylim([-0.6, 2.5])
ax[1].set_ylim([-0.6, 2.5])

ax[0].barh(pos, a, width, color='#108282', alpha=0.7, label='t-test')
ax[1].barh(pos, b, width, color='#ad190f', alpha=0.7, label='t-test')
ax[0].barh([p + width + 0.01 for p in pos], c, width, color='#5becff', alpha=0.7, label='F-test')
ax[1].barh([p + width + 0.01 for p in pos], d, width, color='#ed6145', alpha=0.7, label='F-test')

legends_FDR = ax[1].legend(frameon=False, loc=8)
legends_FAR = ax[0].legend(frameon=False, loc=8)

for l in legends_FAR.get_texts():
    l.set_color('k')
    l.set_alpha(0.7)

ax[0].set_title('FDR', color='k',alpha=0.7)
ax[1].set_title('FAR', color='k',alpha=0.7)

ax[0].invert_xaxis()
fig.subplots_adjust(wspace=0)

for i, v in enumerate(a[::-1]):
    ax[0].text(v + 0.20, i, '{:.2f}'.format(v), color='#108282', alpha=0.7, fontweight='bold')
for i, v in enumerate(b[::-1]):
    ax[1].text(v + 0.01, i, '{:.2f}'.format(v), color='#ad190f', alpha=0.7, fontweight='bold')
for i, v in enumerate(c[::-1]):
    ax[0].text(v + 0.20, i + width, '{:.2f}'.format(v), color='#5becff', alpha=0.7, fontweight='bold')
for i, v in enumerate(d[::-1]):
    ax[1].text(v + 0.01, i + width, '{:.2f}'.format(v), color='#ed6145', alpha=0.7, fontweight='bold')

ax[0].set_xlim([1.3, 0])
ax[1].set_xlim([0, 1.3])

ax[0].text(1.3, 2 + width/2, 'A', color='k', alpha=0.9)
ax[0].text(1.3, 1 + width/2, 'B', color='k', alpha=0.9)
ax[0].text(1.3, 0 + width/2, 'C', color='k', alpha=0.9)

ax[0].set(yticks=[i + width/2 for i in pos], yticklabels=['D', 'P', 'V'])
ax[0].yaxis.tick_right()

for i, a in enumerate(ax):
    a.spines["top"].set_visible(False)
    a.spines["bottom"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[0].spines["left"].set_visible(False)
ax[0].spines['right'].set_color('k')
ax[0].spines['right'].set_alpha(0.7)
ax[1].spines['left'].set_color('k')
ax[1].spines['right'].set_alpha(0.7)

ax[0].get_xaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)

ax[0].tick_params(axis='y', right='off')
ax[1].tick_params(axis='y', right='off')

fig.subplots_adjust(top=0.83,bottom=0.1)
fig.suptitle('Promedios de FDR y FAR por Método y Tipo de Falla',size=12, color='k',fontweight='bold',alpha=0.7)
plt.figtext(0.35, 0.05, 'A: Deriva, B: Pulso, C: Varianza', color='k',alpha=0.7)

'''______________________________________________Segunda grafica_____________________________________________________'''

FDR_FAR_fallas_tanque_copy = t_test_F_test.FDR_FAR_fallas_tanque.copy()

FDR_FAR_fallas_tanque_copy.sort_values(by=['intensidad_falla'], ascending=True, inplace=True)

#numero_intensidad_0 = np.arange(1, 13, 1)
numero_intensidad = [i for i in range(1, 13)]*len(datos_tanque.detect_delta_media)*3
#numero_intensidad = np.append(numero_intensidad_0, numero_intensidad_0)
#numero_intensidad = np.append(numero_intensidad, numero_intensidad_0)

FDR_FAR_fallas_tanque_copy['numero_intensidad'] = numero_intensidad
groups = FDR_FAR_fallas_tanque_copy.groupby(['numero_intensidad'])

FDR_FAR_fallas_tanque_por_intensidad_falla = groups.agg('mean')

fdr_ttest = FDR_FAR_fallas_tanque_por_intensidad_falla.ttest_FDR
far_ttest = FDR_FAR_fallas_tanque_por_intensidad_falla.ttest_FAR
fdr_ftest = FDR_FAR_fallas_tanque_por_intensidad_falla.ftest_FDR
far_ftest = FDR_FAR_fallas_tanque_por_intensidad_falla.ftest_FAR

pos = np.arange(1, 13, 1)
width = 0.3
fig_1, ax_1 = plt.subplots(nrows=1, sharex=True)

ax_1 = plt.bar(pos, fdr_ttest, width, color='#108282', alpha=0.7, label='FDR: t-test')
ax_2 = plt.bar(pos, np.negative(far_ttest), width, color='#ad190f', alpha=0.7, label='FAR: t-test')
ax_3 = plt.bar(pos + width, fdr_ftest, width, color='#5becff', alpha=0.7, label='FDR: F-test')
ax_4 = plt.bar(pos + width, np.negative(far_ftest), width, color='#ed6145', alpha=0.7, label='FAR: F-test')

axes = plt.gca()

for i, v in enumerate(fdr_ttest[::1]):
    axes.text(i + 0.9, v + 0.14, '{:.2f}'.format(v), color='#108282', alpha=0.7, fontweight='bold', fontsize=6, rotation='vertical')
for i, v in enumerate(far_ttest[::1]):
    axes.text(i + 0.9, -v - 0.06, '{:.2f}'.format(v), color='#ad190f', alpha=0.7, fontweight='bold', fontsize=6, rotation='vertical')
for i, v in enumerate(fdr_ftest[::1]):
    axes.text(i + 0.9 + width, v + 0.14, '{:.2f}'.format(v), color='#5becff', alpha=0.7, fontweight='bold', fontsize=6, rotation='vertical')
for i, v in enumerate(far_ftest[::1]):
    axes.text(i + 0.9 + width, -v - 0.06, '{:.2f}'.format(v), color='#ed6145', alpha=0.7, fontweight='bold', fontsize=6, rotation='vertical')

axes.set_ylim([-1.5, 1.6])


axes.spines["top"].set_visible(False)
axes.spines["bottom"].set_visible(False)
axes.spines["right"].set_visible(False)
axes.spines["left"].set_visible(False)
axes.tick_params(axis='both', left='off', top='off', bottom='off')
axes.set_yticks([0,0])
axes.axes.get_yaxis().set_visible(False)
[i.set_alpha(0.6) for i in axes.get_xticklabels()]
[i.set_fontsize(8) for i in axes.get_xticklabels()]

axes.set_title('Promedios de FDR y FAR Por Intensidad de Falla\n(Mínima Intensidad = 1 - Máxima Intensidad = 12)',
               alpha=0.6, fontsize=10)
axes.set_xlabel('Intensidad', alpha=0.6, fontsize=8)

legend_FDR = axes.legend(handles=[ax_1, ax_3], frameon=False, loc=2)
plt.gca().add_artist(legend_FDR)

legend_FAR = axes.legend(handles=[ax_2, ax_4], frameon=False, loc=3)

for l in legend_FDR.get_texts():
    l.set_fontsize(8)
    l.set_alpha(0.6)
for l in legend_FAR.get_texts():
    l.set_fontsize(8)
    l.set_alpha(0.6)

'''__________________________________________________Tercera grafica_________________________________________________'''
# Grafica de Prueba de deteccion mediante t-test y F-test

fig = plt.figure(figsize=(7, 6.5))

gs_0 = gridspec.GridSpec(3, 1)

gs_01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_0[0])

ax_1 = plt.Subplot(fig, gs_01[:-1, :])
fig.add_subplot(ax_1)
ax_2 = plt.Subplot(fig, gs_01[-1, :])
fig.add_subplot(ax_2, sharey=ax_1)

gs_02 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_0[1])

ax_3 = plt.Subplot(fig, gs_02[:-1, :])
fig.add_subplot(ax_3, sharey=ax_1)
ax_4 = plt.Subplot(fig, gs_02[-1, :])
fig.add_subplot(ax_4, sharey=ax_1)

gs_03 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_0[2])

ax_5 = plt.Subplot(fig, gs_03[:-1, :])
fig.add_subplot(ax_5, sharey=ax_1)
ax_6 = plt.Subplot(fig, gs_03[-1, :])
fig.add_subplot(ax_6, sharey=ax_1)


ej_deriva_pred = fallas_tanque.df_tanque_falla[(fallas_tanque.df_tanque_falla.tipo_falla == 'deriva')
                                               & (fallas_tanque.df_tanque_falla.caracteristica_1 == 0.02)
                                               & (fallas_tanque.df_tanque_falla.caracteristica_2 == 16.1)]
ej_pulso_pred = fallas_tanque.df_tanque_falla[(fallas_tanque.df_tanque_falla.tipo_falla == 'pulso')
                                              & (fallas_tanque.df_tanque_falla.caracteristica_1 == 0.025)
                                              & (fallas_tanque.df_tanque_falla.caracteristica_2 == 80)]
ej_varianza_pred = fallas_tanque.df_tanque_falla[(fallas_tanque.df_tanque_falla.tipo_falla == 'varianza')
                                                 & (fallas_tanque.df_tanque_falla.caracteristica_1 == 0.025)
                                                 & (fallas_tanque.df_tanque_falla.caracteristica_2 == 16.1)]


markersize = 6.5

delta_plot = 5

# t-test Deriva

_, t_test_pred_drift, _, _ = f.fault_detector(ej_deriva_pred.nivel).t_test(ej_deriva_pred.nivel_sin_falla,
                                                                           ej_deriva_pred.nivel_sin_falla.std(),
                                                                           0.95, datos_tanque.detect_delta_media[delta_plot])

ax_1.plot(ej_deriva_pred.tiempo, ej_deriva_pred.nivel, c='k',  alpha=0.7, linewidth=1)
ax_1.plot(ej_deriva_pred.tiempo, ej_deriva_pred.nivel_sin_falla, c='k',  alpha=0.4, linewidth=1)
ax_1.scatter(ej_deriva_pred.tiempo[(t_test_pred_drift == 1) & (ej_deriva_pred.condicion_falla != 0)],
              ej_deriva_pred.nivel[(t_test_pred_drift == 1) & (ej_deriva_pred.condicion_falla != 0)], c='#108282',
              alpha=0.7, s=markersize)

ax_1.scatter(ej_deriva_pred.tiempo[(ej_deriva_pred.condicion_falla == 0) & (t_test_pred_drift == 1)],
              ej_deriva_pred.nivel[(ej_deriva_pred.condicion_falla == 0) & (t_test_pred_drift == 1)], c='#ad190f',
              alpha=0.7, s=markersize)
ax_1.axhline(y=2, color='k', alpha=0.2, linestyle='--')
ax_1.annotate('2m', xy=(24, 2.002), color='k', alpha=0.4)
ax_1.annotate('t-test', xy=(3.5, 2.01), color='k', alpha=0.4, weight='bold')
ax_1.set_title('Deriva', color='#9e9a98', fontsize=10, weight='bold')

# t-test Pulso

_, t_test_pred_pulse, _, _ = f.fault_detector(ej_pulso_pred.nivel).t_test(ej_pulso_pred.nivel_sin_falla,
                                                                          ej_pulso_pred.nivel_sin_falla.std(),
                                                                          0.95, datos_tanque.detect_delta_media[delta_plot])
ax_3.plot(ej_pulso_pred.tiempo, ej_pulso_pred.nivel, c='k',  alpha=0.7, linewidth=1)
ax_3.plot(ej_pulso_pred.tiempo, ej_pulso_pred.nivel_sin_falla, c='k',  alpha=0.4, linewidth=1)
ax_3.scatter(ej_pulso_pred.tiempo[(t_test_pred_pulse == 1) & (ej_pulso_pred.condicion_falla != 0)],
              ej_pulso_pred.nivel[(t_test_pred_pulse == 1) & (ej_pulso_pred.condicion_falla != 0)], c='#108282',
              alpha=0.7, s=markersize)

ax_3.scatter(ej_pulso_pred.tiempo[(ej_pulso_pred.condicion_falla == 0) & (t_test_pred_pulse == 1)],
              ej_pulso_pred.nivel[(ej_pulso_pred.condicion_falla == 0) & (t_test_pred_pulse == 1)],
              c='#ad190f', alpha=0.7, s=markersize)
ax_3.axhline(y=2, color='k', alpha=0.2, linestyle='--')
ax_3.annotate('2m', xy=(24, 2.002), color='k', alpha=0.4)
ax_3.annotate('t-test', xy=(3.5, 2.01), color='k', alpha=0.4, weight='bold')
ax_3.set_title('Pulso', color='#9e9a98', fontsize=10, weight='bold')

# t-test Varianza
_, t_test_pred_var, _, _ = f.fault_detector(ej_varianza_pred.nivel).t_test(ej_varianza_pred.nivel_sin_falla,
                                                                           ej_varianza_pred.nivel_sin_falla.std(),
                                                                           0.95, datos_tanque.detect_delta_media[delta_plot])

ax_5.plot(ej_varianza_pred.tiempo, ej_varianza_pred.nivel, c='k',  alpha=0.7, label='Nivel con falla', linewidth=1)
ax_5.plot(ej_varianza_pred.tiempo, ej_varianza_pred.nivel_sin_falla, c='k',  alpha=0.4, label='Nivel sin falla', linewidth=1)
ax_5.scatter(ej_varianza_pred.tiempo[(t_test_pred_var == 1) & (ej_varianza_pred.condicion_falla != 0)],
              ej_varianza_pred.nivel[(t_test_pred_var == 1) & (ej_varianza_pred.condicion_falla != 0)], c='#108282',
              alpha=0.7, label='Verdaderos positivos', s=markersize)

ax_5.scatter(ej_varianza_pred.tiempo[(ej_varianza_pred.condicion_falla == 0) & (t_test_pred_var == 1)],
              ej_varianza_pred.nivel[(ej_varianza_pred.condicion_falla == 0) & (t_test_pred_var == 1)],
              c='#ad190f', alpha=0.7, label='Falsos positivos', s=markersize)
ax_5.axhline(y=2, color='k', alpha=0.2, linestyle='--')
ax_5.annotate('2m', xy=(24, 2.002), color='k', alpha=0.4)
ax_5.annotate('t-test', xy=(3.5, 2.01), color='k', alpha=0.4, weight='bold')
ax_5.set_title('Varianza', color='#9e9a98', fontsize=10, weight='bold')

# F-test Deriva

_, f_test_pred_drift, _, _ = f.fault_detector(ej_deriva_pred.nivel).f_test(ej_deriva_pred.nivel_sin_falla,
                                                                           ej_deriva_pred.nivel_sin_falla.std(),
                                                                           datos_tanque.detect_delta_var[delta_plot], 0.95)

ax_2.plot(ej_deriva_pred.tiempo, ej_deriva_pred.nivel, c='k',  alpha=0.7, linewidth=1)
ax_2.plot(ej_deriva_pred.tiempo, ej_deriva_pred.nivel_sin_falla, c='k',  alpha=0.4, linewidth=1)
ax_2.scatter(ej_deriva_pred.tiempo[(f_test_pred_drift == 1) & (ej_deriva_pred.condicion_falla != 0)],
             ej_deriva_pred.nivel[(f_test_pred_drift == 1) & (ej_deriva_pred.condicion_falla != 0)], c='#108282',
             alpha=0.7, s=markersize)

ax_2.scatter(ej_deriva_pred.tiempo[(ej_deriva_pred.condicion_falla == 0) & (f_test_pred_drift == 1)],
             ej_deriva_pred.nivel[(ej_deriva_pred.condicion_falla == 0) & (f_test_pred_drift == 1)], c='#ad190f',
             alpha=0.7, s=markersize)
ax_2.axhline(y=2, color='k', alpha=0.2, linestyle='--')
ax_2.annotate('2m', xy=(24, 2.002), color='k', alpha=0.4)
ax_2.annotate('F-test', xy=(3.5, 2.01), color='k', alpha=0.4, weight='bold')

# F-test Pulso

_, f_test_pred_pulse, _, _ = f.fault_detector(ej_pulso_pred.nivel).f_test(ej_pulso_pred.nivel_sin_falla,
                                                                          ej_pulso_pred.nivel_sin_falla.std(),
                                                                          datos_tanque.detect_delta_var[delta_plot], 0.95)

ax_4.plot(ej_pulso_pred.tiempo, ej_pulso_pred.nivel, c='k',  alpha=0.7, linewidth=1)
ax_4.plot(ej_pulso_pred.tiempo, ej_pulso_pred.nivel_sin_falla, c='k',  alpha=0.4, linewidth=1)
ax_4.scatter(ej_pulso_pred.tiempo[(f_test_pred_pulse == 1) & (ej_pulso_pred.condicion_falla != 0)],
             ej_pulso_pred.nivel[(f_test_pred_pulse == 1) & (ej_pulso_pred.condicion_falla != 0)], c='#108282',
             alpha=0.7, s=markersize)

ax_4.scatter(ej_pulso_pred.tiempo[(ej_pulso_pred.condicion_falla == 0) & (f_test_pred_pulse == 1)],
             ej_pulso_pred.nivel[(ej_pulso_pred.condicion_falla == 0) & (f_test_pred_pulse == 1)],
             c='#ad190f', alpha=0.7, s=markersize)
ax_4.axhline(y=2, color='k', alpha=0.2, linestyle='--')
ax_4.annotate('2m', xy=(24, 2.002), color='k', alpha=0.4)
ax_4.annotate('F-test', xy=(3.5, 2.01), color='k', alpha=0.4, weight='bold')


# F-test Varianza

_, f_test_pred_var, _, _ = f.fault_detector(ej_varianza_pred.nivel).f_test(ej_varianza_pred.nivel_sin_falla,
                                                                           ej_varianza_pred.nivel_sin_falla.std(),
                                                                           datos_tanque.detect_delta_var[delta_plot], 0.95)

ax_6.plot(ej_varianza_pred.tiempo, ej_varianza_pred.nivel, c='k',  alpha=0.7, label='Nivel con falla', linewidth=1)
l3, = ax_6.plot(ej_varianza_pred.tiempo, ej_varianza_pred.nivel_sin_falla, c='k',  alpha=0.4, linewidth=1, linestyle='-')
l1 = ax_6.scatter(ej_varianza_pred.tiempo[(f_test_pred_var == 1) & (ej_varianza_pred.condicion_falla != 0)],
             ej_varianza_pred.nivel[(f_test_pred_var == 1) & (ej_varianza_pred.condicion_falla != 0)], c='#108282',
             alpha=0.7, label='Verdaderos positivos', s=markersize)

l2 = ax_6.scatter(ej_varianza_pred.tiempo[(ej_varianza_pred.condicion_falla == 0) & (f_test_pred_var == 1)],
             ej_varianza_pred.nivel[(ej_varianza_pred.condicion_falla == 0) & (f_test_pred_var == 1)],
             c='#ad190f', alpha=0.7, label='Falsos positivos', s=markersize)
l4 = ax_6.axhline(y=2, color='k', alpha=0.2, linestyle='--')
ax_6.annotate('2m', xy=(24, 2.002), color='k', alpha=0.4)
ax_6.annotate('F-test', xy=(3.5, 2.01), color='k', alpha=0.4, weight='bold')

ax_1.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
ax_2.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
ax_3.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
ax_4.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
ax_5.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
ax_6.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on', labelcolor='#9e9a98')
ax_6.set_xlabel('Tiempo (h)', color='#9e9a98', fontsize=10)

for ax in [ax_1, ax_2, ax_3, ax_4, ax_5, ax_6]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xlim([3.3, 25])

legend_1 = fig.legend((l1, l2), ('Verdadero positivo', 'Falso positivo'), 'upper left', frameon=False)
legend_2 = fig.legend((l3, l4), ('Nivel sin falla', 'Set point'), 'upper right', frameon=False)

for l in legend_1.get_texts():
    l.set_fontsize(8)
    l.set_alpha(0.7)
for ll in legend_2.get_texts():
    ll.set_fontsize(8)
    ll.set_alpha(0.7)
plt.tight_layout(h_pad=2)

'''________________________________________________Cuarta Grafica____________________________________________________'''

sep_1 = 0.001
sep_2 = 0.01

fig, ax = plt.subplots(nrows=2, ncols=1)
ax_2 = ax[0].twiny()
ax_3 = ax[1].twiny()
fig.subplots_adjust(wspace=0.05)

groups = t_test_F_test.FDR_FAR_fallas_tanque.groupby(level=['delta'])
FDR_FAR_fallas_tanque_por_delta = groups.agg('mean')

ax[0].plot(datos_tanque.detect_delta_media*100/2, FDR_FAR_fallas_tanque_por_delta.ttest_FDR, marker='.', color='#108282', label='t-test: FDR')
ax[0].plot(datos_tanque.detect_delta_media*100/2, FDR_FAR_fallas_tanque_por_delta.ttest_FAR, marker='.', color='#ad190f', label='t-test: FAR')
ax[0].set_xlabel(r'$\Delta\mu (\%)$', color='k', alpha=0.8)

ax[1].plot(datos_tanque.detect_delta_var*100/np.var(fallas_tanque.nivel_ee), FDR_FAR_fallas_tanque_por_delta.ftest_FDR, marker='.', color='#5becff', label='F-test: FDR')
ax[1].plot(datos_tanque.detect_delta_var*100/np.var(fallas_tanque.nivel_ee), FDR_FAR_fallas_tanque_por_delta.ftest_FAR, marker='.', color='#ed6145', label='F-test: FAR')
ax[1].set_xlabel(r'$\Delta\sigma^2 (\%)$', color='k', alpha=0.8)

ax_2.set_xlabel('N', alpha=0.8, fontsize=6.5)
ax_3.set_xlabel('N', alpha=0.8, fontsize=6.5)

legends_1 = ax[1].legend(frameon=False, loc='upper right')
for l in legends_1.get_texts():
    l.set_alpha(0.8)
legends_2 = ax[0].legend(frameon=False, loc='lower left')
for l in legends_2.get_texts():
    l.set_alpha(0.8)

ax_2.set_xticks(np.arange(0, len(datos_tanque.detect_delta_media), 1))
ax_3.set_xticks(np.arange(0, len(datos_tanque.detect_delta_var), 1))

ax_2.set_xticklabels(['{}'.format(int(N)) for N in FDR_FAR_fallas_tanque_por_delta.N_ttest])
ax_3.set_xticklabels(['{}'.format(int(N)) for N in FDR_FAR_fallas_tanque_por_delta.N_ftest])

ax_2.tick_params(which='major', labelsize=6.5)
ax_3.tick_params(which='major', labelsize=6.5)

for i, a in enumerate(ax):
    a.spines["top"].set_alpha(0.8)
    a.spines["bottom"].set_alpha(0.8)
    a.spines["right"].set_alpha(0.8)
    a.spines["left"].set_alpha(0.8)
ax_2.spines['top'].set_alpha(0.8)
ax_3.spines['top'].set_alpha(0.8)

[i.set_alpha(0.8) for i in ax[0].get_xticklabels()]
[i.set_alpha(0.8) for i in ax[1].get_xticklabels()]
[i.set_alpha(0.8) for i in ax[0].get_yticklabels()]
[i.set_alpha(0.8) for i in ax[1].get_yticklabels()]

ax[0].tick_params(axis='both', left='off', top='off', bottom='off')
ax[1].tick_params(axis='both', left='off', top='off', bottom='off')
ax_2.tick_params(axis='both', top='off')
ax_3.tick_params(axis='both', top='off')

major_ticks_ttest_x = np.linspace(min(datos_tanque.detect_delta_media*100/2), max(datos_tanque.detect_delta_media*100/2), 5, endpoint=True)
major_ticks_ftest_x = np.linspace(min(datos_tanque.detect_delta_var*100/np.var(fallas_tanque.nivel_ee)), max(datos_tanque.detect_delta_var*100/np.var(fallas_tanque.nivel_ee)), 5, endpoint=True)
major_ticks_y = np.linspace(0, 1, 5, endpoint=True)

ax[0].set_xticklabels(['{:.2e}'.format(i) for i in major_ticks_ttest_x])
ax[1].set_xticklabels(['{:.2e}'.format(i) for i in major_ticks_ftest_x])

ax[0].set_xlim([min(datos_tanque.detect_delta_media*100/2)-min(datos_tanque.detect_delta_media*100/2)/10,
                max(datos_tanque.detect_delta_media*100/2)+min(datos_tanque.detect_delta_media*100/2)/10])
ax[1].set_xlim([min(datos_tanque.detect_delta_var*100/np.var(fallas_tanque.nivel_ee))-min(datos_tanque.detect_delta_var*100/np.var(fallas_tanque.nivel_ee))/2,
                max(datos_tanque.detect_delta_var*100/np.var(fallas_tanque.nivel_ee))+min(datos_tanque.detect_delta_var*100/np.var(fallas_tanque.nivel_ee))/2])

ax[0].xaxis.set_ticks(major_ticks_ttest_x)
ax[1].xaxis.set_ticks(major_ticks_ftest_x)
ax[0].yaxis.set_ticks(major_ticks_y)
ax[1].yaxis.set_ticks(major_ticks_y)

fig.subplots_adjust(top=0.83, hspace=0.8)
fig.suptitle(r'Promedios de FDR y FAR por Método y $\Delta\mu$ y $\Delta\sigma^2$ a Detectar', size=12, color='k',
             fontweight='bold', alpha=0.7)
plt.show()

