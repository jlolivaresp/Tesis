import numpy as np
import fault as f
import matplotlib.pyplot as plt
import random

# Primera grafica

a = [0.59, 0.23, 0.00]
b = [0.03, 0.17, 0.00]
c = [0.04, 0.95, 1.00]
d = [0.64, 0.69, 0.15]
pos = [2, 1, 0]
width = 0.25
fig, ax = plt.subplots(ncols=2, sharey=True)
ax[0].barh(pos, a, width, color='#4682B4', alpha=0.7, label='t-test')
ax[1].barh(pos, b, width, color='#4682B4', alpha=0.5, label='t-test')
ax[0].barh([p + width + 0.01 for p in pos], c, width, color='#FF7F50', alpha=0.7, label='F-test')
ax[1].barh([p + width + 0.01 for p in pos], d, width, color='#FF7F50', alpha=0.5, label='F-test')

legends = ax[1].legend(frameon=False)
for l in legends.get_texts():
    l.set_color('k')
    l.set_alpha(0.7)

ax[0].set_title('FDR', color='k',alpha=0.7)
ax[1].set_title('FAR', color='k',alpha=0.7)

ax[0].invert_xaxis()
fig.subplots_adjust(wspace=0)

for i, v in enumerate(a[::-1]):
    ax[0].text(v + 0.15, i , str(v), color='#4682B4', alpha=0.7, fontweight='bold')
for i, v in enumerate(b[::-1]):
    ax[1].text(v + 0.01, i , str(v), color='#4682B4', alpha=0.5, fontweight='bold')
for i, v in enumerate(c[::-1]):
    ax[0].text(v + 0.15, i + width, str(v), color='#FF7F50', alpha=0.7, fontweight='bold')
for i, v in enumerate(d[::-1]):
    ax[1].text(v + 0.01, i + width, str(v), color='#FF7F50', alpha=0.5, fontweight='bold')

ax[0].text(1 + 0.28, 2 + width/2, 'A', color='k', alpha=0.9)
ax[0].text(1 + 0.28, 1 + width/2, 'B', color='k', alpha=0.9)
ax[0].text(1 + 0.28, 0 + width/2, 'C', color='k', alpha=0.9)

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

# Segunda grafica
sep_1 = 0.001
sep_2 = 0.01

fdr_ttest = [0.3507, 0.2577, 0.2081]
far_ttest = [0.0964, 0.0631, 0.0403]
fdr_ftest = [0.66067037, 0.662757407, 0.659128889]
far_ftest = [0.500240741, 0.520685185, 0.456425926]


fdr_ttest_drift = [0.127777778, 0.0003055556, 0.000211111]
fdr_ttest_pulse = [0.423888889, 0.299444444, 0.201333333]
fdr_ttest_var = [0.500555556, 0.470555556, 0.422611111]

far_ttest_drift = [0.016111111, 0.001944444, 0]
far_ttest_pulse = [0.1115, 0.065888889, 0.031611111]
far_ttest_var = [0.161666667, 0.121555556, 0.089222222]

fdr_ftest_drift =[0.687544444, 0.681688889, 0.67245]
fdr_ftest_pulse = [0.6491, 0.655288889, 0.654646667]
fdr_ftest_var = [0.645366667, 0.651294444, 0.65029]

far_ftest_drift = [0.538388889, 0.565555556, 0.497833333]
far_ftest_pulse = [0.483388889, 0.4965, 0.440166667]
far_ftest_var = [0.478944444, 0.5, 0.431277778]

delta_media = [0.01, 0.02, 0.03]
delta_var = [0.1, 0.2, 0.3]

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.subplots_adjust(wspace=0.05)

ax[0].scatter(delta_media, fdr_ttest, color='#4682B4', alpha=0.8)
ax[0].scatter(delta_media, far_ttest, color='#FF7F50', alpha=0.8)
ax[0].set_title('t-test', alpha=0.8)
ax[0].set_xlabel(r'$\Delta\mu$', color='k',alpha=0.8)

ax[1].scatter(delta_var, fdr_ftest, color='#4682B4', alpha=0.8, label='FDR')
ax[1].scatter(delta_var, far_ftest, color='#FF7F50', alpha=0.8, label='FAR')
ax[1].set_title('F-test', alpha=0.8)
ax[1].set_xlabel(r'$\Delta\sigma^2$', color='k',alpha=0.8)

legends = ax[1].legend(frameon=False, loc='lower right')
for l in legends.get_texts():
    l.set_alpha(0.8)


'''
ax[0].scatter(delta_media, fdr_ttest_drift, color='b', alpha=0.8)
ax[0].scatter(delta_media, fdr_ttest_pulse, color='b', alpha=0.5)
ax[0].scatter(delta_media, fdr_ttest_var, color='b', alpha=0.2)
ax[0].scatter(delta_media, far_ttest_drift, color='r', alpha=0.8)
ax[0].scatter(delta_media, far_ttest_pulse, color='r', alpha=0.5)
ax[0].scatter(delta_media, far_ttest_var, color='r', alpha=0.2)

ax[1].scatter(delta_var, fdr_ftest_drift, color='b', alpha=0.8)
ax[1].scatter(delta_var, fdr_ftest_pulse, color='b', alpha=0.5)
ax[1].scatter(delta_var, fdr_ftest_var, color='b', alpha=0.2)
ax[1].scatter(delta_var, far_ftest_drift, color='r', alpha=0.8)
ax[1].scatter(delta_var, far_ftest_pulse, color='r', alpha=0.5)
ax[1].scatter(delta_var, far_ftest_var, color='r', alpha=0.2)
'''

N_ttest = ['852', '213', '95']
N_ftest = ['134', '34', '16']

for i, j, k in zip(delta_media,fdr_ttest,N_ttest):
    ax[0].annotate('N = {}'.format(k), xy=(i+sep_1, j+sep_1), fontsize=7, alpha=0.8)

for i, j, k in zip(delta_var,far_ftest,N_ftest):
    ax[1].annotate('N = {}'.format(k), xy=(i+sep_2, j+sep_2), fontsize=7, alpha=0.8)


for i, a in enumerate(ax):
    a.spines["top"].set_visible(False)
    a.spines["bottom"].set_visible(False)
    a.spines["right"].set_visible(False)
ax[0].spines["left"].set_visible(False)
ax[0].spines['left'].set_color('k')
ax[1].spines['left'].set_alpha(0.8)

[i.set_alpha(0.8) for i in plt.gca().get_xticklabels()]
[i.set_alpha(0.8) for i in ax[0].get_xticklabels()]
[i.set_alpha(0.8) for i in ax[1].get_yticklabels()]

ax[0].tick_params(axis='both', left='off', top='off', bottom='off')
ax[1].tick_params(axis='x', right='off', top='off', bottom='off')
ax[0].axes.get_yaxis().set_visible(False)
ax[1].axes.get_yaxis().set_visible(True)

major_ticks_ttest_x = np.arange(0.01, 0.04, 0.01)
major_ticks_ftest_x = np.arange(0.1, 0.4, 0.1)
major_ticks_ftest_y = np.arange(0, 1.2, 0.2)

ax[0].set_xticks(major_ticks_ttest_x)
ax[1].set_xticks(major_ticks_ftest_x)
ax[1].set_yticks(major_ticks_ftest_y)

ax[0].set_xlim([0.005,0.035])
ax[1].set_xlim([0.05,0.35])

fig.subplots_adjust(top=0.83,bottom=0.1)
fig.suptitle(r'Promedios de FDR y FAR por Método y $\Delta\mu$ y $\Delta\sigma^2$ a Detectar',size=12, color='k',
             fontweight='bold',alpha=0.7)

plt.show()