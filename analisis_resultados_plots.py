import numpy as np
import fault as f
import matplotlib.pyplot as plt
import random


a = [0.59, 0.23, 0.00]
b = [0.03, 0.17, 0.00]
c = [0.04, 1.00, 1.00]
d = [0.64, 0.15, 0.15]
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
plt.show()
