import matplotlib.pyplot as plt
import numpy as np

ttest_FDR_avg = 0.9487528906876839
ttest_FAR_avg = 0.37917236044049607
ftest_FDR_avg = 0.7625664963312369
ftest_FAR_avg = 0.6950780494418988
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

ttest_FDR_avg_resid = 0.35
ttest_FAR_avg_resid = 0.11
ftest_FDR_avg_resid = 0.02
ftest_FAR_avg_resid = 0.0

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

'''Grafica de FDR y FAR = 0'''

ttest_FDR_avg_resid = 0
ttest_FAR_avg_resid = 0
ftest_FDR_avg_resid = 0
ftest_FAR_avg_resid = 0

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

'''Para el vector de residuos abs norm'''

ttest_FDR_avg_resid = 0.21
ttest_FAR_avg_resid = 0.02
ftest_FDR_avg_resid = 0.16
ftest_FAR_avg_resid = 0.00

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
ax.set(aspect="equal", title='Promedios de FDR y FAR Globales\n(Residuos normalizados)')
plt.show()