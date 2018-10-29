import pandas as pd
import numpy as np
import fault_2 as f
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy import stats
import random

random.seed(0)
x = sorted(np.random.normal(0, 1, 1000))
fig, ax = plt.subplots()
ax.plot(x, stats.norm.pdf(x, np.mean(x), np.std(x)))
ax.hist(x, normed=True, alpha=0.7, bins=15)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_ylabel('Frecuencia')
#xtick_labels = [r'$\mu - 3\sigma$', r'$\mu - 2\sigma$', r'$\mu - 1\sigma$', r'$\mu$',
#                r'$\mu + 1\sigma$', r'$\mu + 2\sigma$', r'$\mu + 3\sigma$']
#plt.xticks(np.linspace(-3, 3, 7, endpoint=True), xtick_labels)
plt.savefig('norm_dist', dpi=900)
plt.show()

pd.set_option('precision', 5)

data_non_faulty = pd.read_excel('C:/Users/User/Documents/Python/Tesis/Datos Fallas TEP/xmeas_non_faulty.xlsx', decimal=',', header=None)
tuple_list = [i for i in zip(np.arange(0,51), np.arange(1,52))]
data_non_faulty.rename(columns=dict(tuple_list), inplace=True)
data_non_faulty = data_non_faulty[np.arange(1,42)]
#data_non_faulty_normed = (data_non_faulty-data_non_faulty.mean())/data_non_faulty.std()

data_faulty = pd.read_csv('C:/Users/User/Documents/Python/Tesis/Datos Fallas TEP/idv4/y.dat', sep='	 ', header=None, decimal='.')
tuple_list = [i for i in zip(np.arange(0,51), np.arange(1,52))]
data_faulty.rename(columns=dict(tuple_list), inplace=True)
data_faulty = data_faulty[np.arange(1, 42)]
#data_faulty_normed = (data_faulty-data_non_faulty.mean())/data_non_faulty.std()

for i in range(1,42):
    data_non_faulty[i] -= data_non_faulty[i].mean() - data_faulty[i].mean()

plt.figure()
plt.plot(np.linspace(0,50,301, endpoint=True), data_faulty[9], color='red')
plt.plot(np.linspace(0,50,301, endpoint=True), data_non_faulty[9], color='blue')
plt.vlines(1, min(data_faulty[9]), max(data_non_faulty[9]), colors='green')

plt.figure()
plt.plot(np.linspace(0,50,301, endpoint=True), data_faulty[21], color='red')
plt.plot(np.linspace(0,50,301, endpoint=True), data_non_faulty[21], color='blue')
plt.vlines(1, min(data_faulty[21]), max(data_non_faulty[21]), colors='green')

plt.figure()
plt.plot(np.linspace(0,50,301, endpoint=True), data_faulty[7], color='red')
plt.plot(np.linspace(0,50,301, endpoint=True), data_non_faulty[7], color='blue')
plt.vlines(1, min(data_faulty[7]), max(data_non_faulty[7]), colors='green')

#plt.show()

stats_data_non_faulty = data_non_faulty.describe(include='all')
stats_data_faulty = data_faulty.describe(include='all')

# print((stats_data_non_faulty - stats_data_faulty)*100/stats_data_non_faulty)


df_bool_ttest = pd.DataFrame([])
FDR_ttest = np.array([])
FAR_ttest = np.array([])

bool_y = np.zeros(len(data_faulty))
bool_y[10::] = 1

# delta_mean_

for i in range(1,42):
    delta_mean = data_non_faulty[i].mean()*0.001
    print(i, delta_mean)
    a,b,c,d = f.fault_detector(data_faulty[i]).t_test(non_faulty_data=data_non_faulty[i].values,
                                                      stand_dev=data_non_faulty[i].std(), conf_lev=0.95,
                                                      delta_mean=1)
    df_bool_ttest[i] = b

    tn, fp, fn, tp = confusion_matrix(y_true=bool_y, y_pred=b).ravel()
    FDR_ttest = np.append(FDR_ttest, tp/(tp+fn))
    FAR_ttest = np.append(FAR_ttest, fp/(fp+tn))

print(FDR_ttest)
print(FAR_ttest)

#print(df_bool_ttest)
