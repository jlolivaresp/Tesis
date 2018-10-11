import pandas as pd
import numpy as np
import fault_2 as f
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

pd.set_option('precision', 5)

data_non_faulty = pd.read_excel('C:/Users/User/Documents/Python/Tesis/Datos Fallas TEP/xmeas_non_faulty.xlsx', decimal=',', header=None)
tuple_list = [i for i in zip(np.arange(0,51), np.arange(1,52))]
data_non_faulty.rename(columns=dict(tuple_list), inplace=True)
data_non_faulty = data_non_faulty[np.arange(1,42)]
data_non_faulty_normed = (data_non_faulty-data_non_faulty.mean())/data_non_faulty.std()

data_faulty = pd.read_csv('C:/Users/User/Documents/Python/Tesis/Datos Fallas TEP/idv4/y.dat', sep='	 ', header=None, decimal='.')
tuple_list = [i for i in zip(np.arange(0,51), np.arange(1,52))]
data_faulty.rename(columns=dict(tuple_list), inplace=True)
data_faulty = data_faulty[np.arange(1,42)]
data_faulty_normed = (data_faulty-data_non_faulty.mean())/data_non_faulty.std()

plt.figure()
plt.plot(np.linspace(0,50,301, endpoint=True), data_faulty_normed[9], color='red')
plt.plot(np.linspace(0,50,301, endpoint=True), data_non_faulty_normed[9], color='blue')
plt.vlines(1, min(data_faulty_normed[9]), max(data_non_faulty_normed[9]), colors='green')

plt.figure()
plt.plot(np.linspace(0,50,301, endpoint=True), data_faulty_normed[21], color='red')
plt.plot(np.linspace(0,50,301, endpoint=True), data_non_faulty_normed[21], color='blue')
plt.vlines(1, min(data_faulty[21]), max(data_non_faulty_normed[21]), colors='green')

plt.figure()
plt.plot(np.linspace(0,50,301, endpoint=True), data_faulty_normed[7], color='red')
plt.plot(np.linspace(0,50,301, endpoint=True), data_non_faulty_normed[7], color='blue')
plt.vlines(1, min(data_faulty_normed[7]), max(data_non_faulty_normed[7]), colors='green')

plt.show()

stats_data_non_faulty = data_non_faulty_normed.describe(include='all')
stats_data_faulty = data_faulty_normed.describe(include='all')
print(data_faulty.describe(include='all'))
print(data_non_faulty.describe(include='all'))
# print((stats_data_non_faulty - stats_data_faulty)*100/stats_data_non_faulty)

'''
df_bool_ttest = pd.DataFrame([])
FDR_ttest = np.array([])
FAR_ttest = np.array([])

bool_y = np.zeros(len(data_faulty))
bool_y[10::] = 1

# delta_mean_

for i in range(1,42):
    diff = data_faulty[i] - data_non_faulty[i]
    diff_mean = abs(diff.mean())
    print(i, abs(diff.mean())*100/data_non_faulty[i].mean())
    #print(i,diff_mean)
    a,b,c,d = f.fault_detector(data_faulty[i]).t_test(non_faulty_data=data_non_faulty[i].values,
                                                      stand_dev=data_non_faulty[i].std(), conf_lev=0.95,
                                                      delta_mean=diff_mean)
    df_bool_ttest[i] = b

    tn, fp, fn, tp = confusion_matrix(y_true=bool_y, y_pred=b).ravel()
    FDR_ttest = np.append(FDR_ttest, tp/(tp+fn))
    FAR_ttest = np.append(FAR_ttest, fp/(fp+tn))

print(FDR_ttest)

#print(df_bool_ttest)
'''