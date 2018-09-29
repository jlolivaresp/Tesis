import pandas as pd
import numpy as np
import fault_2 as f
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data_faulty = pd.read_csv('C:/Users/User/Documents/Python/Tesis/Datos Fallas TEP/idv4/y.dat', sep='	 ', header=None, decimal='.')
tuple_list = [i for i in zip(np.arange(0,51), np.arange(1,52))]
data_faulty.rename(columns=dict(tuple_list), inplace=True)
data_faulty = data_faulty[np.arange(1,42)]

data_non_faulty = pd.read_excel('C:/Users/User/Documents/Python/Tesis/Datos Fallas TEP/xmeas_non_faulty.xlsx', decimal=',', header=None)
tuple_list = [i for i in zip(np.arange(0,51), np.arange(1,52))]
data_non_faulty.rename(columns=dict(tuple_list), inplace=True)
data_non_faulty = data_faulty[np.arange(1,42)]

df_bool_ttest = pd.DataFrame([])
FDR_ttest = np.array([])
FAR_ttest = np.array([])

bool_y = np.zeros(len(data_faulty))
bool_y[10::] = 1

for i in range(1,42):
    a,b,c,d = f.fault_detector(data_faulty[i]).t_test(non_faulty_data=data_non_faulty[i].values,
                                                      stand_dev=data_non_faulty[i].std(), conf_lev=0.95,
                                                      delta_mean=data_non_faulty[i].mean()*0.01)
    df_bool_ttest[i] = b

    tn, fp, fn, tp = confusion_matrix(y_true=bool_y, y_pred=b).ravel()
    FDR_ttest = np.append(FDR_ttest, tp/(tp+fn))
    FAR_ttest = np.append(FAR_ttest, fp/(fp+tn))

print(FDR_ttest)
#print(df_bool_ttest)