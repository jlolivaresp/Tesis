import pandas as pd
import numpy as np
import fault_2 as f
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy import stats
import random

# Leemos el .csv sin fallas
data_non_faulty = pd.read_excel('C:/Users/User/Documents/Python/Tesis/Datos Fallas TEP/xmeas_non_faulty.xlsx',
                                decimal=',', header=None)
tuple_list = [i for i in zip(np.arange(0,51), np.arange(1,52))]
data_non_faulty.rename(columns=dict(tuple_list), inplace=True)
data_non_faulty = data_non_faulty[np.arange(1,42)]

# Para cada data set con falla, iteramos y detectamos las fallas
for k in [3, 4]:
    data_faulty = pd.read_csv('C:/Users/User/Documents/Python/Tesis/Datos Fallas TEP/idv{}/y.dat'.format(k), sep='	 ',
                              header=None, decimal='.')
    tuple_list = [i for i in zip(np.arange(0,51), np.arange(1,52))]
    data_faulty.rename(columns=dict(tuple_list), inplace=True)
    for i in range(1,42):
        data_non_faulty[i] -= data_non_faulty[i].mean() - data_faulty[i].mean()

    bool_y = np.zeros(len(data_faulty))
    bool_y[10::] = 1

    # Aplicamos t-test y F-test
    delta_media = np.linspace(0.01, 0.05, 3, endpoint=True)
    delta_var = np.linspace(0.01, 0.05, 3, endpoint=True)

    t_test_FDR_mat = np.array([])
    t_test_FAR_mat = np.array([])
    t_test_FDR_df = pd.DataFrame(columns=delta_media)
    t_test_FAR_df = pd.DataFrame(columns=delta_media)

    F_test_FDR_mat = np.array([])
    F_test_FAR_mat = np.array([])
    F_test_FDR_df = pd.DataFrame(columns=delta_var)
    F_test_FAR_df = pd.DataFrame(columns=delta_var)

    for i in range(1, 42):
        print(i)
        t_test_tp_mat = np.array([])
        t_test_tn_mat = np.array([])
        t_test_fp_mat = np.array([])
        t_test_fn_mat = np.array([])

        F_test_tp_mat = np.array([])
        F_test_tn_mat = np.array([])
        F_test_fp_mat = np.array([])
        F_test_fn_mat = np.array([])
        for n, m in zip(delta_media, delta_var):
            # t-test
            _, t_test_pred, _, N = f.fault_detector(data_faulty[i]).t_test(non_faulty_data=data_non_faulty[i].values,
                                                                           stand_dev=data_non_faulty[i].std(),
                                                                           conf_lev=0.95,
                                                                           delta_mean=data_non_faulty[i].mean()*n)
            t_test_tn, t_test_fp, t_test_fn, t_test_tp = confusion_matrix(y_true=bool_y, y_pred=t_test_pred).ravel()
            t_test_tp_mat = np.append(t_test_tp_mat, t_test_tp)
            t_test_tn_mat = np.append(t_test_tn_mat, t_test_tn)
            t_test_fp_mat = np.append(t_test_fp_mat, t_test_fp)
            t_test_fn_mat = np.append(t_test_fn_mat, t_test_fn)

            # F-test
            _, F_test_pred, _, N = f.fault_detector(data_faulty[i]).f_test(non_faulty_data=data_non_faulty[i].values,
                                                                           std=data_non_faulty[i].std(),
                                                                           delta_var=np.var(data_non_faulty[i])*m,
                                                                           conf_lev=0.95)
            F_test_tn, F_test_fp, F_test_fn, F_test_tp = confusion_matrix(y_true=bool_y, y_pred=F_test_pred).ravel()
            F_test_tp_mat = np.append(F_test_tp_mat, F_test_tp)
            F_test_tn_mat = np.append(F_test_tn_mat, F_test_tn)
            F_test_fp_mat = np.append(F_test_fp_mat, F_test_fp)
            F_test_fn_mat = np.append(F_test_fn_mat, F_test_fn)

        t_test_FDR = t_test_tp_mat/(t_test_tp_mat+t_test_fn_mat)
        t_test_FAR = t_test_fp_mat/(t_test_fp_mat+t_test_tn_mat)
        t_test_FDR_df.loc[i] = t_test_FDR
        t_test_FAR_df.loc[i] = t_test_FAR

        F_test_FDR = F_test_tp_mat/(F_test_tp_mat+F_test_fn_mat)
        F_test_FAR = F_test_fp_mat/(F_test_fp_mat+F_test_tn_mat)
        F_test_FDR_df.loc[i] = F_test_FDR
        F_test_FAR_df.loc[i] = F_test_FAR

    ax_FDR_t_test = t_test_FDR_df.plot(kind='bar', title='IDV {}'.format(k), figsize=(8, 3), ylim=(0, 1.1), colormap='viridis')
    ax_FDR_t_test.set_xlabel('Variable')
    ax_FDR_t_test.set_ylabel('FDR')
    ax_FDR_t_test.legend([r'$\Delta\mu$ = {}'.format(delta_media[0]), r'$\Delta\mu$ = {}'.format(delta_media[1]),
                          r'$\Delta\mu$ = {}'.format(delta_media[2])])
    ax_FDR_t_test.tick_params(axis='both', which='both', length=0)
    for i in ['right', 'left', 'top', 'bottom']:
        ax_FDR_t_test.spines[i].set_visible(False)
    plt.tight_layout()

    ax_FAR_t_test = t_test_FAR_df.plot(kind='bar', title='IDV {}'.format(k), figsize=(8, 3), ylim=(0, 1.1), colormap='viridis')
    ax_FAR_t_test.set_xlabel('Variable')
    ax_FAR_t_test.set_ylabel('FAR')
    ax_FAR_t_test.legend([r'$\Delta\mu$ = {}'.format(delta_media[0]), r'$\Delta\mu$ = {}'.format(delta_media[1]),
                          r'$\Delta\mu$ = {}'.format(delta_media[2])])
    ax_FAR_t_test.tick_params(axis='both', which='both', length=0)
    for i in ['right', 'left', 'top', 'bottom']:
        ax_FAR_t_test.spines[i].set_visible(False)
    plt.tight_layout()

    ax_FDR_F_test = F_test_FDR_df.plot(kind='bar', title='IDV {}'.format(k), figsize=(8, 3), ylim=(0, 1.1), colormap='viridis')
    ax_FDR_F_test.set_xlabel('Variable')
    ax_FDR_F_test.set_ylabel('FDR')
    ax_FDR_F_test.legend([r'$\Delta\sigma^2$ = {}'.format(delta_media[0]), r'$\Delta\sigma^2$ = {}'.format(delta_media[1]),
                          r'$\Delta\sigma^2$ = {}'.format(delta_media[2])])
    ax_FDR_F_test.tick_params(axis='both', which='both', length=0)
    for i in ['right', 'left', 'top', 'bottom']:
        ax_FDR_F_test.spines[i].set_visible(False)
    plt.tight_layout()

    ax_FAR_F_test = F_test_FAR_df.plot(kind='bar', title='IDV {}'.format(k), figsize=(8, 3), ylim=(0, 1.1), colormap='viridis')
    ax_FAR_F_test.set_xlabel('Variable')
    ax_FAR_F_test.set_ylabel('FAR')
    ax_FAR_F_test.legend([r'$\Delta\sigma^2$ = {}'.format(delta_media[0]), r'$\Delta\sigma^2$ = {}'.format(delta_media[1]),
                          r'$\Delta\sigma^2$ = {}'.format(delta_media[2])])
    ax_FAR_F_test.tick_params(axis='both', which='both', length=0)
    for i in ['right', 'left', 'top', 'bottom']:
        ax_FAR_F_test.spines[i].set_visible(False)
    plt.tight_layout()
plt.show()