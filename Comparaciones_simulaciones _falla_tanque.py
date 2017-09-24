from simultank_F1_variable import simultank
from aumento_gradual_r import aumento_gradual_r
from vector_caudal import caudal
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from timeit import default_timer as timer
from math import exp
import fault as f
from sklearn.metrics import confusion_matrix
import seaborn as sbn
import matplotlib.ticker as mtick
import matplotlib as mpl

# # Deteccion de fallas t-test

# Drift: Resistencia hidraulica - Variacion de la pendiente del drift y del tamano de la ventana de prueba

r_inicial = 0.28                        # Valor inicial de la resistencia hidraulica
delta_r = [0.01, 0.04, 0.12]            # Valor final de la resistencia hidraulica (Intensidad del drift)
longitud = 4000                         # Tiempo de simulacion [h] (5 meses y medio)
paso = 0.1                              # Paso de integracion [h]
decimal = str(paso)[::-1].find('.')     # Numero de decimales del paso
q_set = 24                              # Caudal de entrada [m^3/h]
ruido = True                            # Ruido en el caudal de entrada
area = 2                                # Area transversal del tanque [m^2]
nivel_inicial = 0                       # Nivel inicial del tanque [m]
t_inicial = 0                           # Tiempo inciial de la simulacion [h]
t_final = longitud                      # Tiempo final de la simulacion [h]
conf_lev = 0.95                         # Nivel de confiabilidad
tiempo = np.arange(t_inicial,t_final,paso)              # Vector de rangos de tiempo de la simulacion
tss_2 = int(np.round(4*area*r_inicial,decimal)/paso)    # Tiempo de establecimiento del nivel
t_i_falla = 200
t_f_falla_drift = [200.2, 500, 1000, 2000, 3000, 4000]
delta_media = [0.01, 0.02, 0.03]

r = np.ones(longitud/paso)*r_inicial
true_positives_ttest_drift = np.zeros([len(delta_media), len(t_f_falla_drift)])
false_positives_ttest_drift = np.zeros([len(delta_media), len(t_f_falla_drift)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
fdr_far_ttest = np.zeros([2,3])
avg_tp_ttest_drift = np.array([])
avg_fp_ttest_drift = np.array([])

fdr_far_ttest_tp = np.zeros([3,6])
fdr_far_ttest_fp = np.zeros([3,6])

for i in delta_r:
    print(i)
    for k,kk in zip(delta_media,range(0,len(delta_media))):
        pendiente = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(t_f_falla_drift,range(0,len(t_f_falla_drift))):
            r_falla, slope, y_test = f.fault_generator(r).drift(start=t_i_falla,stop=j,step=paso,change=i)
            nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r_falla,caudal_entrada=Q,
                              tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
            pendiente = np.append(pendiente,[slope])
            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel[tss_2:]).t_test(stand_dev=1, conf_lev=conf_lev,
                                      delta_mean=nivel[tss_2:t_i_falla/paso].mean()*k, N='auto')
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives_ttest_drift[kk, jj] = tp/(tp+fn)
            false_positives_ttest_drift[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]

    avg_tp_ttest_drift = np.append(avg_tp_ttest_drift, np.average(true_positives_ttest_drift))
    avg_fp_ttest_drift = np.append(avg_fp_ttest_drift, np.average(false_positives_ttest_drift))
    fdr_far_ttest_tp += true_positives_ttest_drift
    fdr_far_ttest_fp += false_positives_ttest_drift
avg_tp_ttest_drift = np.average(avg_tp_ttest_drift)
avg_fp_ttest_drift = np.average(avg_fp_ttest_drift)
fdr_far_ttest[0,0] = avg_tp_ttest_drift
fdr_far_ttest[1,0] = avg_fp_ttest_drift

# pulse: lecturas del sensor variando la densidad de pulsasiones

r_inicial = 0.28                        # Valor inicial de la resistencia hidraulica
longitud = 360                         # Tiempo de simulacion [h] (15 dias)
paso = 0.1                              # Paso de integracion [h]
decimal = str(paso)[::-1].find('.')     # Numero de decimales del paso
q_set = 24                              # Caudal de entrada [m^3/h]
ruido = True                            # Ruido en el caudal de entrada
area = 2                                # Area transversal del tanque [m^2]
nivel_inicial = 0                       # Nivel inicial del tanque [m]
t_inicial = 0                           # Tiempo inciial de la simulacion [h]
t_final = longitud                      # Tiempo final de la simulacion [h]
conf_lev = 0.95                         # Nivel de confiabilidad
tiempo = np.arange(t_inicial,t_final,paso)              # Vector de rangos de tiempo de la simulacion
tss_2 = int(np.round(4*area*r_inicial,decimal)/paso)    # Tiempo de establecimiento del nivel
t_i_falla = 60
t_f_falla_pulse = 360
N_faults = [3, 30, 300, 750, 1500, 2250]
pulse_intensity = [0.1, 1, 2]
delta_media = [0.01, 0.02, 0.03]

r = np.ones(longitud/paso)*r_inicial
true_positives_ttest_pulse = np.zeros([len(delta_media), len(N_faults)])
false_positives_ttest_pulse = np.zeros([len(delta_media), len(N_faults)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
avg_tp_ttest_pulse = np.array([])
avg_fp_ttest_pulse = np.array([])

for i in pulse_intensity:
    print(i)
    nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                      tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
    for k,kk in zip(delta_media,range(0,len(delta_media))):
        frac_injected_y = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(N_faults,range(0,len(N_faults))):
            nivel_falla, frac_injected, y_test = f.fault_generator(nivel). \
                random_pulse(start=t_i_falla,stop=t_f_falla_pulse,step=paso,N=j,
                             amplitude=i, random_seed=0, mode='random')
            frac_injected_y = np.append(frac_injected_y,[frac_injected])

            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel_falla[tss_2:]).t_test(stand_dev=1, conf_lev=conf_lev,
                                            delta_mean=nivel[tss_2:t_i_falla/paso].mean()*k, N='auto')
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives_ttest_pulse[kk, jj] = tp/(tp+fn)
            false_positives_ttest_pulse[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]
    avg_tp_ttest_pulse = np.append(avg_tp_ttest_pulse, np.average(true_positives_ttest_pulse))
    avg_fp_ttest_pulse = np.append(avg_fp_ttest_pulse, np.average(false_positives_ttest_pulse))
    fdr_far_ttest_tp += true_positives_ttest_pulse
    fdr_far_ttest_fp += false_positives_ttest_pulse
avg_tp_ttest_pulse = np.average(avg_tp_ttest_pulse)
avg_fp_ttest_pulse = np.average(avg_fp_ttest_pulse)
fdr_far_ttest[0,1] = avg_tp_ttest_pulse
fdr_far_ttest[1,1] = avg_fp_ttest_pulse

# Variance

r_inicial = 0.28                        # Valor inicial de la resistencia hidraulica
longitud = 360                         # Tiempo de simulacion [h] (15 dias)
paso = 0.1                              # Paso de integracion [h]
decimal = str(paso)[::-1].find('.')     # Numero de decimales del paso
q_set = 24                              # Caudal de entrada [m^3/h]
ruido = True                            # Ruido en el caudal de entrada
area = 2                                # Area transversal del tanque [m^2]
nivel_inicial = 0                       # Nivel inicial del tanque [m]
t_inicial = 0                           # Tiempo inciial de la simulacion [h]
t_final = longitud                      # Tiempo final de la simulacion [h]
conf_lev = 0.95                         # Nivel de confiabilidad
tiempo = np.arange(t_inicial,t_final,paso)              # Vector de rangos de tiempo de la simulacion
tss_2 = int(np.round(4*area*r_inicial,decimal)/paso)    # Tiempo de establecimiento del nivel
t_i_falla = 60
t_f_falla_var = [60.3, 63, 90, 135, 210, 285]
std = [0.3162278, 0.4472136, 0.5477226]
var = [0.1, 0.2, 0.3]
delta_media = [0.01, 0.02, 0.03]

r = np.ones(longitud/paso)*r_inicial
true_positives_ttest_var = np.zeros([len(delta_media), len(t_f_falla_var)])
false_positives_ttest_var = np.zeros([len(delta_media), len(t_f_falla_var)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
avg_tp_ttest_var = np.array([])
avg_fp_ttest_var = np.array([])

for i,ii in zip(std, var):
    print(i)
    nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                      tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
    for k,kk in zip(delta_media,range(0,len(delta_media))):
        frac_injected_y = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(t_f_falla_var,range(0,len(t_f_falla_var))):
            nivel_falla, frac_injected, y_test = f.fault_generator(nivel). \
                variance(start=t_i_falla,stop=j,step=paso,stand_dev=i, random_seed=0)
            frac_injected_y = np.append(frac_injected_y,[frac_injected])

            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel_falla[tss_2:]).t_test(stand_dev=1, conf_lev=conf_lev,
                                            delta_mean=nivel[tss_2:t_i_falla/paso].mean()*k, N='auto')
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives_ttest_var[kk, jj] = tp/(tp+fn)
            false_positives_ttest_var[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]
    avg_tp_ttest_var = np.append(avg_tp_ttest_var, np.average(true_positives_ttest_var))
    avg_fp_ttest_var = np.append(avg_fp_ttest_var, np.average(false_positives_ttest_var))
    fdr_far_ttest_tp += true_positives_ttest_var
    fdr_far_ttest_fp += false_positives_ttest_var
avg_tp_ttest_var = np.average(avg_tp_ttest_var)
avg_fp_ttest_var = np.average(avg_fp_ttest_var)
fdr_far_ttest[0,2] = avg_tp_ttest_var
fdr_far_ttest[1,2] = avg_fp_ttest_var

print(fdr_far_ttest)

# # Deteccion de fallas F-test

# Drift: Resistencia hidraulica - Variacion de la pendiente del drift y del tamano de la ventana de prueba

r_inicial = 0.28                        # Valor inicial de la resistencia hidraulica
delta_r = [0.01, 0.04, 0.12]            # Valor final de la resistencia hidraulica (Intensidad del drift)
longitud = 4000                         # Tiempo de simulacion [h] (5 meses y medio)
paso = 0.1                              # Paso de integracion [h]
decimal = str(paso)[::-1].find('.')     # Numero de decimales del paso
q_set = 24                              # Caudal de entrada [m^3/h]
ruido = True                            # Ruido en el caudal de entrada
area = 2                                # Area transversal del tanque [m^2]
nivel_inicial = 0                       # Nivel inicial del tanque [m]
t_inicial = 0                           # Tiempo inciial de la simulacion [h]
t_final = longitud                      # Tiempo final de la simulacion [h]
conf_lev = 0.95                         # Nivel de confiabilidad
tiempo = np.arange(t_inicial,t_final,paso)              # Vector de rangos de tiempo de la simulacion
tss_2 = int(np.round(4*area*r_inicial,decimal)/paso)    # Tiempo de establecimiento del nivel
t_i_falla = 200
t_f_falla_drift = [200.2, 500, 1000, 2000, 3000, 4000]
delta_var = [0.1, 0.2, 0.3]

r = np.ones(longitud/paso)*r_inicial
true_positives_ftest_drift = np.zeros([len(delta_var), len(t_f_falla_drift)])
false_positives_ftest_drift = np.zeros([len(delta_var), len(t_f_falla_drift)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
fdr_far_ftest = np.zeros([2,3])
avg_tp_ftest_drift = np.array([])
avg_fp_ftest_drift = np.array([])

fdr_far_ftest_tp = np.zeros([3,6])
fdr_far_ftest_fp = np.zeros([3,6])

for i in delta_r:
    print(i)
    for k,kk in zip(delta_var,range(0,len(delta_var))):
        pendiente = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(t_f_falla_drift,range(0,len(t_f_falla_drift))):
            r_falla, slope, y_test = f.fault_generator(r).drift(start=t_i_falla,stop=j,step=paso,change=i)
            nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r_falla,caudal_entrada=Q,
                              tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
            pendiente = np.append(pendiente,[slope])
            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel[tss_2:]).f_test(std=nivel.std(),delta_var=nivel.std()*k, conf_lev=conf_lev, N='auto')
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives_ftest_drift[kk, jj] = tp/(tp+fn)
            false_positives_ftest_drift[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]
    avg_tp_ftest_drift = np.append(avg_tp_ftest_drift, np.average(true_positives_ftest_drift))
    avg_fp_ftest_drift = np.append(avg_fp_ftest_drift, np.average(false_positives_ftest_drift))
    fdr_far_ftest_tp += true_positives_ftest_drift
    fdr_far_ftest_fp += false_positives_ftest_drift
avg_tp_ftest_drift = np.average(avg_tp_ftest_drift)
avg_fp_ftest_drift = np.average(avg_fp_ftest_drift)
fdr_far_ftest[0,0] = avg_tp_ftest_drift
fdr_far_ftest[1,0] = avg_fp_ftest_drift

# pulse: lecturas del sensor variando la densidad de pulsasiones

r_inicial = 0.28                        # Valor inicial de la resistencia hidraulica
longitud = 360                         # Tiempo de simulacion [h] (15 dias)
paso = 0.1                              # Paso de integracion [h]
decimal = str(paso)[::-1].find('.')     # Numero de decimales del paso
q_set = 24                              # Caudal de entrada [m^3/h]
ruido = True                            # Ruido en el caudal de entrada
area = 2                                # Area transversal del tanque [m^2]
nivel_inicial = 0                       # Nivel inicial del tanque [m]
t_inicial = 0                           # Tiempo inciial de la simulacion [h]
t_final = longitud                      # Tiempo final de la simulacion [h]
conf_lev = 0.95                         # Nivel de confiabilidad
tiempo = np.arange(t_inicial,t_final,paso)              # Vector de rangos de tiempo de la simulacion
tss_2 = int(np.round(4*area*r_inicial,decimal)/paso)    # Tiempo de establecimiento del nivel
t_i_falla = 60
t_f_falla_pulse = 360
N_faults = [3, 30, 300, 750, 1500, 2250]
pulse_intensity = [0.1, 1, 2]
delta_var = [0.1, 0.2, 0.3]

r = np.ones(longitud/paso)*r_inicial
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
avg_tp_ftest_pulse = np.array([])
avg_fp_ftest_pulse = np.array([])

for i in pulse_intensity:
    print(i)
    nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                      tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
    true_positives_ftest_pulse = np.zeros([len(delta_var), len(N_faults)])
    false_positives_ftest_pulse = np.zeros([len(delta_var), len(N_faults)])
    for k,kk in zip(delta_var,range(0,len(delta_var))):
        frac_injected_y = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(N_faults,range(0,len(N_faults))):
            nivel_falla, frac_injected, y_test = f.fault_generator(nivel). \
                random_pulse(start=t_i_falla,stop=t_f_falla_pulse,step=paso,N=j,
                             amplitude=i, random_seed=0, mode='random')
            frac_injected_y = np.append(frac_injected_y,[frac_injected])

            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel_falla[tss_2:]).f_test(std=nivel.std(),delta_var=nivel.std()*k, conf_lev=conf_lev, N='auto')
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives_ftest_pulse[kk, jj] = tp/(tp+fn)
            false_positives_ftest_pulse[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]
    avg_tp_ftest_pulse = np.append(avg_tp_ftest_pulse, np.average(true_positives_ftest_pulse))
    avg_fp_ftest_pulse = np.append(avg_fp_ftest_pulse, np.average(false_positives_ftest_pulse))
    fdr_far_ftest_tp += true_positives_ftest_pulse
    fdr_far_ftest_fp += false_positives_ftest_pulse
avg_tp_ftest_pulse = np.average(avg_tp_ftest_pulse)
avg_fp_ftest_pulse = np.average(avg_fp_ftest_pulse)
fdr_far_ftest[0,1] = avg_tp_ftest_pulse
fdr_far_ftest[1,1] = avg_fp_ftest_pulse

# Variance

r_inicial = 0.28                        # Valor inicial de la resistencia hidraulica
longitud = 360                         # Tiempo de simulacion [h] (15 dias)
paso = 0.1                              # Paso de integracion [h]
decimal = str(paso)[::-1].find('.')     # Numero de decimales del paso
q_set = 24                              # Caudal de entrada [m^3/h]
ruido = True                            # Ruido en el caudal de entrada
area = 2                                # Area transversal del tanque [m^2]
nivel_inicial = 0                       # Nivel inicial del tanque [m]
t_inicial = 0                           # Tiempo inciial de la simulacion [h]
t_final = longitud                      # Tiempo final de la simulacion [h]
conf_lev = 0.95                         # Nivel de confiabilidad
tiempo = np.arange(t_inicial,t_final,paso)              # Vector de rangos de tiempo de la simulacion
tss_2 = int(np.round(4*area*r_inicial,decimal)/paso)    # Tiempo de establecimiento del nivel
t_i_falla = 60
t_f_falla_var = [60.3, 63, 90, 135, 210, 285]
std = [0.3162278, 0.4472136, 0.5477226]
var = [0.1, 0.2, 0.3]
delta_var = [0.1, 0.2, 0.3]

r = np.ones(longitud/paso)*r_inicial
true_positives_ftest_var = np.zeros([len(delta_var), len(t_f_falla_var)])
false_positives_ftest_var = np.zeros([len(delta_var), len(t_f_falla_var)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
avg_tp_ftest_var = np.array([])
avg_fp_ftest_var = np.array([])
fdr_far_ftest = np.zeros([2,3])
for i,ii in zip(std, var):
    print(i)
    nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                      tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
    for k,kk in zip(delta_var, range(0, len(delta_var))):
        frac_injected_y = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(t_f_falla_var,range(0,len(t_f_falla_var))):
            nivel_falla, frac_injected, y_test = f.fault_generator(nivel). \
                variance(start=t_i_falla,stop=j,step=paso,stand_dev=i, random_seed=0)
            frac_injected_y = np.append(frac_injected_y,[frac_injected])

            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel_falla[tss_2:]).f_test(std=nivel.std(),delta_var=nivel.std()*k, conf_lev=conf_lev, N='auto')
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives_ftest_var[kk, jj] = tp/(tp+fn)
            false_positives_ftest_var[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]
    avg_tp_ftest_var = np.append(avg_tp_ftest_var, np.average(true_positives_ftest_var))
    avg_fp_ftest_var = np.append(avg_fp_ftest_var, np.average(false_positives_ftest_var))
    fdr_far_ftest_tp += true_positives_ftest_var
    fdr_far_ftest_fp += false_positives_ftest_var
avg_tp_ftest_var = np.average(avg_tp_ftest_var)
avg_fp_ftest_var = np.average(avg_fp_ftest_var)
fdr_far_ftest[0,2] = avg_tp_ftest_var
fdr_far_ftest[1,2] = avg_fp_ftest_var

print(fdr_far_ftest)

avg_fdr_ttest = np.array([np.average(i) for i in fdr_far_ttest_tp])
avg_far_ttest = np.array([np.average(i) for i in fdr_far_ttest_fp])
avg_fdr_ftest = np.array([np.average(i) for i in fdr_far_ftest_tp])
avg_far_ftest = np.array([np.average(i) for i in fdr_far_ftest_fp])

print(avg_fdr_ttest/9)
print(avg_far_ttest/9)
print(avg_fdr_ftest/9)
print(avg_far_ftest/9)
