"""
Simulaciones de distintas fallas en un sistema dinamico compuesto por un tanque con caudal de entrada Q, area
transversal A y resistencia hidraulica de la valvula de salida R.
"""

from simultank_F1_variable import simultank
from aumento_gradual_r import aumento_gradual_r
from vector_caudal import caudal
from fault_detector_ttest import fault_detector_ttest
from fault_detector_F import fault_detector_F
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from timeit import default_timer as timer
from math import exp
import fault as f
from sklearn.metrics import confusion_matrix
import seaborn as sbn

'''
Prueba de funciones simultank, vector_caudal y aumento_gradual_r
'''

r_inicial = 0.28            # Valor inicial de la resistencia hidraulica
r_final = 0.32              # Valor final de la resistencia hidraulica
longitud = 1000               # Tiempo de simulacion [h]
t_inicial_falla_1 = 1       # Tiempo en que inicial la falla [h]
t_final_falla_1 = 47        # Tiempo en que se estabiliza la falla [h]
t_inicial_falla_2 = 10      # Tiempo en que inicial la falla [h]
t_final_falla_2 = 20        # Tiempo en que se estabiliza la falla [h]
t_inicial_falla_3 = 20.1    # Tiempo en que inicial la falla [h]
t_final_falla_3 = 20.3      # Tiempo en que se estabiliza la falla [h]
paso = 0.1                  # Paso de integracion [h]
decimal = str(paso)[::-1].find('.') # Numero de decimales del paso
q_set = 24                  # Caudal de entrada [m^3/h]
ruido = True                # Ruido en el caudal de entrada
area = 2                    # Area transversal del tanque [m^2]
nivel_inicial = 0           # Nivel inicial del tanque [m]
t_inicial = 0               # Tiempo inciial de la simulacion [h]
t_final = longitud          # Tiempo final de la simulacion [h]
conf_lev = 0.95             # Nivel de confiabilidad
tiempo = np.arange(t_inicial,t_final,paso) # Vector de rangos de tiempo de la simulacion
tss_2 = int(np.round(4*area*r_inicial,decimal)/paso) # Tiempo de establecimiento del nivel

# vector_caudal
Qi_ruido_0_01 = caudal(set_point=q_set, longitud=longitud, paso=paso, factor_ruido=0.01, ruido=True)
Qi_ruido_0_005 = caudal(set_point=q_set, longitud=longitud, paso=paso, factor_ruido=0.005, ruido=True)
Qi_no_ruido = caudal(set_point=24, longitud=longitud, paso=paso, ruido=False)
plt.figure(figsize=(9, 3))
plt.plot(tiempo,Qi_ruido_0_01,label='Con ruido (0,01)')
plt.plot(tiempo,Qi_ruido_0_005,label='Con ruido (0,005)')
plt.plot(tiempo,Qi_no_ruido,label='Sin ruido')
plt.title('Caudal vs Tiempo')
plt.ylabel('Caudal (m^3/h)')
plt.xlabel('Tiempo (h)')
plt.legend()
plt.tight_layout()


# aumento_gradual_r
r_1 = aumento_gradual_r(r_inicial=r_inicial, r_final=r_final, longitud=longitud,
                        t_inicial=t_inicial_falla_1, t_final=t_final_falla_1, paso=paso)
r_2 = aumento_gradual_r(r_inicial=r_inicial, r_final=r_final, longitud=longitud,
                        t_inicial=t_inicial_falla_2, t_final=t_final_falla_2, paso=paso)
r_3 = aumento_gradual_r(r_inicial=r_inicial, r_final=r_final, longitud=longitud,
                        t_inicial=t_inicial_falla_3, t_final=t_final_falla_3, paso=paso)
r_4 = np.ones(int(longitud/paso))*r_inicial
r_4[200:210] = 0.5
r_4[300:310] = 0.5
r_5 = np.ones(int(longitud/paso))*r_inicial
plt.figure(figsize=(9, 3))
plt.plot(tiempo,r_1,label='falla 1 (gradual)')
plt.plot(tiempo,r_2,label='falla 2 (gradual)')
plt.plot(tiempo,r_3,label='falla 3 (repentina)')
plt.title('Resistencia Hidraulica vs Tiempo')
plt.ylabel('Resistencia Hidraulica')
plt.xlabel('Tiempo (h)')
plt.legend()
plt.tight_layout()


# simultank
nivel_1 = simultank(area=area, nivel_inicial=nivel_inicial, resist_hidraulica=r_1, caudal_entrada=Qi_ruido_0_01,
                    tiempo_inicial=t_inicial, tiempo_final=t_final, paso=paso, analitic_sol=False)
nivel_2 = simultank(area=area, nivel_inicial=nivel_inicial, resist_hidraulica=r_2, caudal_entrada=Qi_ruido_0_005,
                    tiempo_inicial=t_inicial, tiempo_final=t_final, paso=paso, analitic_sol=False)
nivel_3 = simultank(area=area, nivel_inicial=nivel_inicial, resist_hidraulica=r_3, caudal_entrada=Qi_no_ruido,
                    tiempo_inicial=t_inicial, tiempo_final=t_final, paso=paso, analitic_sol=False)
nivel_4 = simultank(area=area, nivel_inicial=nivel_inicial, resist_hidraulica=r_4, caudal_entrada=Qi_ruido_0_005,
                    tiempo_inicial=t_inicial, tiempo_final=t_final, paso=paso, analitic_sol=False)
nivel_5 = simultank(area=area, nivel_inicial=nivel_inicial, resist_hidraulica=r_5, caudal_entrada=Qi_ruido_0_01,
                    tiempo_inicial=t_inicial, tiempo_final=t_final, paso=paso, analitic_sol=False)
nivel_5[100:150] = nivel_5[100:150]+np.random.normal(0,1,50)
plt.figure(figsize=(9, 3))
plt.plot(tiempo,nivel_1,label='Nivel: Con ruido (0,01) - falla 1 (gradual)')
plt.plot(tiempo,nivel_2,label='Nivel: Con ruido (0,005) - falla 2 (gradual)')
plt.plot(tiempo,nivel_3,label='Nivel: Sin ruido - falla 3 (repentina)')
plt.title('Nivel vs Tiempo')
plt.ylabel('Nivel (m)')
plt.xlabel('Tiempo (h)')
plt.legend()
plt.tight_layout()

'''
# Comparacion de pasos con solucion analitica
plt.figure(figsize=(9, 5))
for paso in np.logspace(-4,0,5,endpoint=True):
    start = timer()
    Qi = caudal(set_point=q_set, longitud=10, paso=paso, ruido=False)
    r = aumento_gradual_r(r_inicial=r_inicial, r_final=r_final, longitud=10, t_inicial=0, t_final=10, paso=paso)
    nivel = simultank(area=area, nivel_inicial=nivel_inicial, resist_hidraulica=r, caudal_entrada=Qi,
                      tiempo_inicial=0, tiempo_final=10, paso=paso,analitic_sol=False)
    end = timer()
    tiempo = end - start
    plt.plot(np.arange(0,10,paso),nivel, label='Ruido: 0,01 - '
                                               'falla gradual - paso = {} - tiempo = {:.4f}s'.format(paso,tiempo))
Qi_analitico = caudal(set_point=q_set, longitud=10, paso=0.01, ruido=False)
r_analitico = aumento_gradual_r(r_inicial=r_inicial, r_final=r_final, longitud=10, t_inicial=0, t_final=10, paso=0.01)
nivel_analitico = simultank(area=area, nivel_inicial=nivel_inicial, resist_hidraulica=r_analitico,
                            caudal_entrada=Qi_analitico,tiempo_inicial=0, tiempo_final=10, paso=0.01,analitic_sol=True)
plt.plot(np.arange(0,10,0.01),nivel_analitico,label='Solucion analitica')
plt.title('Nivel vs Tiempo')
plt.ylabel('Nivel (m)')
plt.xlabel('Tiempo (h)')
plt.legend()
plt.tight_layout()
#plt.show()'''

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
t_f_falla_drift = [200.2, 500, 1000, 2000, 4000]
window_size = [1, 5, 10, 50, 100, 200, 400, 'auto']

r = np.ones(longitud/paso)*r_inicial
true_positives = np.zeros([len(t_f_falla_drift), len(window_size)])
false_positives = np.zeros([len(t_f_falla_drift), len(window_size)])
for i in delta_r:
    pendiente = np.array([])
    for j,jj in zip(t_f_falla_drift,range(0,len(t_f_falla_drift))):
        print(j)
        for k,kk in zip(window_size,range(0,len(window_size))):
            print(k)
            r_falla, slope, y_test = f.fault_generator(r).drift(start=t_i_falla,stop=j,step=paso,change=i)
            Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
            nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r_falla,caudal_entrada=Q,
                              tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel[tss_2:]).t_test(stand_dev=nivel.std(), conf_lev=conf_lev,
                                      delta_mean=nivel[tss_2:t_i_falla/paso].mean()*0.01, N=k)
            windows_size_y_labels = [1, 5, 10, 50, 100, 200, 400, '{}'.format(N_auto)]
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives[jj][kk] = tp/(tp+fn)
            false_positives[jj][kk] = fp/(fp+tp)
            pendiente = np.append(pendiente,[slope])
            if i == delta_r[1] and k == window_size[-1]:
                plt.figure()
                plt.plot(tiempo,nivel,label='Tank level')
                tiempo_falla = tiempo[tss_2:][y_pred == 1]
                plt.scatter(tiempo_falla,vector_detected,c='r',marker='o',alpha=0.2,label='Fault Detected')
                plt.title('Faults detected for tank level simulation\nwindow size = {}, with drift in Hydraulic '
                          'Resistance from 0.28 to {:.2f}'.format(k,i + 0.28))
                if k == window_size[-1]:
                    plt.title('Faults detected for tank level simulation\nwindow size = {}, with drift in Hydraulic '
                              'Resistance from 0.28 to {:.2f}'.format(N_auto,i + 0.28))
                plt.xlabel('Time (h)')
                plt.ylabel('Tank Level (m)')
                plt.legend()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.yaxis.get_major_formatter().set_powerlimits((-4, 5))
    sbn.heatmap(true_positives,annot=True,xticklabels=windows_size_y_labels,yticklabels=pendiente)
    plt.yticks(rotation=0)
    plt.title('True Positive Rate (FDR)\nMoving window t-test (95% confidence, Mean change detected = 1.0%)\n'
              'in Tank level simulation with drift in Hydraulic Resistance from 0.28 to {:.2f}'.format(0.28+i))
    plt.xlabel('Window size')
    plt.ylabel('Drift Slope')
    plt.figure()
    sbn.heatmap(false_positives,annot=True,xticklabels=windows_size_y_labels,yticklabels=pendiente)
    plt.yticks(rotation=0)
    plt.title('False Positive Rate (FAR)\nMoving window t-test (95% confidence, Mean change detected = 1.0%)\n'
              'in Tank level simulation with drift in Hydraulic Resistance from 0.28 to {:.2f}'.format(0.28+i))
    plt.xlabel('Window size')
    plt.ylabel('Drift Slope')
plt.show()

'''
for i in t_f_falla_drift:
    print(i)
    r_falla, frac_injected,faulty_bool = f.fault_generator(r).drift(start=100,stop=i,step=paso,change=0.04)
    Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
    nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r_falla,caudal_entrada=Q,
                      tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
    vector_detected, detected_bool, nro_fallas, c = f.fault_detector(
                            nivel[tss_2:]).t_test(stand_dev=nivel.std(), conf_lev=conf_lev,
                                                  delta_mean=nivel.mean()*0.01, N='auto')
    tn, fp, fn, tp = confusion_matrix(faulty_bool[tss_2:], detected_bool).ravel()
    tiempo_falla = tiempo[tss_2:][detected_bool == 1]
plt.show()
'''
'''
# pulse

N_faults = [1, 10, 50, 100, 500, 1000, 2000, 3000, 5000, 8000]
N_faults_density = np.copy(N_faults)
N_faults_density  = N_faults_density/9900
N_faults_density = ['{:.4f}'.format(i) for i in N_faults_density]
print(N_faults_density)
windows = [5, 10, 25, 50, 100, 200, 400, 'auto']
r = np.ones(longitud/paso)*r_inicial
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                  tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
true_positives = np.zeros([10, 8])
false_positives = np.zeros([10, 8])

for n_faults,i in zip(N_faults,range(0,10)):
    print(i)
    for window,j in zip(windows,range(0,8)):
        nivel_falla, frac_injected, faulty_bool = f.fault_generator(nivel).random_pulse(start=100,stop=1000, step=paso,
                                                                                        N=int(n_faults),amplitude=1,
                                                                                        mode='random')
        vector_detected, detected_bool, nro_fallas, N_auto = f.fault_detector(
            nivel_falla[tss_2:]).t_test(stand_dev=nivel_falla.std(), conf_lev=conf_lev,
                                        delta_mean=nivel_falla.mean()*0.01, N=window)
        tn, fp, fn, tp = confusion_matrix(faulty_bool[tss_2:], detected_bool).ravel()
        true_positives[i][j] = tp/(tp+fn)
        false_positives[i][j] = fp/(fp+tp)
print(true_positives)

windows = [5, 10, 25, 50, 100, 200, 400, '{}'.format(N_auto)]
plt.figure()
sbn.heatmap(true_positives,annot=True,xticklabels=windows,yticklabels=N_faults_density)
plt.yticks(rotation=0)
plt.title('True Positive Rate (FDR)')
plt.xlabel('Window size')
plt.ylabel('Fault density (Faults per time unit)')
plt.figure()
sbn.heatmap(false_positives,annot=True,xticklabels=windows,yticklabels=N_faults_density)
plt.yticks(rotation=0)
plt.title('False Alarm Rate (FAR)')
plt.xlabel('Window size')
plt.ylabel('Fault density (Faults per time unit)')
plt.show()

'''
'''
r_falla = r_4
Q = Qi_ruido_0_005
nivel = nivel_4

df_obstruccion_gradual_valvula_salida = pd.DataFrame({'res_hidraulica': r_falla,
                                                      'caudal_entrada': Q,'nivel_tanque': nivel},index=tiempo)
rango_falla_media, falla,a,b = fault_detector_ttest(nivel[tss_2:], nivel[tss_2:].std(), conf_lev=conf_lev,
                                                delta_mean=nivel[tss_2:].mean()*0.01, N=3)
print('N = {}'.format(b))
tiempo_falla = tiempo[tss_2:][falla == 1]
gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
df_obstruccion_gradual_valvula_salida['nivel_tanque'].plot(ax=ax1)
ax1.scatter(tiempo_falla,rango_falla_media,alpha=0.2,c='r')
ax1.vlines(t_inicial_falla_3,0, max(df_obstruccion_gradual_valvula_salida['nivel_tanque']),
           colors='r',label='Inicio de falla')
ax2.vlines(t_inicial_falla_3,0, max(df_obstruccion_gradual_valvula_salida['res_hidraulica']),
           colors='r',label='Inicio de falla')
df_obstruccion_gradual_valvula_salida['res_hidraulica'].plot(ax=ax2,c='g')
ax2.set_ylim([df_obstruccion_gradual_valvula_salida['res_hidraulica'].min()-
              0.1*df_obstruccion_gradual_valvula_salida['res_hidraulica'].min(),
              df_obstruccion_gradual_valvula_salida['res_hidraulica'].max()+
              0.1*df_obstruccion_gradual_valvula_salida['res_hidraulica'].max()])
plt.xlabel('Tiempo (h)')
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.show()
'''

'''

# Deteccion de fallas F-test

r_falla = r_5
Q = Qi_ruido_0_005
nivel = nivel_5

df_obstruccion_gradual_valvula_salida = pd.DataFrame({'res_hidraulica': r_falla,
                                                      'caudal_entrada': Q,'nivel_tanque': nivel},index=tiempo)
rango_falla_varianza, falla, a, b = fault_detector_F(nivel[tss_2:], delta_var=0.01, conf_lev=conf_lev, N=8)
tiempo_falla = tiempo[tss_2:][falla == 1]
gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
df_obstruccion_gradual_valvula_salida['nivel_tanque'].plot(ax=ax1)
ax1.scatter(tiempo_falla, rango_falla_varianza, alpha=0.2, c='r')
ax1.vlines(t_inicial_falla_3,0, max(df_obstruccion_gradual_valvula_salida['nivel_tanque']),
           colors='r',label='Inicio de falla')
ax2.vlines(t_inicial_falla_3,0, max(df_obstruccion_gradual_valvula_salida['res_hidraulica']),
           colors='r',label='Inicio de falla')
df_obstruccion_gradual_valvula_salida['res_hidraulica'].plot(ax=ax2,c='g')
ax2.set_ylim([df_obstruccion_gradual_valvula_salida['res_hidraulica'].min()-
              0.1*df_obstruccion_gradual_valvula_salida['res_hidraulica'].min(),
              df_obstruccion_gradual_valvula_salida['res_hidraulica'].max()+
              0.1*df_obstruccion_gradual_valvula_salida['res_hidraulica'].max()])
plt.xlabel('Tiempo (h)')
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.show()
'''
