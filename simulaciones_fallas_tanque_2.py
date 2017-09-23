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
'''
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
true_positives = np.zeros([len(delta_media), len(t_f_falla_drift)])
false_positives = np.zeros([len(delta_media), len(t_f_falla_drift)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)

for i in delta_r:
    print(i)
    for k,kk in zip(delta_media,range(0,len(delta_media))):
        print(k)
        pendiente = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(t_f_falla_drift,range(0,len(t_f_falla_drift))):
            print(j)
            r_falla, slope, y_test = f.fault_generator(r).drift(start=t_i_falla,stop=j,step=paso,change=i)
            nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r_falla,caudal_entrada=Q,
                              tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
            pendiente = np.append(pendiente,[slope])
            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel[tss_2:]).t_test(stand_dev=1, conf_lev=conf_lev,
                                      delta_mean=nivel[tss_2:t_i_falla/paso].mean()*k, N='auto')
            print(N_auto)
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives[kk, jj] = tp/(tp+fn)
            false_positives[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]

            ax = fign.add_subplot(2,3,jj+1)
            ax.plot(tiempo,nivel,label='Nivel del tanque')
            ax.scatter(tiempo_falla,vector_detected,c='r',marker='o',alpha=0.2,label='Falla detectada')
            plt.title('Pendiente = {:.2e}'.format(pendiente[jj]))
            plt.xlabel('Tiempo (h)')
            plt.ylabel('Nivel del tanque (m)')
            plt.legend(loc=4)
            if jj > 0 and jj != 3:
                plt.gca().axes.yaxis.set_ticklabels([])
                ax.set_ylabel('')
            if jj < 3:
                plt.gca().axes.xaxis.set_ticklabels([])
                ax.set_xlabel('')
        fign.tight_layout()
        fign.subplots_adjust(top=0.85)
        fign.suptitle('Deriva en la Resistencia Hidráulica (R) de la Válvula de Salida de 0.28 a {:.2f}\nTamaño de '
                      'Ventana de Prueba N = {} y Diferencia en la Media a detectar = {}'
                      .format(0.28+i, N_auto,k), size=13)
        fign.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/t-test/Drift/{}-{}.png'.format(i,kk))
    pendiente_y = ['{:.2e}'.format(ii) for ii in pendiente]

    fig, (ax1,ax2,ax_cbar) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]}, figsize=(12,4))
    ax1.get_shared_y_axes().join(ax2)
    g1 = sbn.heatmap(true_positives, annot=True, xticklabels=pendiente_y,
                     yticklabels=delta_media, ax=ax1, vmin=0, vmax=1, cbar=False)
    g2 = sbn.heatmap(false_positives, annot=True, xticklabels=pendiente_y,
                     yticklabels=delta_media, ax=ax2, vmin=0, vmax=1, cbar_ax=ax_cbar)
    ax1.set_title('FDR')
    ax1.set_ylabel('Diferencia en la Media a Detectar')
    ax1.set_xlabel('Pendiente del Incremento')
    ax2.set_title('FAR')
    ax2.set_xlabel('Pendiente del Incremento')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    ax2.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.suptitle('Índices de Detección de Falla en Simulación del Nivel del Tanque por Detección de Cambios en la Media'
                 '\ncon t-test (95% Confiabilidad) en Deriva de Resistencia Hidráulica de Válvula de Salida de 0.28 a '
                 '{:.2f}'.format(0.28+i), size=13)
    fig.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/t-test/Drift/{}.png'.format(i))
'''

# pulse: lecturas del sensor variando la densidad de pulsasiones
'''
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
true_positives = np.zeros([len(delta_media), len(N_faults)])
false_positives = np.zeros([len(delta_media), len(N_faults)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)

for i in pulse_intensity:
    print(i)
    nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                      tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
    for k,kk in zip(delta_media,range(0,len(delta_media))):
        print(k)
        frac_injected_y = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(N_faults,range(0,len(N_faults))):
            print(j)
            nivel_falla, frac_injected, y_test = f.fault_generator(nivel). \
                random_pulse(start=t_i_falla,stop=t_f_falla_pulse,step=paso,N=j,
                             amplitude=i, random_seed=0, mode='random')
            frac_injected_y = np.append(frac_injected_y,[frac_injected])

            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel_falla[tss_2:]).t_test(stand_dev=1, conf_lev=conf_lev,
                                            delta_mean=nivel[tss_2:t_i_falla/paso].mean()*k, N='auto')
            print(N_auto)
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives[kk, jj] = tp/(tp+fn)
            false_positives[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]

            ax = fign.add_subplot(2,3,jj+1)
            ax.plot(tiempo,nivel_falla,label='Nivel del tanque')
            ax.scatter(tiempo_falla,vector_detected,c='r',marker='o',alpha=0.2,label='Falla detectada')
            plt.title('FMF = {:.2e}'.format(frac_injected))
            plt.xlabel('Tiempo (h)')
            plt.ylabel('Nivel del tanque (m)')
            plt.legend(loc=4)
            if jj > 0 and jj != 3:
                plt.gca().axes.yaxis.set_ticklabels([])
                ax.set_ylabel('')
            if jj < 3:
                plt.gca().axes.xaxis.set_ticklabels([])
                ax.set_xlabel('')
        fign.tight_layout()
        fign.subplots_adjust(top=0.82)
        fign.suptitle('Mediciones del Sensor de Nivel con Pulsos de Amplitud Normalmente Distribuida\nDesviación '
                      'Estándar = {:.2f}, Fracción de Muestras con Fallas FMF y Tamaño de Ventana de Prueba N = {}\n'
                      'Diferencia en la Media a Detectar = {}'
                      .format(i,N_auto,k), size=13)
        fign.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/t-test/Pulse/{}-{}.png'.format(i,kk))

    fig, (ax1,ax2,ax_cbar) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]}, figsize=(12,4))
    ax1.get_shared_y_axes().join(ax2)
    frac_injected_y = ['{:.2e}'.format(a) for a in frac_injected_y]
    g1 = sbn.heatmap(true_positives, annot=True, xticklabels=frac_injected_y,
                     yticklabels=delta_media, ax=ax1, vmin=0, vmax=1, cbar=False)
    g2 = sbn.heatmap(false_positives, annot=True, xticklabels=frac_injected_y,
                     yticklabels=delta_media, ax=ax2, vmin=0, vmax=1, cbar_ax=ax_cbar)
    ax1.set_title('FDR')
    ax1.set_ylabel('Diferencia en la Media a Detectar')
    ax1.set_xlabel('Fracción de Muestras con Falla')
    ax2.set_title('FAR')
    ax2.set_xlabel('Fracción de Muestras con Falla')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    ax2.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.suptitle('Índices de Detección de Falla en Simulación del Nivel del Tanque por Detección de Cambios en la Media'
                 '\nt-test (95% Confiabilidad) en Mediciones del Sensor de Nivel con Pulsos de Amplitud Normalmente '
                 'Distribuida (std = {:.2f})'.format(i), size=13)
    fig.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/t-test/Pulse/{}.png'.format(i))
'''

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
true_positives = np.zeros([len(delta_media), len(t_f_falla_var)])
false_positives = np.zeros([len(delta_media), len(t_f_falla_var)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)

for i,ii in zip(std, var):
    print(i)
    nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                      tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
    for k,kk in zip(delta_media,range(0,len(delta_media))):
        print(k)
        frac_injected_y = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(t_f_falla_var,range(0,len(t_f_falla_var))):
            print(j)
            nivel_falla, frac_injected, y_test = f.fault_generator(nivel). \
                variance(start=t_i_falla,stop=j,step=paso,stand_dev=i, random_seed=0)
            frac_injected_y = np.append(frac_injected_y,[frac_injected])

            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel_falla[tss_2:]).t_test(stand_dev=1, conf_lev=conf_lev,
                                            delta_mean=nivel[tss_2:t_i_falla/paso].mean()*k, N='auto')
            print(N_auto)
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives[kk, jj] = tp/(tp+fn)
            false_positives[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]

            ax = fign.add_subplot(2,3,jj+1)
            ax.plot(tiempo,nivel_falla,label='Nivel del tanque')
            ax.scatter(tiempo_falla,vector_detected,c='r',marker='o',alpha=0.2,label='Falla detectada')
            plt.title('FMF = {:.2e}'.format(frac_injected))
            plt.xlabel('Tiempo (h)')
            plt.ylabel('Nivel del tanque (m)')
            plt.legend(loc=4)
            if jj > 0 and jj != 3:
                plt.gca().axes.yaxis.set_ticklabels([])
                ax.set_ylabel('')
            if jj < 3:
                plt.gca().axes.xaxis.set_ticklabels([])
                ax.set_xlabel('')
        fign.tight_layout()
        fign.subplots_adjust(top=0.82)
        fign.suptitle('Incremento en la Varianza de las Lecturas del Sensor de Nivel del Tanque\nVarianza = '
                      '{:.2f}, Fracción de Muestras con Falla = FMF y Tamaño de la Ventana de Prueba N = {}\n'
                      'Diferencia en la Media a Detectar = {}'
                      .format(ii,N_auto,k), size=13)
        fign.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/t-test/Variance/{}-{}.png'.format(i,kk))

    fig, (ax1,ax2,ax_cbar) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]}, figsize=(12,4))
    ax1.get_shared_y_axes().join(ax2)
    frac_injected_y = ['{:.2e}'.format(a) for a in frac_injected_y]
    g1 = sbn.heatmap(true_positives, annot=True, xticklabels=frac_injected_y,
                     yticklabels=delta_media, ax=ax1, vmin=0, vmax=1, cbar=False)
    g2 = sbn.heatmap(false_positives, annot=True, xticklabels=frac_injected_y,
                     yticklabels=delta_media, ax=ax2, vmin=0, vmax=1, cbar_ax=ax_cbar)
    ax1.set_title('FDR')
    ax1.set_ylabel('Diferencia en la Media a Detectar')
    ax1.set_xlabel('Fracción de Muestras con Falla')
    ax2.set_title('FAR')
    ax2.set_xlabel('Fracción de Muestras con Falla')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    ax2.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.suptitle('Índices de Detección de Falla en Simulación del Nivel del Tanque por Detección de Cambios en la Media'
                 '\nt-test (95% Confiabilidad) en Mediciones del Sensor de Nivel con Incremento de Varianza '
                 '= {:.2f}'.format(ii),size=13)
    fig.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/t-test/Variance/{}.png'.format(i))
#plt.show()


# # Deteccion de fallas F-test

# Drift: Resistencia hidraulica - Variacion de la pendiente del drift y del tamano de la ventana de prueba
'''
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
true_positives = np.zeros([len(delta_var), len(t_f_falla_drift)])
false_positives = np.zeros([len(delta_var), len(t_f_falla_drift)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)

for i in delta_r:
    print(i)
    for k,kk in zip(delta_var,range(0,len(delta_var))):
        print(k)
        pendiente = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(t_f_falla_drift,range(0,len(t_f_falla_drift))):
            print(j)
            r_falla, slope, y_test = f.fault_generator(r).drift(start=t_i_falla,stop=j,step=paso,change=i)
            nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r_falla,caudal_entrada=Q,
                              tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
            pendiente = np.append(pendiente,[slope])
            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel[tss_2:]).f_test(std=nivel.std(),delta_var=nivel.std()*k, conf_lev=conf_lev, N='auto')
            print(N_auto)
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives[kk, jj] = tp/(tp+fn)
            false_positives[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]

            ax = fign.add_subplot(2,3,jj+1)
            ax.plot(tiempo,nivel,label='Nivel del tanque')
            ax.scatter(tiempo_falla,vector_detected,c='r',marker='o',alpha=0.2,label='Falla detectada')
            plt.title('Pendiente = {:.2e}'.format(pendiente[jj]))
            plt.xlabel('Tiempo (h)')
            plt.ylabel('Nivel del tanque (m)')
            plt.legend(loc=4)
            if jj > 0 and jj != 3:
                plt.gca().axes.yaxis.set_ticklabels([])
                ax.set_ylabel('')
            if jj < 3:
                plt.gca().axes.xaxis.set_ticklabels([])
                ax.set_xlabel('')
        fign.tight_layout()
        fign.subplots_adjust(top=0.85)
        fign.suptitle('Deriva en la Resistencia Hidráulica (R) de la Válvula de Salida de 0.28 a {:.2f}\nTamaño de '
                      'Ventana de Prueba N = {} y Diferencia en la Varianza a detectar = {}'
                      .format(0.28+i, N_auto,k), size=13)
        fign.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/F-test/Drift/{}-{}.png'.format(i,kk))
    pendiente_y = ['{:.2e}'.format(ii) for ii in pendiente]

    fig, (ax1,ax2,ax_cbar) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]}, figsize=(12,4))
    ax1.get_shared_y_axes().join(ax2)
    g1 = sbn.heatmap(true_positives, annot=True, xticklabels=pendiente_y,
                     yticklabels=delta_var, ax=ax1, vmin=0, vmax=1, cbar=False)
    g2 = sbn.heatmap(false_positives, annot=True, xticklabels=pendiente_y,
                     yticklabels=delta_var, ax=ax2, vmin=0, vmax=1, cbar_ax=ax_cbar)
    ax1.set_title('FDR')
    ax1.set_ylabel('Diferencia en la Varianza a Detectar')
    ax1.set_xlabel('Pendiente del Incremento')
    ax2.set_title('FAR')
    ax2.set_xlabel('Pendiente del Incremento')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    ax2.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.suptitle('Índices de Detección de Falla en Simulación del Nivel del Tanque por Detección de Cambios en la Media'
                 '\ncon F-test (95% Confiabilidad) en Deriva de Resistencia Hidráulica de Válvula de Salida de 0.28 a '
                 '{:.2f}'.format(0.28+i), size=13)
    fig.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/F-test/Drift/{}.png'.format(i))


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

for i in pulse_intensity:
    print(i)
    nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                      tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
    true_positives = np.zeros([len(delta_var), len(N_faults)])
    false_positives = np.zeros([len(delta_var), len(N_faults)])
    for k,kk in zip(delta_var,range(0,len(delta_var))):
        print(k)
        frac_injected_y = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(N_faults,range(0,len(N_faults))):
            print(j)
            nivel_falla, frac_injected, y_test = f.fault_generator(nivel). \
                random_pulse(start=t_i_falla,stop=t_f_falla_pulse,step=paso,N=j,
                             amplitude=i, random_seed=0, mode='random')
            frac_injected_y = np.append(frac_injected_y,[frac_injected])

            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel_falla[tss_2:]).f_test(std=nivel.std(),delta_var=nivel.std()*k, conf_lev=conf_lev, N='auto')
            print(N_auto)
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives[kk, jj] = tp/(tp+fn)
            false_positives[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]

            ax = fign.add_subplot(2,3,jj+1)
            ax.plot(tiempo,nivel_falla,label='Nivel del tanque')
            ax.scatter(tiempo_falla,vector_detected,c='r',marker='o',alpha=0.2,label='Falla detectada')
            plt.title('FMF = {:.2e}'.format(frac_injected))
            plt.xlabel('Tiempo (h)')
            plt.ylabel('Nivel del tanque (m)')
            plt.legend(loc=4)
            if jj > 0 and jj != 3:
                plt.gca().axes.yaxis.set_ticklabels([])
                ax.set_ylabel('')
            if jj < 3:
                plt.gca().axes.xaxis.set_ticklabels([])
                ax.set_xlabel('')
        fign.tight_layout()
        fign.subplots_adjust(top=0.82)
        fign.suptitle('Mediciones del Sensor de Nivel con Pulsos de Amplitud Normalmente Distribuida\nDesviación '
                      'Estándar = {:.2f}, Fracción de Muestras con Fallas FMF y Tamaño de Ventana de Prueba N = {}\n'
                      'Diferencia en la Varianza a detectar = {}'.format(i,N_auto,k), size=13)
        fign.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/F-test/Pulse_4/{}-{}.png'.format(i,kk))

    fig, (ax1,ax2,ax_cbar) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]}, figsize=(12,4))
    ax1.get_shared_y_axes().join(ax2)
    frac_injected_y = ['{:.2e}'.format(a) for a in frac_injected_y]
    g1 = sbn.heatmap(true_positives, annot=True, xticklabels=frac_injected_y,
                     yticklabels=delta_var, ax=ax1, vmin=0, vmax=1, cbar=False)
    g2 = sbn.heatmap(false_positives, annot=True, xticklabels=frac_injected_y,
                     yticklabels=delta_var, ax=ax2, vmin=0, vmax=1, cbar_ax=ax_cbar)
    ax1.set_title('FDR')
    ax1.set_ylabel('Diferencia en la Varianza a Detectar')
    ax1.set_xlabel('Fracción de Muestras con Falla')
    ax2.set_title('FAR')
    ax2.set_xlabel('Fracción de Muestras con Falla')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    ax2.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.suptitle('Índices de Detección de Falla en Simulación del Nivel del Tanque por Detección de Cambios en la Media'
                 '\nF-test (95% Confiabilidad) en Mediciones del Sensor de Nivel con Pulsos de Amplitud Normalmente '
                 'Distribuida (std = {:.2f})'.format(i), size=13)
    fig.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/F-test/Pulse_4/{}.png'.format(i))


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
true_positives = np.zeros([len(delta_var), len(t_f_falla_var)])
false_positives = np.zeros([len(delta_var), len(t_f_falla_var)])
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)

for i,ii in zip(std, var):
    print(i)
    nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                      tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
    for k,kk in zip(delta_var, range(0, len(delta_var))):
        print(k)
        frac_injected_y = np.array([])
        fign = plt.figure(figsize=(10,6))
        for j,jj in zip(t_f_falla_var,range(0,len(t_f_falla_var))):
            print(j)
            nivel_falla, frac_injected, y_test = f.fault_generator(nivel). \
                variance(start=t_i_falla,stop=j,step=paso,stand_dev=i, random_seed=0)
            frac_injected_y = np.append(frac_injected_y,[frac_injected])

            vector_detected, y_pred, nro_fallas, N_auto = f.fault_detector(
                nivel_falla[tss_2:]).f_test(std=nivel.std(),delta_var=nivel.std()*k, conf_lev=conf_lev, N='auto')
            print(N_auto)
            tn, fp, fn, tp = confusion_matrix(y_true=y_test[tss_2:], y_pred=y_pred).ravel()
            true_positives[kk, jj] = tp/(tp+fn)
            false_positives[kk, jj] = fp/(fp+tn)

            tiempo_falla = tiempo[tss_2:][y_pred == 1]

            ax = fign.add_subplot(2,3,jj+1)
            ax.plot(tiempo,nivel_falla,label='Nivel del tanque')
            ax.scatter(tiempo_falla,vector_detected,c='r',marker='o',alpha=0.2,label='Falla detectada')
            plt.title('FMF = {:.2e}'.format(frac_injected))
            plt.xlabel('Tiempo (h)')
            plt.ylabel('Nivel del tanque (m)')
            plt.legend(loc=4)
            if jj > 0 and jj != 3:
                plt.gca().axes.yaxis.set_ticklabels([])
                ax.set_ylabel('')
            if jj < 3:
                plt.gca().axes.xaxis.set_ticklabels([])
                ax.set_xlabel('')
        fign.tight_layout()
        fign.subplots_adjust(top=0.82)
        fign.suptitle('Incremento en la Varianza de las Lecturas del Sensor de Nivel del Tanque\nVarianza = '
                      '{:.2f}, Fracción de Muestras con Falla = FMF y Tamaño de la Ventana de Prueba N = {}\n'
                      'Diferencia en la Varianza a detectar = {}'.format(ii,N_auto,k), size=13)
        fign.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/F-test/Variance_4/{}-{}.png'.format(i,kk))

    fig, (ax1,ax2,ax_cbar) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]}, figsize=(12,4))
    ax1.get_shared_y_axes().join(ax2)
    frac_injected_y = ['{:.2e}'.format(a) for a in frac_injected_y]
    g1 = sbn.heatmap(true_positives, annot=True, xticklabels=frac_injected_y,
                     yticklabels=delta_var, ax=ax1, vmin=0, vmax=1, cbar=False)
    g2 = sbn.heatmap(false_positives, annot=True, xticklabels=frac_injected_y,
                     yticklabels=delta_var, ax=ax2, vmin=0, vmax=1, cbar_ax=ax_cbar)
    ax1.set_title('FDR')
    ax1.set_ylabel('Diferencia en la Varianza a Detectar')
    ax1.set_xlabel('Fracción de Muestras con Falla')
    ax2.set_title('FAR')
    ax2.set_xlabel('Fracción de Muestras con Falla')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    ax2.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.suptitle('Índices de Detección de Falla en Simulación del Nivel del Tanque por Detección de Cambios en la Media'
                 '\nF-test (95% Confiabilidad) en Mediciones del Sensor de Nivel con Incremento de Varianza '
                 '= {:.2f}'.format(ii),size=13)
    fig.savefig('C:/Users/User/Documents/Python/Tesis/Resultados_2/F-test/Variance_4/{}.png'.format(i))
#plt.show()
'''
