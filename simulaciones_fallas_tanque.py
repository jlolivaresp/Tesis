"""
Simulaciones de distintas fallas en un sistema dinamico compuesto por un tanque con caudal de entrada Q, area
transversal A y resistencia hidraulica de la valvula de salida R.
"""

from simultank_F1_variable import simultank
from aumento_gradual_r import aumento_gradual_r
from vector_caudal import caudal
from fault_detector_ttest import fault_detector_ttest
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from timeit import default_timer as timer

'''
Prueba de funciones simultank, vector_caudal y aumento_gradual_r
'''
#paso = 0.02
# vector_caudal
#Qi_ruido_0_01 = caudal(set_point=24, longitud=8800, paso=paso, factor_ruido=0.01, ruido=True)
#Qi_ruido_0_005 = caudal(set_point=24, longitud=8800, paso=paso, factor_ruido=0.001, ruido=True)
#Qi_no_ruido = caudal(set_point=24, longitud=8800, paso=paso, ruido=False)
#plt.figure(figsize=(9, 3))
#plt.plot(np.arange(0,8800,paso),Qi_ruido_0_01,label='Con ruido (0,01)')
#plt.plot(np.arange(0,8800,paso),Qi_ruido_0_005,label='Con ruido (0,005)')
#plt.plot(np.arange(0,8800,paso),Qi_no_ruido,label='Sin ruido')
#plt.title('Caudal vs Tiempo')
#plt.ylabel('Caudal (m^3/h)')
#plt.xlabel('Tiempo (h)')
#plt.legend()
#plt.tight_layout()


# aumento_gradual_r
#r_1 = aumento_gradual_r(r_inicial=0.28, r_final=0.32, longitud=8800, t_inicial=1000, t_final=5000, paso=paso)
#r_2 = aumento_gradual_r(r_inicial=0.28, r_final=0.32, longitud=8800, t_inicial=2000, t_final=7000, paso=paso)
#r_3 = aumento_gradual_r(r_inicial=0.28, r_final=0.32, longitud=8800, t_inicial=3000, t_final=3200, paso=paso)
#plt.figure(figsize=(9, 3))
#plt.plot(np.arange(0,8800,paso),r_1,label='falla 1 (gradual)')
#plt.plot(np.arange(0,8800,paso),r_2,label='falla 2 (gradual)')
#plt.plot(np.arange(0,8800,paso),r_3,label='falla 3 (repentina)')
#plt.title('Resistencia Hidraulica vs Tiempo')
#plt.ylabel('Resistencia Hidraulica')
#plt.xlabel('Tiempo (h)')
#plt.legend()
#plt.tight_layout()


# simultank
#nivel_1 = simultank(area=3, nivel_inicial=0, resist_hidraulica=r_1, caudal_entrada=Qi_ruido_0_01,
 #                   tiempo_inicial=0, tiempo_final=8800, paso=paso)
#nivel_2 = simultank(area=3, nivel_inicial=0, resist_hidraulica=r_2, caudal_entrada=Qi_ruido_0_005,
 #                   tiempo_inicial=0, tiempo_final=8800, paso=paso)
#nivel_3 = simultank(area=3, nivel_inicial=0, resist_hidraulica=r_3, caudal_entrada=Qi_no_ruido,
 #                   tiempo_inicial=0, tiempo_final=8800, paso=paso)
#plt.figure(figsize=(9, 3))
#plt.plot(np.arange(0,8800,paso),nivel_1,label='Nivel: Con ruido (0,01) - falla 1 (gradual)')
#plt.plot(np.arange(0,8800,paso),nivel_2,label='Nivel: Con ruido (0,005) - falla 2 (gradual)')
#plt.plot(np.arange(0,8800,paso),nivel_3,label='Nivel: Sin ruido - falla 3 (repentina)')
#plt.title('Nivel vs Tiempo')
#plt.ylabel('Nivel (m)')
#plt.xlabel('Tiempo (h)')
#plt.legend()
#plt.tight_layout()
#plt.show()

'''
plt.figure(figsize=(9, 5))
for paso in [0.005, 0.01, 0.025, 0.05, 0.1]:
    start = timer()
    Qi_ruido_0_01 = caudal(set_point=40, longitud=8800, paso=paso, factor_ruido=0.01, ruido=True)
    r_1 = aumento_gradual_r(r_inicial=0.28, r_final=0.32, longitud=8800, t_inicial=1000, t_final=5000, paso=paso)
    nivel_1 = simultank(area=1.75, nivel_inicial=0, resist_hidraulica=r_1, caudal_entrada=Qi_ruido_0_01,
                        tiempo_inicial=0, tiempo_final=8800, paso=paso)
    end = timer()
    tiempo = end - start
    plt.plot(np.arange(0,8800,paso),nivel_1,
             label='Ruido: 0,01 - falla gradual - paso = {} - tiempo = {}s'.format(paso,tiempo))
plt.title('Nivel vs Tiempo')
plt.ylabel('Nivel (m)')
plt.xlabel('Tiempo (h)')
plt.legend()
plt.tight_layout()
plt.show()'''

'''
Obstruccion gradual en la valvula de salida (Aumento gradual de la resistencia hidraulica de la valvula)
'''
'''
# Parametros de las funciones a utilizar

r_inicial = 0.28           # Valor inicial de la resistencia hidraulica
r_final = 0.32          # Valor final de la resistencia hidraulica
longitud = 8760*2         # Tiempo de simulacion [h]
t_inicial_falla = 1000   # Tiempo en que inicial la falla [h]
t_final_falla = 8760*2     # Tiempo en que se estabiliza la falla [h]
paso = 0.1                # Paso de integracion [h]
q_set = 24            # Caudal de entrada [m^3/h]
ruido = True            # Ruido en el caudal de entrada
area = 3              # Area transversal del tanque [m^2]
nivel_inicial = 0       # Nivel inicial del tanque [m]
t_inicial = 0           # Tiempo inciial de la simulacion [h]
t_final = longitud      # Tiempo final de la simulacion [h]
conf_lev = 0.99         # Nivel de confiabilidad


r_falla_gradual = aumento_gradual_r(r_inicial=r_inicial,r_final=r_final,longitud=longitud,t_inicial=t_inicial_falla,
                                    t_final=t_final_falla,paso=paso)
caudal_q = caudal(set_point=q_set,longitud=longitud,paso=paso,ruido=ruido)
nivel = simultank(area=area, nivel_inicial=0, resist_hidraulica=r_falla_gradual, caudal_entrada=caudal_q,
                  tiempo_inicial=t_inicial, tiempo_final=t_final, paso=paso)
df_obstruccion_gradual_valvula_salida = pd.DataFrame({'res_hidraulica': r_falla_gradual,
                                                      'caudal_entrada': caudal_q,'nivel_tanque': nivel},
                                                     index=np.arange(t_inicial,t_final,paso))
rango_falla_media = fault_detector_ttest(df_obstruccion_gradual_valvula_salida['nivel_tanque'],
                                         df_obstruccion_gradual_valvula_salida['nivel_tanque'].std(),
                                         conf_lev=conf_lev,
                                         delta_mean=df_obstruccion_gradual_valvula_salida['nivel_tanque'].mean()*0.005)


gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
df_obstruccion_gradual_valvula_salida['nivel_tanque'].plot(ax=ax1)
df_obstruccion_gradual_valvula_salida['caudal_entrada'].plot(ax=ax1)
ax1.axvspan(rango_falla_media[0],rango_falla_media[1],fc='r',alpha=0.2,label='Rango de falla (media)')
ax1.vlines(t_inicial_falla,0, max(df_obstruccion_gradual_valvula_salida['nivel_tanque']),
           colors='r',label='Inicio de falla')
ax2.vlines(t_inicial_falla,0, max(df_obstruccion_gradual_valvula_salida['res_hidraulica']),
           colors='r',label='Inicio de falla')
df_obstruccion_gradual_valvula_salida['res_hidraulica'].plot(ax=ax2,c='g')
ax2.set_ylim([df_obstruccion_gradual_valvula_salida['res_hidraulica'].min()-
              0.1*df_obstruccion_gradual_valvula_salida['res_hidraulica'].min(),
              df_obstruccion_gradual_valvula_salida['res_hidraulica'].max()+
              0.1*df_obstruccion_gradual_valvula_salida['res_hidraulica'].max()])
plt.xlabel('Tiempo (h)')
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.show()'''
