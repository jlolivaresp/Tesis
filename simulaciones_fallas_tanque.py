"""
Simulaciones de distintas fallas en un sistema dinamico compuesto por un tanque con caudal de entrada Q, area
transversal A y resistencia hidraulica de la valvula de salida R.
"""

from simultank_F1_variable import simul_tank_caudal_variable
from aumento_gradual_r import aumento_gradual_r
from vector_caudal import caudal
from fault_detector_ttest import fault_detector_ttest
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


'''
Obstruccion gradual en la valvula de salida (Aumento gradual de la resistencia hidraulica de la valvula)
'''

# Parametros de las funciones a utilizar

r_inicial = 0.28           # Valor inicial de la resistencia hidraulica
r_final = 0.32          # Valor final de la resistencia hidraulica
longitud = 8760*2         # Tiempo de simulacion [h]
t_inicial_falla = 1000   # Tiempo en que inicial la falla [h]
t_final_falla = 8760*2     # Tiempo en que se estabiliza la falla [h]
paso = 1                # Paso de integracion [h]
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
nivel = simul_tank_caudal_variable(A=area,Ho=0,R=r_falla_gradual,F1=caudal_q,ti=t_inicial,tf=t_final,paso=paso)
df_obstruccion_gradual_valvula_salida = pd.DataFrame({'res_hidraulica': r_falla_gradual,
                                                      'caudal_entrada': caudal_q,'nivel_tanque': nivel},
                                                     index=range(t_inicial,t_final,paso))
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
plt.show()