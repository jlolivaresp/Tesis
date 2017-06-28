import numpy as np
from simultank_F1_variable import simul_tank_caudal_variable
import math
import pandas as pd
from fault_detector_ttest import fault_detector_ttest
import matplotlib.pyplot as plt
from generacion_vector_caudal_triangular import vector_caudal_triangular
from fault_detector_F import fault_detector_F

''' En este programa se detectara una falla en un sistema simple compuesto por un tanque con un caudal de entrada y
y uno de salida en el que se simulara una variacion en el coeficiente de resistencia hidraulico de la valvula. Dicha
deteccion se hara por metodos estadisticos de t-test para cambios en la media y de F-test para cambios en la varianza'''

# Definimos algunas variables de interes
    # Constantes

t_o = 0                 # Tiempo inicial de la simulacion
t_f = 20000              # Tiempo final de la simulacion
paso = 1                # Paso de integracion para la resolucion de la ecuacion diferencial que rige el nivel del tanque
A = 1.3                 # Area transversal del tanque
h_o = 0                 # Altura inicial del tanque
conf_lev = 0.95         # Nivel de confiabilidad para la deteccion mediate t-test y F-test
delta_mean = 1          # Variacion en la media que queremos detectar
delta_var = 1           # Variacion en la varianza que queremos detectar
q_min = 4               # Caudal de entrada minimo
q_max = 4.05             # Caudal de entrada maximo
T_q = 10                # Periodo de oscilacion/variacion del caudal
long_q = (t_f-t_o)/paso # Longitud del vector caudal
r_h_min = 1.1           # Resistencia hidraulica minima
r_h_max = 1.15           # Resistencia hidraulica maxima
t_o_falla = 11000         # Tiempo inicial de falla
t_f_falla = 18000        # Tiempo final de falla
tss_2 = math.ceil(      # Tiempo de establecimiento
    4*A*r_h_min)

    # Vectores

rango_tiempo = np.arange(t_o,t_f,paso)                          # Rango de tiempo de simulacion
q = vector_caudal_triangular(q_min,q_max,T_q,paso,long_q)       # Vector de Caudales
r_h_falla = np.ones((t_f-t_o)/paso)*r_h_min                     # Vector de Resistencias hidraulicas con falla
r_h_falla[t_o_falla:t_f_falla] = np.arange(r_h_min,r_h_max,(r_h_max-r_h_min)/((t_f_falla-t_o_falla)/paso))
r_h_falla[t_f_falla:] = np.ones(t_f-t_f_falla)*r_h_max
nivel = simul_tank_caudal_variable(A=A,Ho=h_o,R=r_h_falla,F1=q,ti=t_o,tf=t_f,paso=paso)

# Creamos un Data Frame con las varaibles que tenemos

df = pd.DataFrame({'res_hidraulica': r_h_falla,'caudal_entrada': q,'nivel_tanque': nivel}, index=rango_tiempo)

# Primero detectaremos el cambio en la media mediante la funcion fault_detector_ttest()

rango_falla_media = fault_detector_ttest(df['nivel_tanque'].iloc[tss_2:],
                           df['nivel_tanque'].iloc[tss_2:].std(),
                                         conf_lev=conf_lev,delta_mean=df['nivel_tanque'].mean()*0.01)

# Ahora detectamos el cambio en la varianza con la funcion fault_detector_F

#rango_falla_var = fault_detector_F(df['nivel_tanque'].iloc[tss_2:],
 #                                  delta_var=df['nivel_tanque'].iloc[tss_2:].std()*0.5,conf_lev=conf_lev)

ax1 = df.plot()
plt.xlabel('Tiempo (s)')
ax2 = plt.axvspan(rango_falla_media[0],rango_falla_media[1],fc='r',alpha=0.4,label='Rango de falla (media)')
ax3 = plt.vlines(t_o_falla,0,max(df['nivel_tanque']),colors='r',label='Inicio de falla')
#ax4 = plt.axvspan(rango_falla_var[0],rango_falla_var[1],facecolor='b',alpha=0.5,label='Rango de falla (varianza)')
ax1.legend(loc='best')
plt.show()
