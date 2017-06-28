'''
Dados los parametros de la ecuacion diferencial del tanque (A: Area transversal, R: Resistencia hidraulica de la valvula,
Qi: Caudal de entrada y Ho: Altura inicial del tanque), procederemos a calcular el tiempo de establecimiento al 2%
(tss(2%)) del nivel, dado por:
tss(2%) = 4RA.
Con la ayuda del programa simultank y de la libreria ScyPy, haremos un t-test de 2 muestras. Si el pvalue es menor al
nivel de confianza alfa, rechazaremos la hipotesis de que ambas muestras tienen la misma media y el sistema arrojara un
error.
'''

import numpy as np
from generacion_vector_caudal_cuadrado import vector_caudal_cuadrado
from simultank_F1_variable import simul_tank_caudal_variable
import scipy.stats
import scipy as sp
import math
import pandas as pd

# Generamos el Vector Caudal de prueba con la falla
caudal_falla = vector_caudal_cuadrado(4.5,5,100,1,200)
# Generamos el Vector Caudal sin falla para comparacion con simulador
caudal_sin_falla = np.ones(200)*5

# Parametros
A = 1.5
R = 1.3

tss2 = math.ceil(4*A*R)

H_sin_falla = simul_tank_caudal_variable(A,0,R,caudal_sin_falla,0,200,1)
H_con_falla = simul_tank_caudal_variable(A,0,R,caudal_falla,0,200,1)

H_sin_falla_serie = pd.Series(H_sin_falla[tss2:])
H_con_falla_serie = pd.Series(H_con_falla[tss2:])

df = pd.concat([H_sin_falla_serie,H_con_falla_serie],axis=1)
df.rename(columns={0: 'Nivel sin falla',1:'Nivel con falla'},inplace=True)
pd.options.display.max_rows = 200

# Calculo de la media acumulativa, estimacion de la media de forma recursiva y el error asociado entre ambos valores

df['Cumulative Mean'] = df['Nivel con falla'].mean()
df['Estimated Mean'] = df['Cumulative Mean']
for i in df.index:
    df['Cumulative Mean'].iloc[i] = df['Nivel con falla'].iloc[:i].mean()
    if i<2:
        df['Estimated Mean'].iloc[i] = df['Nivel con falla'].iloc[1]
    elif i>=2:
        df['Estimated Mean'].iloc[i] = df['Estimated Mean'].iloc[i-1] + \
                                       (1/i)*(df['Nivel con falla'].iloc[i] - df['Estimated Mean'].iloc[i-1])
df['error'] = df['Cumulative Mean'] - df['Estimated Mean']

# Deteccion de falla

from fault_detector_ttest import fault_detector_ttest

print(fault_detector_ttest(df['Nivel con falla'].iloc[tss2:], df['Nivel con falla'].std(),0.95,df['Nivel sin falla'].mean()*0.03))

