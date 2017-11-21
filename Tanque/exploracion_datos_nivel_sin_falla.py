from simulador_nivel_tanque import simultank
import datos_tanque as datos
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''Simulamos el nivel del tanque sin falla'''

nivel_sin_falla = simultank(area=datos.area, nivel_inicial=datos.nivel_inicial, resist_hidraulica=datos.r_sim,
                            caudal_entrada=datos.q_sim, tiempo_inicial=datos.t_i, tiempo_final=datos.t_f,
                            paso=datos.paso, analitic_sol=False)

# Anadimos ruido Gaussiano a la senal

np.random.seed(0)
nivel_sin_falla += np.random.normal(0, 0.02, len(datos.t_sim))

# Desechamos los valores menores al tiempo de establecimiento al 2% para considerar unicamente estado estacionario

nivel_sin_falla_ee = nivel_sin_falla[datos.tss_2/datos.paso:]

'''Presentamos los datos en un DataFrame'''

# Construimos un DataFrame con los datos del nivel

df_tanque = pd.DataFrame(data={'nivel': nivel_sin_falla_ee, 'tiempo': datos.t_sim_ee}, columns=['tiempo', 'nivel'])

# Agregamos una columna de derivadas de primer y segundo orden sobre los datos del nivel
# Esto tendra sentido para el analisis con Machine learning

df_tanque['derivada'] = [(nivel_sin_falla_ee[k] - nivel_sin_falla_ee[k-1])/datos.paso if k != 0 else 0 for k in
                         range(0, len(nivel_sin_falla_ee))]
df_tanque['derivada_2'] = [(nivel_sin_falla_ee[k] - 2*nivel_sin_falla_ee[k-1] + nivel_sin_falla_ee[k-1])/datos.paso
                           if k > 1 else None for k in range(0, len(nivel_sin_falla_ee))]

# Eliminamos las dos primeras filas del DataFrame porque carecen de derivada de segundo orden

df_tanque.drop([0, 1], inplace=True)

'''Analisis de los datos'''

# Primeros 5 datos del DataFrame
print(df_tanque.head())

# Descripcion estadistica de los datos del DataFrame
print(df_tanque.describe())

'''
# Grafica del nivel del tanque en funcion del tiempo
plt.figure()
plt.title('Nivel del Tanque vs Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Nivel (m)')
plt.plot(t_sim, nivel_sin_falla, linewidth=1)

# Grafica del las densidades de probabilidad de los valores del DataFrame
fig, ax = df_tanque.plot(kind='density', subplots=True, layout=(2, 2), sharex=False, title='Densidad de Probabilidades')

plt.show()
'''