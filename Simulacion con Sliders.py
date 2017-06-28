"""
Este programa genera una grafica del nivel de un tanque en funcion del tiempo. La misma se mantiene abierta en
tiempo real y permite controlar, mediante sliders, los parametros: Caudal de entrada, Altura inicial del tanque,
factor de resistencia hidraulica de valvula y el Area transversal del tanque.
"""

import numpy as np                      # Estas libreria sera util para trabajar con arreglos
import matplotlib.pyplot as plt         # Estas libreria sera util para graficar
from matplotlib.widgets import Slider   # Estas libreria sera util para anadir los Sliders a la grafica

'''
Para ello, creamos una funcion llamada Simultank que resuelve la ecuacion diferencial que describe el comportamiento
de la altura del tanque en funcion del tiempo.

Los parametros de entrada de nuestra funcion son:

A: Area transversal del tanque
Ho: Altura inicial del contenido del tanque
R: Factor de descarga
F1: Caudal de entrada
ti: Tiempo inicial desde el cual se quiere simular el comportamiento
tf: Tiempo final desde el cual se quiere simular el comportamiento
paso: El incremento de tiempo entre ti y tf

Los parametros de salida son:

Grafica de altura H vs tiempo t
'''
def simul_tank(A, Ho, R, F1, ti, tf, paso):         # Creamos nuestra funcion con los parametros de entrada

    vector_tiempo = np.arange(ti, tf, paso)         # Generamos el vector de intervalos de tiempo
    vector_H = np.array([Ho])                       # Creamos el vector de altura que estara en funcion del tiempo
    vector_K1 = np.array([0])                       # K1 y K2 son los parametros del metodo RK2
    vector_K2 = np.array([0])

    iteraciones = len(vector_tiempo)
    contador = 0

    while contador < iteraciones - 1:               # Este ciclo ira iterando para cada valor de tiempo

        ''' Aplicamos RK2
        '''
        vector_K1_mas_1 = paso*((F1/A) - (Ho /(A*R)))
        vector_K2_mas_1 = paso*((F1/A) - (Ho + vector_H[contador] + vector_K1[contador])/(A*R))
        H_mas_1 = vector_H[contador] + 0.5*(vector_K1[contador] + vector_K2[contador])

        contador += 1

        vector_H = np.append(vector_H, H_mas_1)     # El vector de altura va creciendo para cada t
        vector_K1 = np.append(vector_K1,vector_K1_mas_1 )
        vector_K2 = np.append(vector_K2, vector_K2_mas_1)

    return vector_H

fig = plt.figure()

# Generamos la grafica

ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.35)
t = np.arange(0.0, 200, 0.05)
F1_0 = 1
R_0 = 1
A_0 = 1
H0_0 = 1
[line] = ax.plot(t, simul_tank(A_0, H0_0, R_0, F1_0, 0.0, 200, 0.05), linewidth = 2)
ax.set_xlim([0, 200])
ax.set_ylim([-20, 20])

axis_color = 'lightgoldenrodyellow'

# Agregamos los 4 Sliders
caudal_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
caudal_slider = Slider(caudal_slider_ax, 'Caudal', 0.1, 10.0, valinit=F1_0)

resistencia_slider_ax  = fig.add_axes([0.25, 0.1, 0.65, 0.03])
resistencia_slider = Slider(resistencia_slider_ax, 'Valv. Resist.', 0.01, 3, valinit=R_0)

area_tanque_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03])
area_tanque_slider = Slider(area_tanque_slider_ax, 'Area del Tanque', 0.5, 10.0, valinit=A_0)

altura0_slider_ax  = fig.add_axes([0.25, 0.2, 0.65, 0.03])
altura0_slider = Slider(altura0_slider_ax, 'Nivel inicial', 0.0, 10.0, valinit=H0_0)

def sliders_on_changed(val):
    line.set_ydata(simul_tank(area_tanque_slider.val, altura0_slider.val, resistencia_slider.val, caudal_slider.val, 0, 200, 0.05))
    fig.canvas.draw_idle()
caudal_slider.on_changed(sliders_on_changed)
resistencia_slider.on_changed(sliders_on_changed)
area_tanque_slider.on_changed(sliders_on_changed)
altura0_slider.on_changed(sliders_on_changed)

plt.show()


