"""Este programa genera dos graficas del nivel de agua en un tanque en funcion del tiempo, una para un caudal que
cambia como una onda cuadrada entre dos valores de caudal Qmin y Qmx, y otra grafica para un caudal que cambia como
una onda triangular que oscina entre Qmin y Qmax"""

'''Importamos alguna librerias y funciones que nos seran utiles'''

import numpy as np                                                # Estas libreria sera util para trabajar con arreglos
import matplotlib.pyplot as plt                                   # Estas libreria sera util para graficar
from generacion_vector_caudal_cuadrado import vector_caudal_cuadrado # Funcion que genera vector de caudal cuadrado
from generacion_vector_caudal_triangular import vector_caudal_triagular # Funcion que genera vector de caudal triangular
from simultank_F1_variable import simul_tank_caudal_variable        # Funcion que resuelve la ecuacion diferencial
from matplotlib.widgets import Slider

'''Predefinimos algunas variables que seran usadas como parametros de entrada en nuestras funciones creadas'''

F1min = 1       # Caudal minimo
F1max = 2       # Caudal maximo
periodo = 10    # Periodo de oscilacion
paso = 1      # Paso de discretizacion
tiempo = 200   # Intervalo de tiempo sobre el cual se quiere graficar

'''Graficamos'''

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.35)
F1min_0 = 1
F1max_0 = 2
R_0 = 1.3
A_0 = 1
H0_0 = 1
tiempo_0 = 200
paso_0 = .1
periodo_0 = 10
RH_sin_falla = np.ones(tiempo_0/paso_0)*1.3

# Generamos el vector de tiempos
t = np.arange(0.0, tiempo_0, paso_0)

[line] = ax.plot(t, simul_tank_caudal_variable(A_0, H0_0, RH_sin_falla, vector_caudal_triagular(F1min_0, F1max_0, periodo_0,
                                                                                       paso_0, tiempo_0), 0.0, tiempo_0,
                                               paso_0), linewidth = 2)
ax.set_xlim([0, tiempo_0])
ax.set_ylim([-20, 40])

axis_color = 'lightgoldenrodyellow'

'''Aplicamos los Sliders'''

resistencia_slider_ax  = fig.add_axes([0.12, 0.05, 0.325, 0.03])
resistencia_slider = Slider(resistencia_slider_ax, 'Valv. Resist.', 0.01, 3, valinit=R_0)

caudal_minimo_slider_ax = fig.add_axes([0.12, 0.1, 0.325, 0.03])
caudal_minimo_slider = Slider(caudal_minimo_slider_ax, 'Caudal Minimo', 0.1, 10.0, valinit=F1min_0)

caudal_maximo_slider_ax = fig.add_axes([0.12, 0.15, 0.325, 0.03])
caudal_maximo_slider = Slider(caudal_maximo_slider_ax, 'Caudal Maximo', 0.1, 10.0, valinit=F1max_0)

periodo_slider_ax = fig.add_axes([0.6, 0.05, 0.325, 0.03])
periodo_slider = Slider(periodo_slider_ax, 'Periodo', 5, 200.0, valinit=periodo_0)

area_tanque_slider_ax  = fig.add_axes([0.6, 0.1, 0.325, 0.03])
area_tanque_slider = Slider(area_tanque_slider_ax, 'Area del Tanque', 0.5, 10.0, valinit=A_0)

altura0_slider_ax  = fig.add_axes([0.6, 0.15, 0.325, 0.03])
altura0_slider = Slider(altura0_slider_ax, 'Nivel inicial', 0.0, 10.0, valinit=H0_0)

def sliders_on_changed(val):
    F1_triangular = vector_caudal_triagular(caudal_maximo_slider.val,
                            caudal_minimo_slider.val,
                            periodo_slider.val, paso_0,
                            tiempo_0)
    RH_sin_falla = np.ones(tiempo_0/paso_0)*resistencia_slider.val
    line.set_ydata(simul_tank_caudal_variable(area_tanque_slider.val, altura0_slider.val, RH_sin_falla,
                                              F1_triangular, 0, tiempo_0, paso_0))
    fig.canvas.draw_idle()
caudal_minimo_slider.on_changed(sliders_on_changed)
caudal_maximo_slider.on_changed(sliders_on_changed)
periodo_slider.on_changed(sliders_on_changed)
resistencia_slider.on_changed(sliders_on_changed)
altura0_slider.on_changed(sliders_on_changed)
area_tanque_slider.on_changed(sliders_on_changed)

plt.show()