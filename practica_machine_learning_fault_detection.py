from simultank_F1_variable import simultank
from aumento_gradual_r import aumento_gradual_r
from vector_caudal import caudal
import fault_1 as f
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

r_inicial = 0.28                        # Valor inicial de la resistencia hidraulica
delta_r = [0.01, 0.04, 0.12]            # Valor final de la resistencia hidraulica (Intensidad del drift)
longitud = 24                           # Tiempo de simulacion [h]
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
t_i_falla = 8
t_f_falla_drift = [8.2, 10, 15, 20, 24]
delta_media = 0.1

r = np.ones(longitud/paso)*r_inicial
Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)
df_X = pd.DataFrame()
df_y = pd.DataFrame()

nivel_non_faulty = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                             tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)

# Generacion de datos de entrenamiento de falla de deriva

for i in delta_r:
    pendiente = np.array([])
    for j,jj in zip(t_f_falla_drift,range(0,len(t_f_falla_drift))):
        r_falla, slope, y_train = f.fault_generator(r).drift(start=t_i_falla,stop=j,step=paso,change=i)
        nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r_falla,caudal_entrada=Q,
                            tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
        d_nivel = [(nivel[k] - nivel[k-1])/paso if k != 0 else 0 for k in range(0,len(nivel))]

        vector_detected, falla_bool, nro_fallas, N_auto = f.fault_detector(
            nivel[tss_2:]).t_test(data_0=nivel_non_faulty[tss_2:], stand_dev=1, conf_lev=conf_lev,
                                  delta_mean=nivel[tss_2:t_i_falla/paso].mean()*delta_media, N='auto')

        df_X = df_X.append(pd.DataFrame({'nivel': nivel[tss_2:], 'd_nivel': d_nivel[tss_2:], 't_2': falla_bool}),
                           ignore_index=True)
        df_y = df_y.append(pd.DataFrame({'fault': y_train[tss_2:]}), ignore_index=True)

# Generacion de datos de entrenamiento de falla de pulso

t_f_falla_pulse = 24
N_faults = [10,50,100]
pulse_intensity = [0.1, 1, 2]
r = np.ones(longitud/paso)*r_inicial

Q = caudal(set_point=q_set,longitud=longitud,paso=paso,factor_ruido=0.01,ruido=True)

nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r,caudal_entrada=Q,
                  tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)

for i in pulse_intensity:
    print(i)
    for j,jj in zip(N_faults,range(0,len(N_faults))):
        print(j)
        nivel_falla, frac_injected, y_train = f.fault_generator(nivel). \
            random_pulse(start=t_i_falla,stop=t_f_falla_pulse,step=paso,N=j,
                         amplitude=i, random_seed=0, mode='random')
        d_nivel = [(nivel[k] - nivel[k-1])/paso if k != 0 else 0 for k in range(0,len(nivel_falla))]

        vector_detected, falla_bool, nro_fallas, N_auto = f.fault_detector(
            nivel_falla[tss_2:]).t_test(data_0=nivel[tss_2:], stand_dev=1, conf_lev=conf_lev,
                                        delta_mean=nivel[tss_2:t_i_falla/paso].mean()*delta_media, N='auto')
        df_X = df_X.append(pd.DataFrame({'nivel': nivel_falla[tss_2:], 'd_nivel': d_nivel[tss_2:], 't_2': falla_bool}),
                           ignore_index=True)
        df_y = df_y.append(pd.DataFrame({'fault': y_train[tss_2:]}), ignore_index=True)

# Generacion de datos de entrenamiento de falla de varianza


## Entrenamos la red ##

# Comprobamos primero el mejor numero de kNN para el score mas alto

clf = KNeighborsClassifier()

param_grid_clf = {'n_neighbors': range(1,11)}
grid_search_clf = GridSearchCV(clf, param_grid_clf, cv=5)
grid_search_clf.fit(df_X, df_y['fault'])

print('Mejor numero de kNN = {}'.format(grid_search_clf.best_params_))
print('Score en base a dicho nro de kNN = {}'.format(grid_search_clf.best_score_))

X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=list(grid_search_clf.best_params_.values())[0])
clf.fit(X_train, y_train)

# Prueba con random pulse (new data)
nivel_falla, frac_injected, y_train_1 = f.fault_generator(nivel). \
    random_pulse(start=t_i_falla,stop=t_f_falla_pulse,step=paso,N=10,
                 amplitude=0.1, random_seed=15, mode='random')
vector_detected, falla_bool, nro_fallas, N_auto = f.fault_detector(
    nivel_falla[tss_2:]).t_test(data_0=nivel[tss_2:], stand_dev=1, conf_lev=conf_lev,
                                delta_mean=nivel[tss_2:t_i_falla/paso].mean()*delta_media, N='auto')
d_nivel = [(nivel[k] - nivel[k-1])/paso if k != 0 else 0 for k in range(0,len(nivel_falla))]

X_test_1 = pd.DataFrame({'nivel': nivel_falla[tss_2:], 'd_nivel': d_nivel[tss_2:], 't_2': falla_bool})
df_y_1 = pd.DataFrame({'fault': y_train_1[tss_2:]})

y_pred_1 = clf.predict(X_test_1)
print(clf.score(X_test_1,df_y_1))
confusion = confusion_matrix(df_y_1, y_pred_1)
print(confusion)

plt.figure()
plt.plot(tiempo, nivel_falla)
plt.scatter(tiempo[tss_2:][y_pred_1 == 1], nivel_falla[tss_2:][y_pred_1 == 1], c='r')
for x in tiempo[y_train_1 == 1]:
    plt.axvline(x, c='r', alpha=0.5, linewidth=1)
plt.show()

# Prueba con drift (new data)
r_falla, slope, y_train = f.fault_generator(r).drift(start=t_i_falla,stop=20,step=paso,change=0.005)
nivel = simultank(area=area,nivel_inicial=nivel_inicial,resist_hidraulica=r_falla,caudal_entrada=Q,
                  tiempo_inicial=0,tiempo_final=longitud,paso=paso,analitic_sol=False)
d_nivel = [(nivel[k] - nivel[k-1])/paso if k != 0 else 0 for k in range(0,len(nivel))]

vector_detected, falla_bool, nro_fallas, N_auto = f.fault_detector(
    nivel[tss_2:]).t_test(data_0=nivel_non_faulty[tss_2:], stand_dev=1, conf_lev=conf_lev,
                          delta_mean=nivel[tss_2:t_i_falla/paso].mean()*delta_media, N='auto')

X_test_2 = pd.DataFrame({'nivel': nivel[tss_2:], 'd_nivel': d_nivel[tss_2:], 't_2': falla_bool})
df_y_2 = pd.DataFrame({'fault': y_train[tss_2:]})
y_pred_2 = clf.predict(X_test_2)
print(clf.score(X_test_2,df_y_2))
confusion = confusion_matrix(df_y_2, y_pred_2)
print(confusion)

plt.figure()
plt.plot(tiempo, nivel)
plt.scatter(tiempo[tss_2:][y_pred_2 == 1], nivel[tss_2:][y_pred_2 == 1], c='r')
plt.show()

#scores = cross_val_score(clf, df_X, df_y['fault'],cv=10)
#print(scores.mean())
