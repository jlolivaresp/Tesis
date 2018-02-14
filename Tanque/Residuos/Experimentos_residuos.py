import numpy as np
import matplotlib.pyplot as plt
from simulador_nivel_tanque import simultank
import datos_tanque as datos
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd

Q = np.ones(len(datos.t_sim))*datos.q
cont = 0
for i in enumerate(Q):
    cont += 1
    if cont == 15:
        Q[i[0]:i[0]+15] += np.random.uniform(-1, 1)
        cont = 0

nivel = simultank(area=datos.area, nivel_inicial=datos.nivel_inicial, resist_hidraulica=datos.r_sim,
                  caudal_entrada=Q, tiempo_inicial=datos.t_i, tiempo_final=datos.t_f,
                  paso=datos.paso, analitic_sol=False)

ts = pd.Series(nivel)
df = pd.DataFrame(ts, columns=["test"])
df.index = pd.Index(pd.date_range("2011/01/01", periods = len(nivel), freq = 'Q'))

model = ARIMA(df, order=[1, 0, 1], exog=Q)
model_fit = model.fit(trend='nc', disp=0)

nivel_2 = simultank(area=datos.area, nivel_inicial=datos.nivel_inicial, resist_hidraulica=datos.r_sim,
                    caudal_entrada=datos.q_sim+np.random.normal(0,0.1,len(datos.q_sim)), tiempo_inicial=datos.t_i,
                    tiempo_final=datos.t_f,
                    paso=datos.paso, analitic_sol=False)

#nivel_pred_2 = model_fit.predict(start=1, end=len(Q), exog=datos.q_sim+np.random.normal(0,0.1,len(datos.q_sim)))

ysim = np.zeros(len(datos.q_sim))
for i in range(1,len(datos.q_sim)):

    ysim[i] = model_fit.params[1]*nivel_2[i-1] + model_fit.params[0]*datos.q_sim[i-1]

print(model_fit.params)
error = nivel_2-ysim
print(np.sqrt(error.dot(error)))
plt.figure()
plt.plot(datos.t_sim, nivel_2)
plt.plot(datos.t_sim, ysim, 'r', alpha=0.5)
plt.show()
