from residuos_t_test_F_test import df_tanque_falla_residuos as fallas_tanque
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import datos_tanque
import matplotlib.pyplot as plt

fallas_tanque = fallas_tanque.copy(deep=True)

# Dividimos los datos en inputs y outputs

X = fallas_tanque.residuos.copy(deep=True)
y = fallas_tanque.condicion_falla.copy(deep=True)*1
y = y.astype('int')

X_df = pd.DataFrame(X)

# Vemos la cantidad de datos con y sin falla

class_counts = y.to_frame().groupby('condicion_falla').size()
print('Conteo de datos con y sin falla', class_counts)

# Escalamos los datos y estandarizamos

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X_df = scaler.fit_transform(X_df)

standarizer = StandardScaler().fit(X_df)
standarized_X_df = standarizer.transform(X_df)

# Dividimos los datos en prueba y entrenamiento

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.70, random_state=0)
X_train_rescaled, X_test_rescaled, y_train_rescaled, y_test_rescaled = train_test_split(rescaled_X_df, y,
                                                                                        test_size=0.7, random_state=0)
X_train_standarized, X_test_standarized, y_train_standarized, y_test_standarized = train_test_split(standarized_X_df, y,
                                                                                                    test_size=0.70,
                                                                                                    random_state=0)
# Entrenamos y evaluamos el algoritmo

n_neighbors = 1

model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print('Score sin reescalar', result)
predicted = model.predict(X_test)
matrix = confusion_matrix(y_test, predicted)
print(matrix)

model_rescaled = KNeighborsClassifier(n_neighbors=n_neighbors)
model_rescaled.fit(X_train_rescaled, y_train_rescaled)
result_rescaled = model_rescaled.score(X_test_rescaled, y_test_rescaled)
print('Score reescalado', result_rescaled)
predicted_rescaled = model_rescaled.predict(X_test_rescaled)
matrix_rescaled = confusion_matrix(y_test_rescaled, predicted_rescaled)
print(matrix_rescaled)

model_standarized = KNeighborsClassifier(n_neighbors=n_neighbors)
model_standarized.fit(X_train_standarized, y_train_standarized)
result_standarized = model_standarized.score(X_test_standarized, y_test_standarized)
print('Score estandarizado', result_standarized)
predicted_standarized = model_standarized.predict(X_test_standarized)
matrix_standarized = confusion_matrix(y_test_standarized, predicted_standarized)
print(matrix_standarized)

num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
scoring = 'roc_auc'

result_crossval = cross_val_score(model, X_df, y, cv=kfold, scoring=scoring).mean()
result_rescaled_crossval = cross_val_score(model_rescaled, rescaled_X_df, y, cv=kfold, scoring=scoring).mean()
result_standarized_crossval = cross_val_score(model_standarized, standarized_X_df, y, cv=kfold, scoring=scoring).mean()
print('Accuracy Cros-validado', result_crossval)
print('Accuray Cros-validado reescalado', result_rescaled_crossval)
print('Accuracy Cros-validado estandarizado', result_standarized_crossval)

# Procedemos a realizar la deteccion de fallas para el Nivel

tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

FDR_0 = tp/(tp+fn)
FAR_0 = fp/(fp+tn)

print('FDR y FAR sin escalar', FDR_0, FAR_0)


def shift_and_concat(data, lags, groupby, cols_to_keep):
    grupos = data.groupby(groupby)
    data = data[cols_to_keep]
    data_shifted_final = data
    for i in range(1, lags+1):
        data_shifted = pd.DataFrame()
        for grupo in grupos:
            grupo_shift = grupo[1][cols_to_keep].shift(i)
            grupo_shift.dropna(inplace=True, how='any')
            data_shifted = pd.concat([data_shifted, grupo_shift])
        data_shifted.rename(columns={'nivel': 'nivel_t_-_{}'.format(i), 'condicion_falla': 'cond_shift'}, inplace=True)
        data_shifted_final = pd.merge(data_shifted_final, data_shifted, left_index=True, right_index=True)
        data_shifted_final.drop('cond_shift', axis=1, inplace=True)
    return data_shifted_final

FDR_vector = np.array([])
FAR_vector = np.array([])

n_neighbors = 5
lags = 10

for j in range(1, n_neighbors+1):
    model = KNeighborsClassifier(n_neighbors=j)
    FDR = np.array(FDR_0)
    FAR = np.array(FAR_0)
    for i in range(1, lags+1):
        df = shift_and_concat(fallas_tanque, i, ['tipo_falla', 'caracteristica_1', 'caracteristica_2'], ['nivel', 'condicion_falla'])
        X = df[df.columns.difference(['condicion_falla'])]
        y = df.condicion_falla.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)

        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

        FDR_i = tp/(tp+fn)
        FAR_i = fp/(fp+tn)
        FDR = np.append(FDR, FDR_i)
        FAR = np.append(FAR, FAR_i)

        #print('FDR y FAR Shifted x{}'.format(i), FDR, FAR)

    FDR_vector = np.append(FDR_vector, FDR)
    FAR_vector = np.append(FAR_vector, FAR)

max_FDR = max(enumerate(FDR_vector), key=lambda x: x[1])
FAR_for_max_FDR = FAR_vector[max_FDR[0]]
min_FAR = min(enumerate(FAR_vector), key=lambda x: x[1])
FDR_for_min_FAR = FDR_vector[min_FAR[0]]

FDR_vector = FDR_vector.reshape(n_neighbors, lags+1)
FAR_vector = FAR_vector.reshape(n_neighbors, lags+1)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

for k in range(1, n_neighbors+1):
    ax[0].plot(range(0, lags+1), FDR_vector[k-1], color='#108282', alpha=1-1*k/(n_neighbors+1), label='k = {}'.format(k), linewidth=2)
    ax[1].plot(range(0, lags+1), FAR_vector[k-1], color='#ad190f', alpha=1-1*k/(n_neighbors+1), label='k = {}'.format(k), linewidth=2)
ax[0].legend(loc=3, ncol=n_neighbors, frameon=False, borderaxespad=0.,  mode="expand", bbox_to_anchor=[0, -0.1, 1, 0.5], fontsize=8)
ax[1].legend(loc=2, ncol=n_neighbors, mode="expand", frameon=False, borderaxespad=0., fontsize=8)

font = {'weight': 'bold', 'size': 8}
ax[0].text(([i for i in range(0, lags+1)]*n_neighbors)[max_FDR[0]], max_FDR[1]+0.001,
           '{:.2f}'.format(max_FDR[1]), ha='center', va='bottom', fontdict=font, color='#108282')
ax[0].text(([i for i in range(0, lags+1)]*n_neighbors)[max_FDR[0]], max_FDR[1]-0.001,
           '{:.2f}'.format(FAR_for_max_FDR), ha='center', va='top', fontdict=font, color='#ad190f')
ax[1].text(([i for i in range(0, lags+1)]*n_neighbors)[min_FAR[0]], min_FAR[1],
           '{:.2f}'.format(min_FAR[1]), ha='right', va='top', fontdict=font, color='#ad190f')
ax[1].text(([i for i in range(0, lags+1)]*n_neighbors)[min_FAR[0]], min_FAR[1],
           '{:.2f}'.format(FDR_for_min_FAR), ha='right', va='bottom', fontdict=font, color='#108282')

ax[0].set_xlim([-0.2, lags])
ax[1].set_xlim([-0.2, lags])
ax[0].set_ylim([min(FDR_vector.ravel())-0.01, max(FDR_vector.ravel())+0.01])
ax[1].set_ylim([min(FAR_vector.ravel())-0.005, max(FAR_vector.ravel())+0.005])

ax[0].set_ylabel('FDR')
ax[1].set_ylabel('FAR')
ax[1].set_xlabel('Lags')
fig.suptitle('FDR y FAR Por kNN Con k Vecinos Más Cercanos - Residuos', color='k', alpha=0.8, weight='bold')

ax[0].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False)
ax[1].tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False)

for i in ['right', 'left', 'top', 'bottom']:
    ax[0].spines[i].set_visible(False)
    ax[1].spines[i].set_visible(False)

plt.show()

