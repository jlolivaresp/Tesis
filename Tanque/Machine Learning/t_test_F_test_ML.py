import fallas_tanque
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd

array = fallas_tanque.df_tanque_falla.copy(deep=True)

X = array.nivel.copy(deep=True)
y = array.condicion_falla.copy(deep=True)*1
y = y.astype('int')

X_df = pd.DataFrame(X)


# Vemos la cantidad de datos con y sin falla
'''
class_counts = y.to_frame().groupby('condicion_falla').size()
print(class_counts)
'''

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.5)
print(X_train)
model = KNeighborsClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print(result)
predicted = model.predict(X_test)
matrix = confusion_matrix(y_test, predicted)
print(matrix)
'''
# Probaremos con shifts

grupos = array.groupby(['tipo_falla', 'caracteristica_1', 'caracteristica_2'])
X_df = pd.DataFrame()
for grupo in grupos:
    grupo_shift = grupo[1].nivel.shift(1)
    grupo_shift.dropna(inplace=True, how='any')
    X_df = pd.concat([X_df, grupo_shift])
X_df['nivel'] = array.nivel
X_df.rename(columns={0: 'nivel_t_+_1', 'nivel': 'nivel'}, inplace=True)

print(X_df.join(y.to_frame()))

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.5)
print(X_train)
model = KNeighborsClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print(result)
predicted = model.predict(X_test)
matrix = confusion_matrix(y_test, predicted)
print(matrix)
'''
