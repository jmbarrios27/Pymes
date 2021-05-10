# Data Manipulation.
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
import random
import numpy as np
import time

# Data Visualization.
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Data process and metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.feature_selection import f_regression

# Linear Regression
from sklearn.linear_model import LinearRegression

# Random forest Regressor.
from sklearn.ensemble import RandomForestRegressor

# Model Save
import pickle

# statistics
import statsmodels.formula.api as sm
import statsmodels.formula.api as smf

# ANN
import keras

from statsmodels.stats.stattools import durbin_watson


# Reading data from github repository.
url_base_entrenamiento = 'https://raw.githubusercontent.com/jmbarrios27/Pymes/main/base_entrenamiento.csv'
url_base_prueba = 'https://raw.githubusercontent.com/jmbarrios27/Pymes/main/base_prueba.csv'

# Converting into pandas dataframe.
train = pd.read_csv(url_base_entrenamiento, sep=',')
test = pd.read_csv(url_base_prueba, sep=',')


print(train.head())

# revisando tipo de datos y viendo valores nulos por variable.
print(train.info())


# Descripción de datos.
print(train.describe())

# revisando columnas con NAn Values
print(train.isna().sum())

# Score de riesgo
print(train['buro_score_apc'].unique())

# Vamos a observar los detalles de la columna finc_sva
print(train.finc_sva.describe())

# medida de rentabilidad del banco generada por el cliente.
print(train['finc_sva'].unique())


# Remplazando Valores NaN con promedio de columnas.
train['buro_score_apc'] = train.buro_score_apc.fillna(train.buro_score_apc.mean())
train['finc_sva'] = train.finc_sva.fillna(train.finc_sva.mean())

# Quitando valores NaN
train = train.dropna()

# Revisando Forma de datos
print(train.shape)

# Veamos la columna adming_flag_gerenciado
train.admin_flag_gerenciado.value_counts()

# observamos que el único valor para esta columna es 0. Por lo que realmente no agregar ningún valor a los datos. Vamos a eliminarlo.

# Eliminando columna admin_flag, ya que de esto no depende ningún salario.
train = train.drop(columns=['admin_flag_gerenciado'])


plt.figure(figsize=(12, 10))
sns.distplot(train['dem_salario'], color='darkred')
plt.title('Distribución de salarios', fontsize=15)
plt.grid()
plt.show()

# Observando correlacion
corr = train.corr()
corr.style.background_gradient(cmap='coolwarm')

# relación entre salario y variable de activos y pasivos del cliente.
sns.scatterplot(data=train, x='dem_salario', y='admin_antiguedad_banco')
plt.title('RELACIÓN POSITIVA ENTRE SALARIO Y SUMA DE ACTIVOS Y PASIVOS DEL CLIENTE')
plt.grid()
plt.show()

# Relación entre salario y variable de transacciones de atm por el cliente.
sns.scatterplot(data=train, x='dem_salario', y='finc_tamano_comercial')
plt.title('RELACIÓN NEGATIVA ENTRE SALARIO Y TRANSACCIONES REALIZADAS EN ATM')
plt.grid()
plt.show()

# Variables con correlación mayor.
data = train[
    ['admin_antiguedad_banco', 'buro_score_apc', 'comp_perc_atm', 'comp_perc_bpi', 'comp_score_digital', 'comp_txn_bpi',
     'comp_usd_bpi_prom', 'comp_usd_pos_prom', 'dem_edad', 'dem_planilla', 'finc_bal_act'
        , 'finc_bal_pas', 'finc_tamano_comercial', 'pdcto_flag_auto', 'pdcto_flag_tiene_tdd', 'dem_salario']]

# Valores que tiennen una pequeña correlación ya sea positiva o negativa.
corr_data = data.corr()
corr_data.style.background_gradient(cmap='coolwarm')

# Observando correlación entre variable de salario y predictoras.
sns.pairplot(data)
plt.show()

# Borrando variables que poseen valores binarios o discretizados ya que no aportan un comportamiento lineal positivo o negativo.
data = data.drop(columns=['comp_score_digital', 'pdcto_flag_tiene_tdd', 'pdcto_flag_auto', 'dem_planilla'])

# Train _test Split
X = data.drop(columns=['dem_salario'])
y = data['dem_salario']

# Max Min Scaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Modelo de Regresión lineal con variables con mayor correlación (aunque baja).

def linear_regression(X_train, X_test, y_train, y_test):
    # Modelo
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    pred = lm.predict(X_test)

    # Metricas
    print('MAPE', mean_absolute_percentage_error(y_true=y_test, y_pred=pred) * 100, '%')
    print()

    # Dataframe
    prediction = lm.predict(X_test)
    prediction = pd.DataFrame(prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_l = pd.DataFrame(y_test)
    y_test_l = y_test_l.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    dataframe = pd.concat([y_test_l, prediction], axis=1)

    print(dataframe.head(15))

    return lm


# Modelo de Regresión linear
lm = linear_regression(X_train, X_test, y_train, y_test)
print(lm)


# ANN
def ann(X_train, X_test, y_train, y_test):
    model = keras.Sequential()
    model.add(keras.layers.Dense(11, input_shape=(11,)))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(keras.optimizers.Adam(lr=0.1), 'mean_absolute_percentage_error')

    # Compilar modelo
    history = model.fit(X_train, y_train, batch_size=8, epochs=7)

    # prediction
    ann_prediction = model.predict(X_test)
    ann_prediction = pd.DataFrame(ann_prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_ann = pd.DataFrame(y_test)
    y_test_ann = y_test_ann.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    ypred_ann = pd.concat([y_test_ann, ann_prediction], axis=1)

    # Resultado de ANN.
    y_true_ann = ypred_ann['dem_salario']
    y_pred_ann = ypred_ann['salario_estimado']

    print('MAPE:', mean_absolute_percentage_error(y_true=y_true_ann, y_pred=y_pred_ann) * 100, '%')
    print()
    print('VALORES ACTUALES VS PREDICCIONES ')
    print(ypred_ann.head(15))

    return model


# llamando función de ANN
ann_model = ann(X_train, X_test, y_train, y_test)
print(ann_model)


def random_forest_regressor(X_train, X_test, y_train, y_test):
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    pred = rfr.predict(X_test)

    # Metricas
    print('MAPE', mean_absolute_percentage_error(y_true=y_test, y_pred=pred) * 100, '%')
    print()

    # Dataframe
    rfr_prediction = rfr.predict(X_test)
    rfr_prediction = pd.DataFrame(rfr_prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_rfr = pd.DataFrame(y_test)
    y_test_rfr = y_test_rfr.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    dataframe = pd.concat([y_test_rfr, rfr_prediction], axis=1)

    print(dataframe.head(15))

    return rfr


# LLamando a la función
rfr = random_forest_regressor(X_train, X_test, y_train, y_test)
print(rfr)


# Salarios
print(train.dem_salario.value_counts())


def continous_filter(df, low_exclusive=2, high_inclusive=15):
    """
    Función que retorna las columnas que tienen valores menores o iguales a las categorias
    """
    list_of_features = []
    for i in df.columns:
        if low_exclusive == high_inclusive:
            if df[i].nunique() <= low_exclusive:
                list_of_features.append(i)
        else:
            if df[i].nunique() <= high_inclusive and df[i].nunique() > low_exclusive:
                list_of_features.append(i)
    return list_of_features


# elegimos 3 porque no queremos variables binarias ni la de score digital ya que no aportan en nada al salario.
remainder_cols = continous_filter(train, 3, len(train))
print('# Variables continuas con más de 3 atributos) = ', len(remainder_cols))

n_rows = len(train)

# Atributos que tengan más de 3 clases
train_df_cols = train[remainder_cols]

# Extrayendo los atributos más importantes.
target_df = train['dem_salario']
# target_df_log = np.log(target_df)
f, p_val = f_regression(train_df_cols, target_df)

# Extrayendo Valores que tienen el p-valor menor de 0.05 para hacerlo estadisticamente significativos.
f_reg_df = pd.DataFrame(np.array([f, p_val]).T, index=train_df_cols.columns, columns=['f-statistic', 'p-value'])
binary_stored_features = f_reg_df[f_reg_df['p-value'] < 0.05].sort_values(by='f-statistic', ascending=False)
print(binary_stored_features)

# construyendo dataframe con los valores estadisticamente sifnicativos.
p_train = train[['admin_antiguedad_banco', 'buro_score_apc', 'finc_bal_pas', 'comp_perc_atm', 'finc_tamano_comercial',
                 'comp_usd_bpi_prom', 'dem_edad', 'comp_perc_bpi', 'finc_bal_act', 'comp_usd_pos_prom', 'comp_txn_bpi',
                 'finc_perc_pas_tc', 'comp_txn_pos', 'finc_sva', 'comp_usd_suc_prom', 'comp_txn_atm', 'comp_txn_suc',
                 'finc_perc_act_tc', 'dem_salario']]

# Train _test Split
X_p_train = p_train.drop(columns=['dem_salario'])
y_p_train = p_train['dem_salario']

# Max Min Scaler
sc = MinMaxScaler()
X_p_train = sc.fit_transform(X_p_train)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_p_train, y_p_train, test_size=0.33, random_state=42)


# Modelo de Regresión lineal con valores estadisticamente sifnicantes.
def linear_regression_p(X_train, X_test, y_train, y_test):
    # Modelo
    lmp = LinearRegression()
    lmp.fit(X_train, y_train)
    pred = lmp.predict(X_test)

    # Metricas
    print('MAPE', mean_absolute_percentage_error(y_true=y_test, y_pred=pred) * 100, '%')
    print()

    # Dataframe
    prediction = lmp.predict(X_test)
    prediction = pd.DataFrame(prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_p = pd.DataFrame(y_test)
    y_test_p = y_test_p.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    dataframe = pd.concat([y_test_p, prediction], axis=1)

    print(dataframe.head(15))

    return lm


# Modelo de Regresión linear
lmp = linear_regression_p(X_train, X_test, y_train, y_test)
print(lmp)

# ANN
def ann(X_train, X_test, y_train, y_test):
    model = keras.Sequential()
    model.add(keras.layers.Dense(11, input_shape=(18,)))
    model.add(keras.layers.Dense(1, activation='relu'))
    model.compile(keras.optimizers.Adam(lr=0.1), 'mean_absolute_percentage_error')

    # Compilar modelo
    history = model.fit(X_train, y_train, batch_size=8, epochs=7)

    # prediction
    ann_prediction = model.predict(X_test)
    ann_prediction = pd.DataFrame(ann_prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_ann = pd.DataFrame(y_test)
    y_test_ann = y_test_ann.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    ypred_ann = pd.concat([y_test_ann, ann_prediction], axis=1)

    # Resultado de ANN.
    y_true_ann = ypred_ann['dem_salario']
    y_pred_ann = ypred_ann['salario_estimado']

    print('MAPE:', mean_absolute_percentage_error(y_true=y_true_ann, y_pred=y_pred_ann) * 100, '%')
    print()
    print('VALORES ACTUALES VS PREDICCIONES ')
    print(ypred_ann.head(15))

    return model


# llamando función de ANN
ann_model_p = ann(X_train, X_test, y_train, y_test)
print(ann_model_p)


# RFF con variables p-value < 0.05
def random_forest_regressor(X_train, X_test, y_train, y_test):
    rfr_r = RandomForestRegressor()
    rfr_r.fit(X_train, y_train)
    pred = rfr_r.predict(X_test)

    # Metricas
    print('MAPE', mean_absolute_percentage_error(y_true=y_test, y_pred=pred) * 100, '%')
    print()

    # Dataframe
    rfr_prediction = rfr_r.predict(X_test)
    rfr_prediction = pd.DataFrame(rfr_prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_rfr = pd.DataFrame(y_test)
    y_test_rfr = y_test_rfr.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    dataframe = pd.concat([y_test_rfr, rfr_prediction], axis=1)

    print(dataframe.head(15))

    return rfr_r


# LLamando a la función
rfr_r = random_forest_regressor(X_train, X_test, y_train, y_test)
print(rfr_r)

# Train _test Split
X = train.drop(columns=['dem_salario'])
y = train['dem_salario']

# Max Min Scaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def linear_regression(X_train, X_test, y_train, y_test):
    # Modelo
    lm_all = LinearRegression()
    lm_all.fit(X_train, y_train)
    pred = lm_all.predict(X_test)

    # Metricas
    print('MAPE', mean_absolute_percentage_error(y_true=y_test, y_pred=pred) * 100, '%')
    print()

    # Dataframe
    prediction = lm_all.predict(X_test)
    prediction = pd.DataFrame(prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_l = pd.DataFrame(y_test)
    y_test_l = y_test_l.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    dataframe = pd.concat([y_test_l, prediction], axis=1)

    print(dataframe.head(15))

    return lm_all


# Llammando a regresion lineal con todos los datos
lm_all = linear_regression(X_train, X_test, y_train, y_test)
print(lm_all)


# ANN
def ann(X_train, X_test, y_train, y_test):
    model_all = keras.Sequential()
    model_all.add(keras.layers.Dense(11, input_shape=(47,)))
    model_all.add(keras.layers.Dense(1, activation='relu'))
    model_all.compile(keras.optimizers.Adam(lr=0.1), 'mean_absolute_percentage_error')

    # Compilar modelo
    history = model_all.fit(X_train, y_train, batch_size=8, epochs=7)

    # prediction
    ann_prediction = model_all.predict(X_test)
    ann_prediction = pd.DataFrame(ann_prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_ann = pd.DataFrame(y_test)
    y_test_ann = y_test_ann.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    ypred_ann = pd.concat([y_test_ann, ann_prediction], axis=1)

    # Resultado de ANN.
    y_true_ann = ypred_ann['dem_salario']
    y_pred_ann = ypred_ann['salario_estimado']

    print('MAPE:', mean_absolute_percentage_error(y_true=y_true_ann, y_pred=y_pred_ann) * 100, '%')
    print()
    print('VALORES ACTUALES VS PREDICCIONES ')
    print(ypred_ann.head(15))

    return model_all


# Modelo con todas las variables.
model_all = ann(X_train, X_test, y_train, y_test)
print(model_all)


def random_forest_regressor(X_train, X_test, y_train, y_test):
    rfr_f = RandomForestRegressor()
    rfr_f.fit(X_train, y_train)
    pred = rfr_f.predict(X_test)

    # Metricas
    print('MAPE', mean_absolute_percentage_error(y_true=y_test, y_pred=pred) * 100, '%')
    print()

    # Dataframe
    rfr_prediction = rfr_f.predict(X_test)
    rfr_prediction = pd.DataFrame(rfr_prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_rfr = pd.DataFrame(y_test)
    y_test_rfr = y_test_rfr.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    dataframe = pd.concat([y_test_rfr, rfr_prediction], axis=1)

    print(dataframe.head(15))

    return rfr_f


# LLamando a la función
rfr_f = random_forest_regressor(X_train, X_test, y_train, y_test)
print(rfr_f)


# MODELOS INTUITIVOS

intuitivo = train[['finc_bal_pas', 'finc_bal_act', 'finc_tamano_comercial','dem_salario']]

# Train _test Split
X_intuitivo = intuitivo.drop(columns=['dem_salario'])
y_intuitivo = intuitivo['dem_salario']

# Max Min Scaler
sc = MinMaxScaler()
X_intuitivo = sc.fit_transform(X_intuitivo)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_intuitivo, y_intuitivo, test_size=0.33, random_state=42)


def linear_regression(X_train, X_test, y_train, y_test):
    # Modelo
    lm_in = LinearRegression()
    lm_in.fit(X_train, y_train)
    pred = lm_in.predict(X_test)

    # Metricas
    print('MAPE', mean_absolute_percentage_error(y_true=y_test, y_pred=pred) * 100, '%')
    print()

    # Durbin-watson
    residual = (y_test - pred)
    print('Durbin-watson:', durbin_watson(residual))

    # Residuos
    sns.distplot(residual)
    plt.title('Distribución Normal de los residuos')
    plt.show()

    # Predicted Values
    plt.scatter(y_test, pred, s=10, color='r')
    plt.title('predicted values')
    plt.show()

    # Dataframe
    prediction = lm_in.predict(X_test)
    prediction = pd.DataFrame(prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_l = pd.DataFrame(y_test)
    y_test_l = y_test_l.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    dataframe = pd.concat([y_test_l, prediction], axis=1)

    print(dataframe.head(15))

    return lm_in


# Llammando a regresion lineal con todos los datos
lm_in = linear_regression(X_train, X_test, y_train, y_test)
print(lm_in)


# ANN
def ann(X_train, X_test, y_train, y_test):
    model_in = keras.Sequential()
    model_in.add(keras.layers.Dense(3, input_shape=(3,)))
    model_in.add(keras.layers.Dense(1, activation='relu'))
    model_in.compile(keras.optimizers.Adam(lr=0.1), 'mean_absolute_percentage_error')

    # Compilar modelo
    history = model_in.fit(X_train, y_train, batch_size=8, epochs=7)

    # prediction
    ann_prediction = model_in.predict(X_test)
    ann_prediction = pd.DataFrame(ann_prediction, columns=['salario_estimado'])

    # Valores actuales.
    y_test_ann = pd.DataFrame(y_test)
    y_test_ann = y_test_ann.reset_index(drop=True)

    # Dataframe predicciones y actuales.
    ypred_ann = pd.concat([y_test_ann, ann_prediction], axis=1)

    # Resultado de ANN.
    y_true_ann = ypred_ann['dem_salario']
    y_pred_ann = ypred_ann['salario_estimado']

    print('MAPE:', mean_absolute_percentage_error(y_true=y_true_ann, y_pred=y_pred_ann) * 100, '%')
    print()
    print('VALORES ACTUALES VS PREDICCIONES ')
    print(ypred_ann.head(15))

    return model_in


# Modelo con todas las variables.
model_in = ann(X_train, X_test, y_train, y_test)
print(model_in)


# Creando Test para predecir con variables
test_data = test[
    ['admin_antiguedad_banco', 'buro_score_apc', 'comp_perc_atm', 'comp_perc_bpi', 'comp_txn_bpi', 'comp_usd_bpi_prom',
     'comp_usd_pos_prom', 'dem_edad', 'finc_bal_act'
        , 'finc_bal_pas', 'finc_tamano_comercial']]

# Guardadno Key del cliente.
costumer_key = test[['llave_cod_cliente']]

# Replacing NaN Values with mean
test_data['buro_score_apc'] = test_data.buro_score_apc.fillna(test_data.buro_score_apc.mean())
test_data['comp_txn_bpi'] = test_data.comp_txn_bpi.fillna(test_data.comp_txn_bpi.mean())
test_data['comp_usd_bpi_prom'] = test_data.comp_usd_bpi_prom.fillna(test_data.comp_usd_bpi_prom.mean())
test_data['comp_usd_pos_prom'] = test_data.comp_usd_pos_prom.fillna(test_data.comp_usd_pos_prom.mean())

# Normalizando los valores de las columnas elegidas para mejorar el accuracy del modelo.
test_data = sc.fit_transform(test_data)

# Creando data alternativa para trabajar.
df = test_data

# Aplicando modelo de regresión lineal.
test_predictions = lm.predict(df)

# Resultado de predicciones de salarios.
salarios = pd.DataFrame(test_predictions)
salarios.columns = ['dem_salario']

# Archivo final
base_prueba_evaluado = costumer_key.join(salarios)


# Vista previa de archivo evaluado. LR
print(base_prueba_evaluado)

# A CSV de jupyter Notebook a maquína local.
base_prueba_evaluado.to_csv(r'C:\\Users\\Asus\\Desktop\\BANISTMO\\base_prueba_evaluado.csv', index=False, sep=';')


