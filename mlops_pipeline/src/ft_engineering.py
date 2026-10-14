"""
ft_engineering.py
Ingeniería de características para el modelo de riesgo crediticio (PIM5).

Clasificación de columnas:
- Numéricas (mediana): capital_prestado, plazo_meses, edad_cliente, salario_cliente,
  total_otros_prestamos, cuota_pactada, puntaje_datacredito,
  cant_creditosvigentes, huella_consulta, saldo_total, saldo_principal,
  creditos_sectorFinanciero, creditos_sectorCooperativo, creditos_sectorReal,
  promedio_ingresos_datacredito
- Numéricas (fillna 0): saldo_mora, saldo_mora_codeudor
- Categóricas nominales (OneHotEncoder): tipo_credito, tipo_laboral, mes_prestamo
- Categóricas ordinales (OrdinalEncoder): tendencia_ingresos
  (orden: Decreciente < Estable < Creciente)
- Descartada: fecha_prestamo (reemplazada por mes_prestamo)
- Descartada: puntaje — data leakage confirmado en Avance 2: separa a los dos
  grupos de Pago_atiempo sin ningún solapamiento (grupo 0 hasta 62.67, grupo 1
  desde 63.81), algo que ningún score de crédito real produce. Ver README
  para el detalle de la verificación.
- Target: Pago_atiempo

Nota: promedio_ingresos_datacredito y tendencia_ingresos se trataron con muestreo
aleatorio en el EDA (Avance 1). Acá se simplifica a mediana / moda estándar
(ver README para el detalle de esta decisión).
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def cargar_y_limpiar(path_csv):
    df = pd.read_csv(path_csv)

    df = df.drop(columns=['puntaje'])  # data leakage: ver nota en el docstring

    valores_validos = ['Estable', 'Creciente', 'Decreciente']
    df.loc[~df['tendencia_ingresos'].isin(valores_validos), 'tendencia_ingresos'] = np.nan

    df['mes_prestamo'] = pd.to_datetime(df['fecha_prestamo']).dt.month
    df = df.drop(columns=['fecha_prestamo'])

    df['tipo_credito'] = df['tipo_credito'].replace(68, np.nan)
    df['edad_cliente'] = df['edad_cliente'].replace([121, 122, 123], np.nan)
    df.loc[df['salario_cliente'] > 100_000_000, 'salario_cliente'] = np.nan
    df.loc[df['total_otros_prestamos'] > 100_000_000, 'total_otros_prestamos'] = np.nan
    df['cant_creditosvigentes'] = df['cant_creditosvigentes'].replace(62, np.nan)
    df.loc[df['puntaje_datacredito'] == -7, 'puntaje_datacredito'] = np.nan

    return df


columnas_numericas = [
    'capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente',
    'total_otros_prestamos', 'cuota_pactada', 'puntaje_datacredito',
    'cant_creditosvigentes', 'huella_consulta', 'saldo_total',
    'saldo_principal', 'creditos_sectorFinanciero',
    'creditos_sectorCooperativo', 'creditos_sectorReal', 'promedio_ingresos_datacredito'
]

columnas_numericas_cero = ['saldo_mora', 'saldo_mora_codeudor']

columnas_nominales = ['tipo_credito', 'tipo_laboral', 'mes_prestamo']

columnas_ordinales = ['tendencia_ingresos']
orden_tendencia = ['Decreciente', 'Estable', 'Creciente']

pipeline_numerico = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

pipeline_numerico_cero = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])

pipeline_nominal = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

pipeline_ordinal = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(categories=[orden_tendencia]))
])

preprocesador = ColumnTransformer([
    ('num', pipeline_numerico, columnas_numericas),
    ('num_cero', pipeline_numerico_cero, columnas_numericas_cero),
    ('nom', pipeline_nominal, columnas_nominales),
    ('ord', pipeline_ordinal, columnas_ordinales)
])


def generar_datasets(path_csv, test_size=0.2, random_state=42):
    """
    Carga, limpia y transforma el dataset. Devuelve train/test ya procesados.
    """
    df = cargar_y_limpiar(path_csv)

    X = df.drop(columns=['Pago_atiempo'])
    y = df['Pago_atiempo']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_transformado = preprocesador.fit_transform(X_train)
    X_test_transformado = preprocesador.transform(X_test)

    return X_train_transformado, X_test_transformado, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generar_datasets("../../Base_de_datos.csv")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
