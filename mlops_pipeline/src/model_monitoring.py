"""
model_monitoring.py
Monitoreo de data drift para el modelo de riesgo crediticio (PIM5, Avance 3).

Enfoque: como el dataset es una foto fija (no hay datos llegando en vivo), se
usa la columna `mes_prestamo` (creada en ft_engineering.py) como proxy real de
tiempo. Se compara, mes a mes, la distribución de los clientes de ese mes
(dentro del set de test) contra la distribución de referencia: el set de
train que efectivamente usó el modelo para aprender.

Métricas de drift calculadas por variable y por mes (las 4 pedidas en la
consigna del Avance 3):
- KS test (Kolmogorov-Smirnov): variables numéricas.
- PSI (Population Stability Index): variables numéricas.
- Jensen-Shannon divergence: numéricas y categóricas (sobre distribuciones
  discretizadas/de frecuencias).
- Chi-cuadrado: variables categóricas.

Nota: en un proyecto real normalmente se elige una sola métrica por tipo de
variable (PSI es el estándar más común en riesgo crediticio). Acá se calculan
las 4 porque la consigna las pide explícitamente, no porque sea lo más
eficiente.

Nota sobre el criterio de alerta (comparaciones múltiples): se evalúan 12
meses x 20 variables = 240 tests de hipótesis, cada uno al 5% de
significancia. Si una variable no tiene ningún drift real, cada uno de sus
12 tests mensuales igual tiene ~5% de probabilidad de dispararse solo por
azar, y con 12 oportunidades la probabilidad de que se dispare al menos una
vez por puro azar es 1 - 0.95^12 ≈ 46%. Por eso NO alcanza con que una sola
métrica supere su umbral para marcar alerta: se exige que coincidan al
menos 2 de las 4 métricas, lo que reduce drásticamente los falsos positivos
que surgen de testear tantas combinaciones. Además, como el split train/test
es aleatorio (no se particiona por mes), cada "mes" del test es en
realidad una submuestra al azar de la misma población total: no hay drift
temporal real posible en este dataset, por lo que cualquier alerta que
sobreviva este criterio más estricto sigue siendo ruido estadístico y no
evidencia de un corrimiento real — el valor del ejercicio es la metodología,
no el hallazgo.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split

from ft_engineering import (
    cargar_y_limpiar, columnas_numericas, columnas_numericas_cero,
    columnas_nominales, columnas_ordinales
)

# Umbrales de alerta (criterios estándar de la industria para cada métrica).
UMBRAL_PSI_ALERTA = 0.2   # PSI > 0.2 = drift significativo
UMBRAL_KS_PVALUE = 0.05   # p-value < 0.05 = distribuciones distintas
# KS_STAT: el p-value del KS test se vuelve hiper-sensible cuando las dos
# muestras tienen tamaños muy distintos (acá: train ~8600 filas vs cada mes
# del test entre 43 y 406) -> detecta como "significativa" cualquier
# diferencia mínima e irrelevante. Por eso el KS exige ADEMÁS que el
# estadístico (la distancia máxima entre las dos distribuciones, de 0 a 1)
# sea grande, no solo que el p-value sea bajo: significancia estadística y
# relevancia práctica a la vez.
UMBRAL_KS_STAT = 0.15
UMBRAL_CHI2_PVALUE = 0.05
UMBRAL_JS = 0.1           # Jensen-Shannon > 0.1 = divergencia relevante


def cargar_referencia_y_actual(path_csv, test_size=0.2, random_state=42):
    """
    Replica el mismo split que ft_engineering.generar_datasets (mismo
    random_state=42), pero devuelve los datos SIN codificar (antes del
    ColumnTransformer): las métricas de drift se calculan sobre los valores
    originales de cada variable, no sobre la matriz ya transformada.
    """
    df = cargar_y_limpiar(path_csv)
    X = df.drop(columns=['Pago_atiempo'])
    y = df['Pago_atiempo']
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test


def calcular_psi(referencia, actual, bins=10):
    """
    Population Stability Index para una variable numérica. Divide la
    referencia en `bins` franjas de igual frecuencia (cuantiles) y compara
    qué proporción de cada muestra cae en cada franja.
    """
    referencia = referencia.dropna()
    actual = actual.dropna()
    if len(referencia) == 0 or len(actual) == 0:
        return np.nan

    cortes = np.quantile(referencia, np.linspace(0, 1, bins + 1))
    cortes[0] = -np.inf
    cortes[-1] = np.inf
    cortes = np.unique(cortes)
    if len(cortes) < 3:
        # La variable tiene muy poca variabilidad para discretizar en bins.
        return np.nan

    ref_counts = pd.cut(referencia, bins=cortes).value_counts(sort=False) / len(referencia)
    act_counts = pd.cut(actual, bins=cortes).value_counts(sort=False) / len(actual)

    # Evita log(0) / división por 0 cuando una franja queda vacía.
    ref_counts = ref_counts.replace(0, 0.0001)
    act_counts = act_counts.replace(0, 0.0001)

    psi = np.sum((act_counts - ref_counts) * np.log(act_counts / ref_counts))
    return psi


def calcular_js_numerica(referencia, actual, bins=10):
    """Jensen-Shannon divergence para una variable numérica, vía histogramas."""
    referencia = referencia.dropna()
    actual = actual.dropna()
    if len(referencia) == 0 or len(actual) == 0:
        return np.nan

    minimo = min(referencia.min(), actual.min())
    maximo = max(referencia.max(), actual.max())
    if minimo == maximo:
        return np.nan
    cortes = np.linspace(minimo, maximo, bins + 1)

    ref_hist, _ = np.histogram(referencia, bins=cortes)
    act_hist, _ = np.histogram(actual, bins=cortes)

    if ref_hist.sum() == 0 or act_hist.sum() == 0:
        return np.nan

    ref_dist = ref_hist / ref_hist.sum()
    act_dist = act_hist / act_hist.sum()

    return jensenshannon(ref_dist, act_dist)


def calcular_js_categorica(referencia, actual):
    """Jensen-Shannon divergence para una variable categórica, sobre frecuencias relativas."""
    categorias = sorted(set(referencia.dropna().unique()) | set(actual.dropna().unique()))
    if not categorias:
        return np.nan
    ref_freq = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0)
    act_freq = actual.value_counts(normalize=True).reindex(categorias, fill_value=0)
    return jensenshannon(ref_freq.values, act_freq.values)


def evaluar_drift_variable_numerica(referencia, actual):
    """Calcula KS test, PSI y Jensen-Shannon para una variable numérica."""
    referencia_limpia = referencia.dropna()
    actual_limpia = actual.dropna()

    if len(referencia_limpia) < 2 or len(actual_limpia) < 2:
        return {"ks_stat": np.nan, "ks_pvalue": np.nan, "psi": np.nan, "js_divergence": np.nan}

    ks_stat, ks_pvalue = stats.ks_2samp(referencia_limpia, actual_limpia)
    psi = calcular_psi(referencia, actual)
    js = calcular_js_numerica(referencia, actual)

    return {"ks_stat": ks_stat, "ks_pvalue": ks_pvalue, "psi": psi, "js_divergence": js}


def evaluar_drift_variable_categorica(referencia, actual):
    """Calcula chi-cuadrado y Jensen-Shannon para una variable categórica."""
    categorias = sorted(set(referencia.dropna().unique()) | set(actual.dropna().unique()))
    ref_counts = referencia.value_counts().reindex(categorias, fill_value=0)
    act_counts = actual.value_counts().reindex(categorias, fill_value=0)

    tabla_contingencia = pd.DataFrame({"referencia": ref_counts, "actual": act_counts})
    try:
        chi2_stat, chi2_pvalue, _, _ = stats.chi2_contingency(tabla_contingencia)
    except ValueError:
        # Puede pasar si alguna categoría queda con conteo 0 en ambas columnas.
        chi2_stat, chi2_pvalue = np.nan, np.nan

    js = calcular_js_categorica(referencia, actual)

    return {"chi2_stat": chi2_stat, "chi2_pvalue": chi2_pvalue, "js_divergence": js}


MINIMO_METRICAS_EN_ACUERDO = 2


def hay_alerta(metricas):
    """
    Determina si una fila de métricas constituye una alerta de drift.

    No alcanza con que UNA sola métrica supere su umbral: con 240 tests de
    hipótesis en juego (12 meses x 20 variables), exigir una sola coincidencia
    dispara falsos positivos en casi todas las variables solo por azar (ver
    nota de "comparaciones múltiples" en el docstring del módulo). Por eso se
    exige que al menos MINIMO_METRICAS_EN_ACUERDO de las 4 métricas disponibles
    coincidan en señalar drift para esa variable en ese mes.
    """
    señales = 0
    if pd.notna(metricas.get("psi")) and metricas["psi"] > UMBRAL_PSI_ALERTA:
        señales += 1
    # KS: p-value bajo (significancia estadística) Y estadístico alto
    # (relevancia práctica) a la vez -> ver nota de UMBRAL_KS_STAT más arriba.
    if (pd.notna(metricas.get("ks_pvalue")) and pd.notna(metricas.get("ks_stat"))
            and metricas["ks_pvalue"] < UMBRAL_KS_PVALUE
            and metricas["ks_stat"] > UMBRAL_KS_STAT):
        señales += 1
    if pd.notna(metricas.get("chi2_pvalue")) and metricas["chi2_pvalue"] < UMBRAL_CHI2_PVALUE:
        señales += 1
    if pd.notna(metricas.get("js_divergence")) and metricas["js_divergence"] > UMBRAL_JS:
        señales += 1
    return señales >= MINIMO_METRICAS_EN_ACUERDO


def generar_reporte_drift(path_csv="../../Base_de_datos.csv"):
    """
    Función principal del job de monitoreo: calcula el drift mes a mes
    (mes_prestamo) para todas las variables, comparando cada mes del set de
    test contra la distribución de referencia (train).

    Devuelve un DataFrame "largo": una fila por cada combinación
    (mes_prestamo, variable, métricas de esa variable, alerta).
    """
    X_train, X_test = cargar_referencia_y_actual(path_csv)

    variables_numericas = columnas_numericas + columnas_numericas_cero
    # mes_prestamo se usa para definir los períodos, no tiene sentido medirle
    # drift a sí misma -> se excluye de las categóricas a evaluar.
    variables_categoricas = [
        v for v in (columnas_nominales + columnas_ordinales) if v != 'mes_prestamo'
    ]

    filas = []
    meses = sorted(X_test['mes_prestamo'].dropna().unique())

    for mes in meses:
        actual_mes = X_test[X_test['mes_prestamo'] == mes]

        for variable in variables_numericas:
            metricas = evaluar_drift_variable_numerica(X_train[variable], actual_mes[variable])
            filas.append({
                "mes_prestamo": mes, "variable": variable, "tipo": "numerica",
                **metricas, "alerta": hay_alerta(metricas)
            })

        for variable in variables_categoricas:
            metricas = evaluar_drift_variable_categorica(X_train[variable], actual_mes[variable])
            filas.append({
                "mes_prestamo": mes, "variable": variable, "tipo": "categorica",
                **metricas, "alerta": hay_alerta(metricas)
            })

    return pd.DataFrame(filas)


if __name__ == "__main__":
    reporte = generar_reporte_drift()

    carpeta_resultados = "resultados"
    os.makedirs(carpeta_resultados, exist_ok=True)
    reporte.to_csv(os.path.join(carpeta_resultados, "drift_metrics.csv"), index=False)

    variables_con_alerta = sorted(reporte[reporte['alerta']]['variable'].unique().tolist())

    print(f"Reporte de drift generado: {len(reporte)} filas "
          f"({reporte['mes_prestamo'].nunique()} meses x "
          f"{reporte['variable'].nunique()} variables)")
    print(f"Variables con alerta en al menos un mes: {variables_con_alerta}")
    print("Guardado en resultados/drift_metrics.csv")
