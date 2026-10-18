# Credit Risk MLOps

Proyecto integrador (PIM5) de Soy Henry — predicción de riesgo crediticio para una financiera, con el
objetivo de anticipar el comportamiento de pago de nuevos usuarios de crédito. La variable objetivo
(hipótesis a confirmar) es `Pago_atiempo`.

## Estructura del repositorio

```
credit-risk-mlops/
├── mlops_pipeline/
│   └── src/
│       ├── Cargar_datos.ipynb
│       ├── comprension_eda.ipynb
│       ├── ft_engineering.py
│       ├── model_training_evaluation.py
│       ├── model_deploy.py
│       ├── model_monitoring.py
│       └── app_monitoring.py
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
└── readme.md
```

## Cómo levantar el entorno

1. Clonar el repositorio.
2. Crear un entorno virtual en la raíz del proyecto:
   ```
   python -m venv venv
   ```
3. Activar el entorno (Windows):
   ```
   venv\Scripts\activate
   ```
4. Instalar las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Estado actual del proyecto (Avance 1)

- Estructura de carpetas y ramas (`developer`, `certification`, `main`) creadas según la consigna.
- `Cargar_datos.ipynb`: carga del dataset real y verificación inicial (tipos de dato, nulos a alto nivel).
- `comprension_eda.ipynb`: exploración de tipos de variable, tratamiento de nulos (7 columnas),
  verificación de duplicados (sin duplicados) y tratamiento de outliers en varias columnas.
- Entorno virtual y `requirements.txt` generados.

## Avance 2 — Ingeniería de características y modelado

### `ft_engineering.py`

Clasificación de columnas usada en el `ColumnTransformer`:

- **Numéricas (mediana):** capital_prestado, plazo_meses, edad_cliente, salario_cliente,
  total_otros_prestamos, cuota_pactada, puntaje_datacredito, cant_creditosvigentes,
  huella_consulta, saldo_total, saldo_principal, creditos_sectorFinanciero,
  creditos_sectorCooperativo, creditos_sectorReal, promedio_ingresos_datacredito
- **Numéricas (fillna 0):** saldo_mora, saldo_mora_codeudor
- **Categóricas nominales (OneHotEncoder):** tipo_credito, tipo_laboral, mes_prestamo
- **Categóricas ordinales (OrdinalEncoder):** tendencia_ingresos
  (orden: Decreciente < Estable < Creciente)
- **Descartada:** fecha_prestamo (reemplazada por mes_prestamo, para evitar tratar los
  meses como una escala numérica lineal cuando en realidad son cíclicos)
- **Descartada:** puntaje (ver nota de data leakage más abajo)

Nota: `promedio_ingresos_datacredito` y `tendencia_ingresos` se trataron con muestreo
aleatorio en el EDA (Avance 1). Acá se simplificó a mediana / moda estándar, para
mantener el pipeline reproducible con `ColumnTransformer`.

### Selección del modelo

El dataset presenta un fuerte desbalance de clases (~95% de los clientes paga a
tiempo), por lo que la accuracy no es una métrica confiable: un modelo que prediga
siempre la clase mayoritaria ya alcanzaría ~95% sin aprender nada. La métrica de
referencia es el **ROC-AUC**, que no depende de la proporción de clases.

| Modelo | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.953 | 0.976 | **0.639** |
| Random Forest | 0.953 | 0.976 | 0.629 |
| Decision Tree | 0.909 | 0.952 | 0.528 |

**Modelo seleccionado: Logistic Regression.** Tiene el ROC-AUC más alto (empatado
prácticamente con Random Forest) y, al ser un modelo lineal, es más simple e
interpretable — una ventaja relevante en un caso de riesgo crediticio, donde
explicar por qué se aprueba o rechaza un crédito es tan importante como la métrica
en sí.

Resultados completos en `mlops_pipeline/src/resultados/` (`tabla_resumen_modelos.csv`,
`comparacion_modelos.png` y `curva_precision_recall.png`), generados por
`model_training_evaluation.py`. La curva Precision-Recall complementa a `precision`
y `recall` de la tabla (que son un único punto, con el umbral de decisión por
defecto, 0.5): muestra el trade-off completo entre ambas métricas al mover ese
umbral, útil en un caso de riesgo crediticio donde se podría ajustar el umbral
según cuánto pesa, para el negocio, rechazar un buen cliente vs. aprobar uno malo.

### Nota sobre data leakage detectado

Durante la evaluación inicial, los tres modelos dieron métricas perfectas
(ROC-AUC = 1.0), lo cual no es plausible en un problema real. Se identificó que
la variable `puntaje` separaba completamente las dos clases del target sin ningún
solapamiento (grupo "no pagó a tiempo" hasta 62.67, grupo "sí pagó" desde 63.81) —
un patrón imposible en un score de crédito genuino, y típico de una variable
generada en el dataset a partir del propio resultado. Se descartó `puntaje` de las
features (`ft_engineering.py`) y se reentrenaron los modelos; las métricas de la
tabla de arriba corresponden a esa versión corregida.

## Avance 3 — Monitoreo de data drift

### Caso de negocio

Un modelo de riesgo crediticio se entrena una vez, pero sigue prediciendo sobre
clientes nuevos indefinidamente. Si el perfil de esos clientes nuevos se corre
respecto al de entrenamiento (otro rango de edad, salarios distintos, cambios en
las condiciones económicas), el modelo queda desactualizado sin que nadie lo note
hasta que la mora real empieza a subir. El monitoreo de **data drift** existe para
detectar ese corrimiento de forma proactiva, antes de que impacte el negocio.

### Cómo se simuló la periodicidad

El dataset es una foto fija (no hay datos llegando en vivo), así que no hay forma
de medir drift real a lo largo del tiempo. Se usó la columna `mes_prestamo`
(creada en `ft_engineering.py`) como proxy temporal: se compara, mes a mes, la
distribución de los clientes de ese mes dentro del **set de test** contra la
distribución del **set de train** (la población de referencia que efectivamente
usó el modelo para aprender), con el mismo `random_state=42` del Avance 2 para
que ambos splits coincidan.

### Métricas calculadas (`model_monitoring.py`)

Por cada combinación de mes y variable, las cuatro métricas pedidas en la
consigna:

- **KS test (Kolmogorov-Smirnov):** variables numéricas — distancia máxima entre
  las dos distribuciones acumuladas, con su p-value.
- **PSI (Population Stability Index):** variables numéricas — compara, por
  franjas, qué proporción de cada muestra cae en cada una.
- **Jensen-Shannon divergence:** numéricas y categóricas, sobre distribuciones
  discretizadas o de frecuencias.
- **Chi-cuadrado:** variables categóricas — compara frecuencias observadas entre
  train y el mes evaluado.

En un proyecto real normalmente se elige una sola métrica por tipo de variable
(PSI es el estándar más común en riesgo crediticio); acá se calculan las 4 porque
la consigna las pide explícitamente, no porque sea lo más eficiente.

### Decisión sobre el criterio de alerta (hallazgo metodológico)

La primera versión marcaba alerta si **una sola** de las métricas superaba su
umbral. Resultado: 17 de 20 variables quedaban marcadas con alerta en al menos un
mes — prácticamente todo el dataset. Investigando la causa (en vez de aceptar el
resultado), se identificaron dos problemas:

1. **Comparaciones múltiples:** se evalúan 12 meses x 20 variables = 240 tests de
   hipótesis al 5% de significancia. Si una variable no tiene ningún drift real,
   cada uno de sus 12 tests mensuales igual tiene ~5% de probabilidad de
   dispararse por azar; con 12 oportunidades, la probabilidad de que se dispare
   al menos una vez solo por azar es 1 − 0.95¹² ≈ 46%.
2. **KS hiper-sensible por asimetría de tamaños:** el set de train (~8600 filas)
   es mucho más grande que cada mes del test (43 a 406 filas). Con una referencia
   tan grande, el KS test detecta como "estadísticamente significativa" (p-value
   bajo) hasta la diferencia más mínima e irrelevante.

**Corrección aplicada:** una alerta ahora exige que coincidan al menos 2 de las
métricas disponibles (de 3 para numéricas, de 2 para categóricas), y además el
KS exige, junto al p-value bajo, que el estadístico KS en sí (la distancia entre
distribuciones, no solo su significancia) supere `UMBRAL_KS_STAT = 0.15` —
significancia estadística y relevancia práctica a la vez. Con este criterio, el
resultado bajó a **34 alertas de 240 (14%)**, mucho más razonable.

Nota honesta: como el split train/test es aleatorio (no se particiona por mes),
cada "mes" del test es en realidad una submuestra al azar de la misma población
total — no hay drift temporal real posible en este dataset estático. Cualquier
alerta que sobreviva este criterio más estricto sigue siendo, en rigor, ruido
estadístico y no evidencia de un corrimiento real. El valor de este avance es la
metodología (el pipeline detectaría drift real si los datos realmente
evolucionaran), no el hallazgo en sí sobre estos datos puntuales.

### App de Streamlit (`app_monitoring.py`)

Visualiza los resultados de `resultados/drift_metrics.csv`:

- Selector de mes (barra lateral) y tabla de variables de ese mes con semáforo
  (🔴/🟢) según si hay alerta.
- Métrica resumen: cantidad de variables con alerta sobre el total.
- Comparación de distribuciones: histograma superpuesto train vs. el mes
  elegido, para la variable que se seleccione.
- Evolución del drift a lo largo del año: gráfico de barras con la cantidad de
  alertas por mes.
- Recomendación automática (texto que cambia según la cantidad de alertas del
  mes: sin alertas, seguir monitoreando, o revisar variables y evaluar
  reentrenamiento).

Para correrla: `streamlit run app_monitoring.py` desde `mlops_pipeline/src`.

*Proyecto en desarrollo — este README se irá actualizando con los próximos avances.*
