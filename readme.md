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
│       ├── app_monitoring.py
│       └── resultados/
│           ├── modelo_logistic_regression.pkl
│           └── preprocesador.pkl
├── Base_de_datos.csv
├── requirements.txt
├── requirements-api.txt
├── Dockerfile
├── .dockerignore
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

## Avance 1 — Versionamiento y EDA

- Estructura de carpetas y ramas (`developer`, `certification`, `main`) creadas según la consigna.
- `Cargar_datos.ipynb`: carga del dataset real y verificación inicial (tipos de dato, nulos a alto nivel).
- `comprension_eda.ipynb`: exploración de tipos de variable, tratamiento de nulos (7 columnas),
  verificación de duplicados (sin duplicados) y tratamiento de outliers en varias columnas.
- Entorno virtual y `requirements.txt` generados.

### Hallazgos principales del EDA

- **23 variables en total, 7 con valores nulos.** Varias llegaban mal tipadas (como texto en vez de
  numéricas); se corrigieron para este análisis (el tratamiento definitivo queda en `ft_engineering.py`,
  Avance 2). Verificación de duplicados: **ninguno** en todo el dataset.
- **`tipo_credito`**: 6 valores distintos, fuertemente concentrados en dos categorías (4 y 9, ~10.600 de
  ~10.900 casos). El valor `68` aparece en un solo caso, rompe la escala del resto (4 a 10) y es sospechoso
  de ser un error de carga — tratado como nulo en Avance 2.
- **`plazo_meses`**: 18 valores distintos pero sin progresión proporcional entre ellos (de 12 a 18 hay 6
  meses, de 18 a 20 solo 2) — indica que son plazos comerciales específicos de la financiera, no una
  escala numérica continua.
- **Outliers por error de carga, no por comportamiento real del cliente** (criterio: comparar contra el
  percentil 75 real de cada variable, no "a ojo"):
  - `salario_cliente`: valores de hasta 22.000 millones (media real ~17,2 millones) — 74 filas (~0,69%)
    por encima del umbral de 100 millones, tratadas como nulo.
  - `total_otros_prestamos`: mismo patrón, hasta 6.787 millones — 44 filas (~0,41%) por encima del mismo
    umbral.
  - `edad_cliente`: valores 121/122/123 (edades imposibles) — reemplazados por la mediana.
  - `cant_creditosvigentes`: salto abrupto de 38 a 62 sin valores intermedios — 1 caso aislado, tratado
    como error.
  - `puntaje_datacredito`: código `-7` identificado como valor de error de la central de riesgo, no un
    puntaje real.
- **Validación cruzada, no solo descriptiva**: `cuota_pactada` tenía valores que a primera vista parecían
  outliers (hasta 3,8 millones), pero al cruzarlos con `capital_prestado` y `plazo_meses` resultaron ser
  matemáticamente consistentes (capitales altos a plazos cortos) — se mantuvieron sin tratar, evitando
  descartar datos válidos por error.
- **Criterio de imputación**: mediana para las variables numéricas con outliers, por ser la medida más
  robusta frente a valores atípicos — mismo criterio aplicado de forma consistente en `ft_engineering.py`
  (Avance 2).

### Análisis bivariado y multivariado

- **Bivariado (variables vs. `Pago_atiempo`)**: `saldo_mora` es la variable que más separa a los dos
  grupos (promedio ~82 en quienes no pagaron a tiempo vs. ~4 en quienes sí, ~20x) — coherente con que la
  mora ya existente al momento del préstamo es una señal directa de riesgo (a diferencia de `puntaje`, que
  era leakage real). `huella_consulta` también es más alta en el grupo de riesgo (más consultas de crédito
  = más necesidad financiera), y `puntaje_datacredito` / `promedio_ingresos_datacredito` son más altos en
  quienes sí pagan a tiempo, como es esperable. Entre las categóricas, `tendencia_ingresos` = Decreciente
  tiene la tasa de pago a tiempo más baja (93,7% vs. 96,1% de Creciente); `tipo_credito` = 6 muestra una
  tasa mucho más baja (57%) pero con solo 21 casos, insuficiente para una conclusión firme.
- **Multivariado (matriz de correlación)**: las correlaciones más altas entre variables numéricas se
  explican todas por el diseño del producto financiero (ej. `capital_prestado` con `cuota_pactada`,
  0,76 — la cuota se calcula a partir del capital), no por redundancia accidental. Ninguna se acerca a 1,
  a diferencia de lo que pasaba con `puntaje` respecto al target — refuerza que ese caso fue el único de
  leakage real en el dataset.

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
resultado bajó a **31 alertas de 240 (13%)**, mucho más razonable.

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

## Avance 4 — API con FastAPI y empaquetado en Docker

### Persistencia del modelo (requisito previo)

La API necesita el modelo y el preprocesador ya entrenados, sin volver a
entrenarlos en cada arranque. Hasta el Avance 3, `model_training_evaluation.py`
entrenaba los tres modelos pero nunca los guardaba en disco. Se agregó, al final
del bloque principal del script, la persistencia con `joblib` tanto del modelo
ganador (Logistic Regression) como del preprocesador ya fiteado — el mismo
objeto de `ft_engineering.py` que `generar_datasets()` deja ajustado con
`fit_transform(X_train)`. Ambos quedan en `resultados/modelo_logistic_regression.pkl`
y `resultados/preprocesador.pkl`.

Esta distinción importa: en producción el preprocesador siempre se usa con
`.transform(...)`, nunca con `.fit_transform(...)` — volver a fitear sobre datos
nuevos (o sobre un único cliente) rompería la consistencia con lo que el modelo
aprendió en entrenamiento.

### `model_deploy.py` — API con FastAPI

Expone el modelo vía dos endpoints:

- **`GET /`**: chequeo simple de que la API está funcionando.
- **`POST /predict`**: recibe los datos de **un** cliente (validados
  automáticamente contra el esquema `Cliente`, un `BaseModel` de Pydantic con
  los 20 features del modelo) y devuelve `pago_a_tiempo` (bool) y
  `probabilidad` (float, probabilidad de la clase positiva).
- **`POST /predict_batch`**: la misma lógica pero recibe una lista de
  clientes y devuelve una lista de predicciones — pensado para scoring de
  varios créditos a la vez, no solo uno por uno.

Documentación interactiva autogenerada por FastAPI en `/docs` (Swagger UI),
usada para probar ambos endpoints antes de empaquetar la imagen.

### Empaquetado en Docker

**`Dockerfile`:** imagen base `python:3.13-slim`, instala dependencias,
copia el proyecto y corre `uvicorn model_deploy:app --host 0.0.0.0 --port 8000`
desde `mlops_pipeline/src`. El `--host 0.0.0.0` es necesario para que el
servidor sea accesible desde fuera del contenedor (no solo desde `localhost`
dentro de él). Las dependencias se copian e instalan en un paso separado,
*antes* de copiar el resto del código: así, si solo cambia el código de la
API, Docker reutiliza la capa de dependencias ya instaladas en vez de
reinstalar todo de cero.

**`requirements-api.txt` (hallazgo y decisión):** el primer intento de build
usando el `requirements.txt` completo falló al instalar `pywinpty` — un
paquete exclusivo de Windows que llega como dependencia transitiva de Jupyter,
y que no puede compilarse dentro de un contenedor Linux. El `requirements.txt`
original está pensado para el entorno de desarrollo (notebooks, Jupyter, etc.),
no para lo que la API necesita en producción. Se creó `requirements-api.txt`
con únicamente las 7 dependencias reales de `model_deploy.py`
(fastapi, uvicorn, pydantic, pandas, numpy, scikit-learn, joblib) y el
Dockerfile instala desde ese archivo — una imagen más liviana y sin paquetes
incompatibles con Linux. De paso se detectó que `fastapi`, `uvicorn` y
`pydantic` tampoco estaban en el `requirements.txt` del entorno de desarrollo
(se habían instalado con pip pero nunca se habían re-congelado ahí), así que
se agregaron también.

**`.dockerignore`:** excluye del build de la imagen lo que no hace falta en
producción — el entorno virtual (`mod5/`), cachés de Python, `.git/`, la
configuración de VS Code y el propio `Base_de_datos.csv` (el dataset crudo
solo se usa para entrenar, localmente; la API en producción solo necesita los
`.pkl` ya entrenados en `resultados/`).

### Cómo correrlo

```
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
```

Luego, `http://localhost:8000/docs` para la documentación interactiva.

*Proyecto completo — los 4 avances del PIM5 fueron entregados.*
