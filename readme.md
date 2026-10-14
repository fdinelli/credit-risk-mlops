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
│       └── model_monitoring.py
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

Resultados completos en `mlops_pipeline/src/resultados/` (`tabla_resumen_modelos.csv`
y `comparacion_modelos.png`), generados por `model_training_evaluation.py`.

### Nota sobre data leakage detectado

Durante la evaluación inicial, los tres modelos dieron métricas perfectas
(ROC-AUC = 1.0), lo cual no es plausible en un problema real. Se identificó que
la variable `puntaje` separaba completamente las dos clases del target sin ningún
solapamiento (grupo "no pagó a tiempo" hasta 62.67, grupo "sí pagó" desde 63.81) —
un patrón imposible en un score de crédito genuino, y típico de una variable
generada en el dataset a partir del propio resultado. Se descartó `puntaje` de las
features (`ft_engineering.py`) y se reentrenaron los modelos; las métricas de la
tabla de arriba corresponden a esa versión corregida.

*Proyecto en desarrollo — este README se irá actualizando con los próximos avances.*
