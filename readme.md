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

*Proyecto en desarrollo — este README se irá actualizando con los próximos avances.*
