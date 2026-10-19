"""
model_trining_evaluation.py
Entrenamiento y evaluación de modelos supervisados para el modelo de riesgo
crediticio (PIM5, Avance 2).
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve
)

from ft_engineering import generar_datasets, preprocesador


def build_model(modelo, X_train, y_train):
    """
    Entrena un modelo ya instanciado (sin entrenar) y lo devuelve entrenado.
    Función genérica: se llama una vez por cada algoritmo que se quiera
    probar (no entrena varios modelos a la vez), pero garantiza que todos
    se entrenen exactamente de la misma manera.
    """
    modelo.fit(X_train, y_train)
    return modelo


def summarize_classification(modelo, X_test, y_test):
    """
    Evalúa un modelo ya entrenado sobre el set de test y devuelve un
    diccionario con las métricas clave. Estandariza el cálculo para que
    todos los modelos se comparen bajo las mismas condiciones.
    """
    y_pred = modelo.predict(X_test)
    # predict_proba(...)[:, 1] = probabilidad de la clase positiva,
    # necesaria para roc_auc_score (no alcanza con la predicción 0/1).
    y_proba = modelo.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


if __name__ == "__main__":
    # random_state=42 en todos los modelos para que los resultados sean
    # reproducibles (mismo split y mismo entrenamiento en cada corrida).
    X_train, X_test, y_train, y_test = generar_datasets("../../Base_de_datos.csv")

    # max_iter=1000: con el default (100) el solver no llega a converger
    # con este dataset y tira ConvergenceWarning.
    modelo_lr = build_model(LogisticRegression(max_iter=1000), X_train, y_train)
    resultados_lr = summarize_classification(modelo_lr, X_test, y_test)
    print("Logistic Regression:", resultados_lr)

    modelo_dt = build_model(DecisionTreeClassifier(random_state=42), X_train, y_train)
    resultados_dt = summarize_classification(modelo_dt, X_test, y_test)
    print("Decision Tree:", resultados_dt)

    modelo_rf = build_model(RandomForestClassifier(random_state=42), X_train, y_train)
    resultados_rf = summarize_classification(modelo_rf, X_test, y_test)
    print("Random Forest:", resultados_rf)

    # Diagnóstico: ¿qué variable explica las métricas perfectas? (posible data leakage)
    importancias = pd.Series(
        modelo_rf.feature_importances_,
        index=preprocesador.get_feature_names_out()
    ).sort_values(ascending=False)

    print(importancias.head(10))

    # Carpeta donde se guardan los resultados generados por este script
    carpeta_resultados = "resultados"
    os.makedirs(carpeta_resultados, exist_ok=True)

    # Tabla resumen: compara los tres modelos con las mismas métricas
    resultados = {
        "Logistic Regression": resultados_lr,
        "Decision Tree": resultados_dt,
        "Random Forest": resultados_rf,
    }
    tabla_resumen = pd.DataFrame(resultados).T
    print("\nTabla resumen de evaluación:")
    print(tabla_resumen)
    tabla_resumen.to_csv(os.path.join(carpeta_resultados, "tabla_resumen_modelos.csv"))

    # Gráfico comparativo: F1 y ROC-AUC de cada modelo
    tabla_resumen[["f1", "roc_auc"]].plot(kind="bar", figsize=(8, 5))
    plt.title("Comparación de modelos: F1 y ROC-AUC")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_resultados, "comparacion_modelos.png"))
    print(f"\nResultados guardados en la carpeta '{carpeta_resultados}/'")

    # --------------------------------------------------------------------
    # Curva Precision-Recall
    # --------------------------------------------------------------------
    # precision/recall en la tabla resumen son un único punto (con el
    # umbral de decisión por defecto, 0.5). La curva muestra el trade-off
    # completo entre ambas al mover ese umbral: útil en un caso de riesgo
    # crediticio, donde se podría querer ajustar el umbral según cuánto
    # pesa, para el negocio, rechazar un buen cliente vs. aprobar uno malo.
    modelos_entrenados = {
        "Logistic Regression": modelo_lr,
        "Decision Tree": modelo_dt,
        "Random Forest": modelo_rf,
    }

    plt.figure(figsize=(7, 6))
    for nombre, modelo in modelos_entrenados.items():
        y_proba = modelo.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=nombre)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall por modelo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_resultados, "curva_precision_recall.png"))
    print(f"Curva Precision-Recall guardada en '{carpeta_resultados}/curva_precision_recall.png'")

    # --------------------------------------------------------------------
    # Selección del modelo de mejor performance
    # --------------------------------------------------------------------
    # El dataset está fuertemente desbalanceado (~95% de los clientes paga
    # a tiempo), por lo que accuracy no es confiable: un modelo que siempre
    # prediga la clase mayoritaria ya obtendría ~95% sin haber aprendido
    # nada. La métrica relevante acá es ROC-AUC, porque no depende de la
    # proporción de clases y mide qué tan bien el modelo distingue entre
    # clientes que pagan y los que no.
    #
    # Con los resultados obtenidos (ver tabla_resumen_modelos.csv):
    #   - Logistic Regression: ROC-AUC 0.639 (el más alto)
    #   - Random Forest:       ROC-AUC 0.629 (prácticamente empatado)
    #   - Decision Tree:       ROC-AUC 0.528 (el más débil, casi al azar)
    #
    # Se elige Logistic Regression como modelo de mejor performance:
    # tiene el ROC-AUC más alto y, a diferencia de Random Forest, es un
    # modelo simple e interpretable — una ventaja relevante en un caso de
    # riesgo crediticio, donde poder explicar por qué se aprueba o
    # rechaza un crédito es tan importante como la métrica en sí.
    print("\nModelo seleccionado: Logistic Regression (mayor ROC-AUC, mayor interpretabilidad)")

    # --------------------------------------------------------------------
    # Persistencia del modelo y del preprocesador (Avance 4)
    # --------------------------------------------------------------------
    # Se guardan ambos objetos con joblib (más eficiente que pickle para los
    # arrays de numpy que hay adentro de un modelo/ColumnTransformer ya
    # entrenado). El preprocesador que se guarda es el mismo objeto global
    # importado de ft_engineering: generar_datasets() ya lo dejó fiteado con
    # fit_transform(X_train), así que lo que se persiste acá es exactamente
    # el que se usó para entrenar — necesario para que, en producción (la
    # API de model_deploy.py), los datos nuevos se transformen de la misma
    # manera antes de pasarlos al modelo.
    joblib.dump(modelo_lr, os.path.join(carpeta_resultados, "modelo_logistic_regression.pkl"))
    joblib.dump(preprocesador, os.path.join(carpeta_resultados, "preprocesador.pkl"))
    print(f"Modelo y preprocesador guardados en '{carpeta_resultados}/'")
