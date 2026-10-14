"""
model_trining_evaluation.py
Entrenamiento y evaluación de modelos supervisados para el modelo de riesgo
crediticio (PIM5, Avance 2).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from ft_engineering import generar_datasets, preprocesador


def build_model(modelo, X_train, y_train):
    """
    Entrena un modelo ya instanciado (sin entrenar) y lo devuelve entrenado.Se llama una vez por cada algoritmo que se quiera probar.
    """
    modelo.fit(X_train, y_train)
    return modelo


def summarize_classification(modelo, X_test, y_test):
    """
    Evalúa un modelo ya entrenado sobre el set de test. Devuelve un diccionario con las métricas clave para comparar modelos
    de forma consistente.
    """
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generar_datasets("../../Base_de_datos.csv")

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
