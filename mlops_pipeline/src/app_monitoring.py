import streamlit as st
import pandas as pd

st.title("Monitoreo de Data Drift — Riesgo Crediticio")

reporte = pd.read_csv("resultados/drift_metrics.csv")

# Selector de mes en la barra lateral
meses_disponibles = sorted(reporte["mes_prestamo"].unique())
mes_elegido = st.sidebar.selectbox("Elegí el mes a inspeccionar", meses_disponibles)

# Filtramos y agregamos la columna semáforo (misma lógica que en el notebook)
tabla_mes = reporte[reporte["mes_prestamo"] == mes_elegido].copy()
tabla_mes["semaforo"] = tabla_mes["alerta"].map({True: "🔴", False: "🟢"})

import matplotlib.pyplot as plt
from model_monitoring import cargar_referencia_y_actual

# Resumen rápido del mes
cantidad_alertas = tabla_mes["alerta"].sum()
st.metric("Variables con alerta este mes", f"{cantidad_alertas} de {len(tabla_mes)}")

st.subheader(f"Variables del mes {mes_elegido}")
st.dataframe(tabla_mes[["variable", "tipo", "semaforo", "psi", "ks_pvalue", "js_divergence"]])

@st.cache_data
def cargar_datos():
    return cargar_referencia_y_actual("../../Base_de_datos.csv")

X_train, X_test = cargar_datos()

st.subheader("Comparar distribución: train vs. mes elegido")
variable_elegida = st.selectbox("Elegí una variable numérica", sorted(
    [v for v in tabla_mes[tabla_mes["tipo"] == "numerica"]["variable"]]
))

mes_actual = X_test[X_test["mes_prestamo"] == mes_elegido]

fig, ax = plt.subplots()
ax.hist(X_train[variable_elegida].dropna(), bins=20, density=True, alpha=0.5, label="Train (referencia)")
ax.hist(mes_actual[variable_elegida].dropna(), bins=20, density=True, alpha=0.5, label=f"Mes {mes_elegido} (actual)")
ax.legend()
ax.set_title(f"{variable_elegida}: train vs. mes {mes_elegido}")
st.pyplot(fig)

st.subheader("Evolución del drift a lo largo del año")

alertas_por_mes = reporte.groupby("mes_prestamo")["alerta"].sum()
st.bar_chart(alertas_por_mes)

st.subheader("Recomendación")

if cantidad_alertas >= 5:
    st.error(
        f"⚠️ {cantidad_alertas} variables con drift en el mes {mes_elegido}. "
        "Se recomienda revisar estas variables y evaluar un reentrenamiento del modelo."
    )
elif cantidad_alertas > 0:
    st.warning(
        f"{cantidad_alertas} variable(s) con drift en el mes {mes_elegido}. "
        "Conviene seguir monitoreando, todavía no es crítico."
    )
else:
    st.success(f"Sin alertas de drift en el mes {mes_elegido}.")