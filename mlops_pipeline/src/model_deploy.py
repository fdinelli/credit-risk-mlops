from fastapi import FastAPI
import joblib

app = FastAPI()

modelo = joblib.load("resultados/modelo_logistic_regression.pkl")
preprocesador = joblib.load("resultados/preprocesador.pkl")

from pydantic import BaseModel

class Cliente(BaseModel):
    capital_prestado: float
    plazo_meses: float
    edad_cliente: float
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: float
    huella_consulta: float
    saldo_total: float
    saldo_principal: float
    creditos_sectorFinanciero: float
    creditos_sectorCooperativo: float
    creditos_sectorReal: float
    promedio_ingresos_datacredito: float
    saldo_mora: float
    saldo_mora_codeudor: float
    tipo_credito: str
    tipo_laboral: str
    mes_prestamo: int
    tendencia_ingresos: str

@app.get("/")
def home():
    return {"mensaje": "API de riesgo crediticio funcionando"}

import pandas as pd

@app.post("/predict")
def predict(cliente: Cliente):
    datos = pd.DataFrame([cliente.model_dump()])
    datos_transformados = preprocesador.transform(datos)
    prediccion = modelo.predict(datos_transformados)[0]
    probabilidad = modelo.predict_proba(datos_transformados)[0][1]

    return {
        "pago_a_tiempo": bool(prediccion),
        "probabilidad": float(probabilidad)
    }

from typing import List

@app.post("/predict_batch")
def predict_batch(clientes: List[Cliente]):
    datos = pd.DataFrame([c.model_dump() for c in clientes])
    datos_transformados = preprocesador.transform(datos)
    predicciones = modelo.predict(datos_transformados)
    probabilidades = modelo.predict_proba(datos_transformados)[:, 1]

    resultados = []
    for pred, prob in zip(predicciones, probabilidades):
        resultados.append({
            "pago_a_tiempo": bool(pred),
            "probabilidad": float(prob)
        })
    return resultados