# src/model_deploy.py

# librerías
import pandas as pd 
import numpy as np 
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# inicialización de la aplicación
app = FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description="Despliega un modelo de machine learning para predecir si un cliente pagará a tiempo o no.",
    version="1.0.0"
)

# en este punto cargamos el modelo entrenado (.pkl, .joblib)
try:
    # aquí cargamos el modelo desde el archivo model.pkl
    with open("../models/model.pkl", "rb") as f:
        modelo = pickle.load(f)

    print("Modelo cargado exitosamente")

except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

# nos creamos un endpoint de saludo
@app.get("/saludo")
def saludo():
    return {"mensaje":"Hola, esta API está corriendo correctamente..."}

# ahora, nos vamos a crear un endpoint para hacer predicciones
@app.post("/predict")
def predict_batch(input_data: dict):
    if modelo is None:
        return "El modelo no pudo ser cargado. Revisa los logs del servidor."

    try:
        # acá es donde se tiene el modelo cargado y listo para hacer predicciones
        return "El modelo está cargado y listo para hacer predicciones."

    except Exception as e:
        return f"Error al hacer las predicciones: {e}"


# cargar el script
if __name__ == "__main__":
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)