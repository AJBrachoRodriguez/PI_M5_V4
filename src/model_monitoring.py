# src/model_monitoring.py

# librerías
import os
import pandas as pd
# aquí vamos a importar la librería para crear la aplicación 'Streamlit'
import streamlit as st
st.set_page_config(page_title="Monitoreo del Modelo", layout="wide")
# aquí importamos librerías de visualización
import plotly.express as px
# librerías de sklearn
from sklearn.model_selection import train_test_split
# importamos el método para cargar los datos
from cargar_datos import cargarDatos

##########################################
# 1. Configuración
##########################################

API_URL = "http://localhost:8000/predict"
DATASET_PATH = "./Base_de_datos.xlsx"      # dataset "original" (este es el que vamos a monitonear)
MONIOR_LOG = "./Base_datos.csv"            # dataset "transformado"


##########################################
# 2. Cargar el dataset y dividir los datos
##########################################

@st.cache_data
def load_data():
    # 2.1 llamamos a la función cargarDatos para asignarlos a una variable "df"
    df = cargarDatos()

    # 2.2 Acá vamos a crear los features y el target
    target = "Pago_atiempo"
    X = df.drop(columns=[target]) # features
    y = df[target]                # target

    # 2.3 En este punto vamos a dividir los datos en train/test
    X_ref, X_new, y_ref, y_new = train_test_split(
        X,y, test_size=0.2, random_state=42, stratify=y
    )

    # 2.4 Que la función retorne estos arrays
    return X_ref, X_new, y_ref, y_new


X_ref, X_new, y_ref, y_new = load_data()

print("Dataset cargado correctamente y dividido en datos de referencia (ref) y nuevo (new).")
print("X_new son los features nuevos:")
print(X_new)
print("X_new son los features de referencia:")
print(X_ref)
print("y_new es el target nuevo:")
print(y_new)
print("y_ref es el target de referencia:")
print(y_new)

####################################
# 3. Crear el interfaz inicial de la 
# aplicación de Streamlit
####################################
st.title("📊 Aplicación para el monitoreo de datos")













