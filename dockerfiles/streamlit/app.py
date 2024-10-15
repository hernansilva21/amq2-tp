import streamlit as st
import requests

# app title
st.title("Predicción de Precios de Autos Usados")

# input fields
modelo = st.text_input("Modelo del auto", "")
año = st.number_input("Año del auto", min_value=1900, max_value=2100, step=1, value=2020)
kilometraje = st.number_input("Kilometraje (en km)", min_value=0, step=1000, value=50000)

# Predict button
if st.button("Predecir"):
    payload = {
        "modelo": modelo,
        "año": año,
        "kilometraje": kilometraje,
    }

    # Making the request
    url = 'http://fastapi:8000/predict'  # Aquí usamos el nombre del contenedor para llamar a FastAPI
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        resultado = response.json()
        st.success(f"El precio estimado es: {resultado['precio']}")
    else:
        st.error("Error en la predicción")
