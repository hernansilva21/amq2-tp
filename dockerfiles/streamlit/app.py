import streamlit as st
import requests

# Título de la aplicación
st.title("Predicción de Precios de Autos Usados")

# Campos de entrada
marca = st.text_input("Marca del auto", "Nissan")
motor = st.text_input("Tipo de motor", "1.4")
ano = st.number_input("Año del auto", min_value=1900, max_value=2100, step=1, value=2020)
tipo = st.text_input("Tipo de vehículo", "SUV")
transmision = st.selectbox("Transmisión", ["manual", "automática"])

# Botón de predicción
if st.button("Predecir"):
    payload = {
        "marca": marca,
        "Motor": motor,
        "Ano": ano,
        "Tipo": tipo,
        "Transmision": transmision,
    }

    # Realizar la solicitud
    url = 'http://fastapi:8800/predict/'  # Asegúrate de que la URL sea correcta
    response = requests.post(url, json={"features": payload})

    if response.status_code == 200:
        resultado = response.json()
        st.success(f"El precio estimado es: {resultado['int_output']} bolivianos")
    else:
        st.error("Error en la predicción")
