import streamlit as st
import requests

# URL base de tu API FastAPI
API_URL = 'http://fastapi:8800'

# Función para obtener las marcas, utilizando caché
@st.cache_data
def obtener_marcas(api_url):
    try:
        response = requests.get(f"{api_url}/marcas/")
        if response.status_code == 200:
            data = response.json()
            return data['marcas']
        else:
            st.error("No se pudo obtener la lista de marcas. Por favor, ingresa la marca manualmente.")
            return None
    except Exception as e:
        st.error("Error al conectar con la API para obtener las marcas.")
        return None

# Función para obtener los tipos de vehículos, utilizando caché
@st.cache_data
def obtener_tipos(api_url):
    try:
        response = requests.get(f"{api_url}/tipos/")
        if response.status_code == 200:
            data = response.json()
            return data['tipos']
        else:
            st.error("No se pudo obtener la lista de tipos de vehículos. Por favor, ingresa el tipo manualmente.")
            return None
    except Exception as e:
        st.error("Error al conectar con la API para obtener los tipos de vehículos.")
        return None

# Título de la aplicación
st.title("Predicción de Precios de Autos Usados")

# Obtener las listas desde la API utilizando las funciones de caché
lista_marcas = obtener_marcas(API_URL)
lista_tipos = obtener_tipos(API_URL)

# Campos de entrada
if lista_marcas:
    marca = st.selectbox("Marca del auto", options=lista_marcas)
else:
    marca = st.text_input("Marca del auto", "Nissan")

motor = st.text_input("Tipo de motor", "1.4")

ano = st.number_input("Año del auto", min_value=1900, max_value=2100, step=1, value=2020)

if lista_tipos:
    tipo = st.selectbox("Tipo de vehículo", options=lista_tipos)
else:
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

    # Realizar la solicitud de predicción
    url = f"{API_URL}/predict/"  # Asegúrate de que la URL sea correcta
    try:
        response = requests.post(url, json={"features": payload})
        if response.status_code == 200:
            resultado = response.json()
            st.success(f"El precio estimado es: {resultado['int_output']} bolivianos")
        else:
            st.error("Error en la predicción")
    except Exception as e:
        st.error("Error al conectar con la API para realizar la predicción.")
