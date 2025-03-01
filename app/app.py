import streamlit as st
import requests

# para correr:  streamlit run app.py
st.title("Predicción de ganador en la Libertadores")


# Creamos el formulario
with st.form("libertadores"):
    st.write("Ingrese los datos del partido:")
    fase = st.text_input("Fase")
    local = st.text_input("Equipo local")
    visitante = st.text_input("Equipo visitante")
    submit_button = st.form_submit_button("Predecir")

if submit_button:
    url = "https://libertadores-api-production.up.railway.app/predict"
    payload = {
        "round": fase,
        "homeClub": local,
        "awayClub": visitante,
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            resultado = response.json().get("result")
            st.success(f"{resultado}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Ocurrió un error al conectar con el servicio: {e}")
