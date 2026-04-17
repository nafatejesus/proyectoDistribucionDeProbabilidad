import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import google.generativeai as genai
import re

# Configuración de la página
st.set_page_config(page_title="Z-Test Analyzer Pro", layout="wide")

# --- FUNCIONES ESTADÍSTICAS ---
def perform_z_test(sample_mean, pop_std, n, null_mean, alpha, alternative):
    se = pop_std / np.sqrt(n)
    z_stat = (sample_mean - null_mean) / se
    if alternative == 'Bilateral':
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        z_crit = stats.norm.ppf(1 - alpha/2)
        return z_stat, p_value, -z_crit, z_crit
    elif alternative == 'Cola Derecha':
        p_value = 1 - stats.norm.cdf(z_stat)
        z_crit = stats.norm.ppf(1 - alpha)
        return z_stat, p_value, None, z_crit
    else:
        p_value = stats.norm.cdf(z_stat)
        z_crit = stats.norm.ppf(alpha)
        return z_stat, p_value, z_crit, None

# --- SIDEBAR: ENTRADA DE DATOS ---
st.sidebar.header("1. Entrada de Datos")
modo = st.sidebar.radio("Origen:", ["Subir CSV", "Generar Datos de Prueba"])

df = None
if modo == "Subir CSV":
    archivo = st.sidebar.file_uploader("Sube tu encuesta (CSV)", type=["csv"])
    if archivo:
        try:
            df = pd.read_csv(archivo, sep=None, engine='python', encoding='utf-8')
        except:
            df = pd.read_csv(archivo, sep=',', encoding='latin-1')
    else:
        st.info("Esperando archivo... Puedes usar el 'Generador' para probar la app ahora.")
        st.stop()
else:
    # Datos sintéticos para pruebas rápidas
    df = pd.DataFrame({'Datos_Sinteticos': np.random.normal(100, 15, 150)})

# --- PROCESAMIENTO ---
st.title("🧪 Analizador Estadístico (Prueba Z)")

col_seleccionada = st.selectbox("Selecciona la columna a analizar:", df.columns)

# LIMPIEZA DE DATOS
raw_values = df[col_seleccionada].astype(str)
clean_values = raw_values.str.extract(r'(\d+\.?\d*)')[0]
data = pd.to_numeric(clean_values, errors='coerce').dropna()

if data.empty:
    st.error(f"La columna '{col_seleccionada}' no tiene valores numéricos extraíbles.")
    st.stop()

# --- PARÁMETROS ESTADÍSTICOS ---
st.sidebar.header("2. Parámetros de Prueba")
h0 = st.sidebar.number_input("Media Hipotética (H0)", value=float(round(data.mean(), 2)))
sigma = st.sidebar.number_input("Desviación Estándar Poblacional (σ)", value=15.0, min_value=0.1)
alpha = st.sidebar.slider("Significancia (α)", 0.01, 0.10, 0.05)
tipo = st.sidebar.selectbox("Hipótesis Alternativa", ["Bilateral", "Cola Derecha", "Cola Izquierda"])

# --- VISUALIZACIÓN INICIAL ---
col1, col2 = st.columns(2)
with col1:
    fig_h, ax_h = plt.subplots()
    sns.histplot(data, kde=True, color="skyblue", ax=ax_h)
    ax_h.set_title("Distribución de la Muestra")
    st.pyplot(fig_h)

with col2:
    fig_b, ax_b = plt.subplots()
    sns.boxplot(x=data, color="lightcoral", ax=ax_b)
    ax_b.set_title("Identificación de Outliers")
    st.pyplot(fig_b)

# --- CÁLCULOS Y RESULTADOS ---
n = len(data)
x_bar = data.mean()
z_calc, p_val, z_low, z_high = perform_z_test(x_bar, sigma, n, h0, alpha, tipo)

st.divider()
st.subheader("📊 Análisis de Resultados")

c1, c2, c3 = st.columns(3)
c1.metric("Media Muestral", round(x_bar, 4))
c2.metric("Estadístico Z", round(z_calc, 4))
c3.metric("P-Valor", round(p_val, 4))

st.markdown("### Conclusión")
if p_val < alpha:
    st.error(f"Decisión: Rechazar H0. Hay evidencia suficiente con un {int((1-alpha)*100)}% de confianza.")
else:
    st.success(f"Decisión: No Rechazar H0. No hay evidencia suficiente para descartar la hipótesis nula.")

# --- GRÁFICO DE CAMPANA ---
st.subheader("Curva de Decisión")
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)
fig_z, ax_z = plt.subplots(figsize=(10, 4))
ax_z.plot(x, y, color='black', lw=2)

if tipo == 'Bilateral':
    ax_z.fill_between(x, y, where=(x <= z_low) | (x >= z_high), color='red', alpha=0.4, label='Zona de Rechazo')
elif tipo == 'Cola Derecha':
    ax_z.fill_between(x, y, where=(x >= z_high), color='red', alpha=0.4, label='Zona de Rechazo')
else:
    ax_z.fill_between(x, y, where=(x <= z_low), color='red', alpha=0.4, label='Zona de Rechazo')

ax_z.axvline(z_calc, color='blue', ls='--', lw=2, label=f'Z-Calculado: {z_calc:.2f}')
ax_z.legend()
st.pyplot(fig_z)

# Configuración de la IA usando los secretos de Streamlit
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-2.5-flash')

st.divider()
st.header("4. Asistente de IA (Google Gemini)")

if st.button("Generar interpretación con IA"):
    prompt = f"""
    Se realizó una prueba Z con los siguientes parámetros:
    - Media muestral: {x_bar:.4f}
    - Media hipotética (H0): {h0}
    - Tamaño de muestra (n): {n}
    - Desviación estándar (sigma): {sigma}
    - Nivel de significancia (alpha): {alpha}
    - Estadístico Z calculado: {z_calc:.4f}
    - P-valor: {p_val:.4f}

    ¿Se rechaza H0? Explica la decisión técnica y la interpretación estadística.
    """
    with st.spinner("Consultando a la IA..."):
        try:
            response = model.generate_content(prompt)
            st.markdown("### Respuesta de la IA:")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error: {e}")
