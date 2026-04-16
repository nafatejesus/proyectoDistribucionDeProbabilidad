import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración visual de la página
st.set_page_config(page_title="Z-Test Master - Estadística ITI", layout="wide")

def perform_z_test(sample_mean, pop_std, n, null_mean, alpha, alternative):
    """Cálculos lógicos de la Prueba Z."""
    se = pop_std / np.sqrt(n)
    z_stat = (sample_mean - null_mean) / se
    
    if alternative == 'Bilateral':
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        z_critical = stats.norm.ppf(1 - alpha/2)
        lower_critical, upper_critical = -z_critical, z_critical
    elif alternative == 'Cola Derecha':
        p_value = 1 - stats.norm.cdf(z_stat)
        z_critical = stats.norm.ppf(1 - alpha)
        lower_critical, upper_critical = None, z_critical
    else: # Cola Izquierda
        p_value = stats.norm.cdf(z_stat)
        z_critical = stats.norm.ppf(alpha)
        lower_critical, upper_critical = z_critical, None
        
    return z_stat, p_value, lower_critical, upper_critical

# --- SIDEBAR: ENTRADA DE DATOS ---
st.sidebar.header("1. Carga de Datos")
data_source = st.sidebar.radio("Origen:", ("Subir CSV", "Generar Datos Sintéticos"))

df = None

if data_source == "Subir CSV":
    uploaded_file = st.sidebar.file_uploader("Carga tu archivo CSV", type=["csv"])
    if uploaded_file:
        # Intentamos detectar el separador automáticamente (coma o punto y coma)
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
            df.columns = [c.strip() for c in df.columns] # Limpiar espacios en nombres
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            st.stop()
    else:
        st.info("Sube un archivo CSV en el panel de la izquierda.")
        st.stop()
else:
    n_synth = st.sidebar.slider("Tamaño de muestra (n)", 30, 1000, 100)
    mu_synth = st.sidebar.number_input("Media real (simulación)", value=100.0)
    std_synth = st.sidebar.number_input("Desv. Estándar (simulación)", value=15.0)
    df = pd.DataFrame({'datos_generados': np.random.normal(mu_synth, std_synth, n_synth)})

st.sidebar.header("2. Parámetros de la Prueba")
h0_mean = st.sidebar.number_input("Media Hipotética ($H_0$)", value=100.0)
pop_sigma = st.sidebar.number_input("$\sigma$ Poblacional (Conocida)", value=15.0, min_value=0.01)
alpha = st.sidebar.slider("Nivel de Significancia ($\\alpha$)", 0.01, 0.10, 0.05)
test_type = st.sidebar.selectbox("Tipo de Hipótesis", ["Bilateral", "Cola Derecha", "Cola Izquierda"])

# --- CUERPO PRINCIPAL ---
st.title(" Aplicación de Prueba Z")
st.markdown("---")

# Selección de columna (Mostramos todas para que el usuario elija)
target_col = st.selectbox("Selecciona la columna para el análisis:", df.columns.tolist())

# PROCESAMIENTO CRÍTICO: Forzamos conversión a números
data_series = pd.to_numeric(df[target_col], errors='coerce').dropna()

if data_series.empty:
    st.error(f" La columna '{target_col}' no contiene datos numéricos. Por favor, selecciona otra.")
    st.stop()

# Vista previa de datos limpios
with st.expander("Ver datos procesados"):
    st.write(f"Registros válidos encontrados: {len(data_series)}")
    st.write(data_series.head())

# 1. Gráficos
col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(data_series, kde=True, color="#2E86C1", ax=ax1)
    ax1.set_title("Distribución de la Muestra")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=data_series, color="#EC7063", ax=ax2)
    ax2.set_title("Análisis de Outliers")
    st.pyplot(fig2)

# 2. Cálculos Estadísticos
n = len(data_series)
x_bar = data_series.mean()

st.divider()
st.subheader(" Resultados de la Estadística")

z_calc, p_val, z_low, z_high = perform_z_test(x_bar, pop_sigma, n, h0_mean, alpha, test_type)

m1, m2, m3 = st.columns(3)
m1.metric("Z Calculado", f"{z_calc:.4f}")
m2.metric("Valor P", f"{p_val:.4f}")
m3.metric("Media Muestral ($\overline{x}$)", f"{x_bar:.2f}")

# 3. Conclusión
decision = "Rechazar $H_0$" if p_val < alpha else "No Rechazar $H_0$"
color = "green" if "No Rechazar" in decision else "red"

st.markdown(f"### Decisión: :{color}[{decision}]")
if p_val < alpha:
    st.write(f"Como el p-valor ({p_val:.4f}) < {alpha}, se rechaza la hipótesis nula.")
else:
    st.write(f"Como el p-valor ({p_val:.4f}) >= {alpha}, no hay evidencia suficiente para rechazar $H_0$.")

# 4. Gráfico de Curva Normal y Zonas de Rechazo
st.subheader("Visualización de la Campana de Gauss")
x_axis = np.linspace(-4, 4, 500)
y_axis = stats.norm.pdf(x_axis, 0, 1)
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(x_axis, y_axis, color='black')

if test_type == 'Bilateral':
    ax3.fill_between(x_axis, y_axis, where=(x_axis <= z_low) | (x_axis >= z_high), color='red', alpha=0.3, label='Zona de Rechazo')
elif test_type == 'Cola Derecha':
    ax3.fill_between(x_axis, y_axis, where=(x_axis >= z_high), color='red', alpha=0.3, label='Zona de Rechazo')
else:
    ax3.fill_between(x_axis, y_axis, where=(x_axis <= z_low), color='red', alpha=0.3, label='Zona de Rechazo')

ax3.axvline(z_calc, color='blue', linestyle='--', label=f'Z-Calculado ({z_calc:.2f})')
ax3.legend()
st.pyplot(fig3)