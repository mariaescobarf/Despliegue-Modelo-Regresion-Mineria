import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Configuración de página
st.set_page_config(page_title='Predicción de Matrículas', page_icon='🎓', layout='centered')

# Estilos personalizados
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500&display=swap');
        html, body, [class*='css'] { font-family: 'DM Sans', sans-serif; }
        .main { background-color: #f4f1ec; }
        h1 { font-family: 'DM Serif Display', serif !important; color: #1a1a2e !important; font-size: 2.2rem !important; }
        .subtitulo { color: #555; font-size: 1rem; margin-bottom: 2rem; }
        .resultado-box {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            margin-top: 1.5rem;
        }
        .resultado-numero { font-family: 'DM Serif Display', serif; font-size: 3.5rem; color: #e2b96f; }
        .resultado-label { font-size: 0.95rem; color: #aaa; margin-top: 0.3rem; }
    </style>
""", unsafe_allow_html=True)

# Cargar modelo
filename = 'modelo-final.pkl'
modelo, min_max_scaler, variables = pickle.load(open(filename, 'rb'))

# Interfaz
st.title('Predicción de Matrículas Universitarias de Pregrado')
st.markdown('<p class="subtitulo">Ingresa los nacimientos registrados en años anteriores para estimar el total de matrículas esperadas.</p>', unsafe_allow_html=True)

st.markdown('### 📊 Nacimientos por año de rezago')

col1, col2 = st.columns(2)
with col1:
    nac_total_19 = st.slider('Nacimientos hace 19 años', min_value=1000, max_value=200000, value=50000, step=500)
    nac_total_17 = st.slider('Nacimientos hace 17 años', min_value=1000, max_value=200000, value=50000, step=500)
with col2:
    nac_total_18 = st.slider('Nacimientos hace 18 años', min_value=1000, max_value=200000, value=50000, step=500)
    nac_total_16 = st.slider('Nacimientos hace 16 años', min_value=1000, max_value=200000, value=50000, step=500)

st.markdown('### 🏫 Sector institucional')
SECTOR = st.selectbox('Sector', ['OFICIAL', 'PRIVADA'])

# Proporción histórica hombres/mujeres (~51.2% / 48.8%)
def split_sexo(total):
    return round(total * 0.512), round(total * 0.488)

nac_h19, nac_m19 = split_sexo(nac_total_19)
nac_h18, nac_m18 = split_sexo(nac_total_18)
nac_h17, nac_m17 = split_sexo(nac_total_17)
nac_h16, nac_m16 = split_sexo(nac_total_16)

# Dataframe con los mismos nombres de variables del entrenamiento
datos = [[
    nac_total_19, nac_h19, nac_m19,
    nac_total_18, nac_h18, nac_m18,
    nac_total_17, nac_h17, nac_m17,
    nac_total_16, nac_h16, nac_m16,
    225428.59,  # PIB promedio del periodo
    1.0963,     # IPC promedio del periodo
    SECTOR
]]

data = pd.DataFrame(datos, columns=[
    'NACIMIENTOS_TOTAL_XMENOS19', 'NACIMIENTOS_HOMBRES_XMENOS19', 'NACIMIENTOS_MUJERES_XMENOS19',
    'NACIMIENTOS_TOTAL_XMENOS18', 'NACIMIENTOS_HOMBRES_XMENOS18', 'NACIMIENTOS_MUJERES_XMENOS18',
    'NACIMIENTOS_TOTAL_XMENOS17', 'NACIMIENTOS_HOMBRES_XMENOS17', 'NACIMIENTOS_MUJERES_XMENOS17',
    'NACIMIENTOS_TOTAL_XMENOS16', 'NACIMIENTOS_HOMBRES_XMENOS16', 'NACIMIENTOS_MUJERES_XMENOS16',
    'PIB_ANIO_ANTERIOR_MILLONES_COP', 'IPC_ANIO_ANTERIOR_PORC',
    'SECTOR'
])

# Preparación de datos
data_preparada = data.copy()
data_preparada = pd.get_dummies(data_preparada, columns=['SECTOR'], drop_first=True, dtype=int)
data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

predictoras_numericas = [
    'NACIMIENTOS_TOTAL_XMENOS19', 'NACIMIENTOS_HOMBRES_XMENOS19', 'NACIMIENTOS_MUJERES_XMENOS19',
    'NACIMIENTOS_TOTAL_XMENOS18', 'NACIMIENTOS_HOMBRES_XMENOS18', 'NACIMIENTOS_MUJERES_XMENOS18',
    'NACIMIENTOS_TOTAL_XMENOS17', 'NACIMIENTOS_HOMBRES_XMENOS17', 'NACIMIENTOS_MUJERES_XMENOS17',
    'NACIMIENTOS_TOTAL_XMENOS16', 'NACIMIENTOS_HOMBRES_XMENOS16', 'NACIMIENTOS_MUJERES_XMENOS16',
    'PIB_ANIO_ANTERIOR_MILLONES_COP', 'IPC_ANIO_ANTERIOR_PORC'
]
data_preparada[predictoras_numericas] = min_max_scaler.transform(data_preparada[predictoras_numericas])

# Predicción
Y_pred = modelo.predict(data_preparada)
matriculas_pred = int(round(Y_pred[0], 0))

# Resultado
st.markdown(f"""
    <div class="resultado-box">
        <div class="resultado-label">Total de matrículas estimadas</div>
        <div class="resultado-numero">{matriculas_pred:,}</div>
        <div class="resultado-label">estudiantes de pregrado</div>
    </div>
""", unsafe_allow_html=True)

st.warning('⚠️ El modelo tiene un MAE promedio de 1,766 matrículas y un MAPE de 5.14% (SVM, kernel RBF, C=10)')
