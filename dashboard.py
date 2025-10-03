# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import hmac

# Agregar path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))

# ▼▼▼ CORRECCIÓN DE LA RUTA DE IMPORTACIÓN ▼▼▼
from wine_scraper.utils.data_consolidator import DataConsolidator
from wine_scraper.utils.data_quality import DataQuality


# ===============================================
# SISTEMA DE AUTENTICACIÓN (sin cambios)
# ===============================================
def check_password():
    def login_form():
        st.markdown("## 🔐 Acceso al Dashboard")
        st.markdown("---")
        with st.form("login_form"):
            username = st.text_input("Usuario", key="username_input")
            password = st.text_input("Contraseña", type="password", key="password_input")
            submit = st.form_submit_button("Iniciar Sesión")
            if submit:
                users = st.secrets.get("passwords", {"admin": "admin123", "cliente": "cliente2024"})
                if username in users and hmac.compare_digest(password, users[username]):
                    st.session_state["password_correct"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("❌ Usuario o contraseña incorrectos")
    if st.session_state.get("password_correct", False):
        return True
    login_form()
    return False

if not check_password():
    st.stop()

with st.sidebar:
    st.markdown("---")
    if st.button("🚪 Cerrar Sesión"):
        st.session_state["password_correct"] = False
        st.session_state["username"] = None
        st.rerun()
    st.markdown(f"**Usuario:** {st.session_state.get('username', 'N/A')}")


# ===============================================
# DASHBOARD
# ===============================================
st.set_page_config(page_title="Wine Market Analysis", page_icon="🍷", layout="wide")
st.title("🍷 Análisis de Mercado de Vinos - México")
st.markdown("---")

# Cargar datos
@st.cache_data
def load_data():
    """Carga datos de la carpeta consolidada más reciente"""
    # La lógica para NO crear una nueva carpeta se mantiene correcta
    consolidator = DataConsolidator('./data', create_new_dir=False)
    
    latest_dir = consolidator.output_dir
    
    if not latest_dir or not latest_dir.exists():
        st.error("❌ No se encontró ninguna carpeta consolidada.")
        st.stop()
    
    filepath = latest_dir / 'datos_completos_listos.csv'
    
    if not filepath.exists():
        st.error(f"❌ No se encontró 'datos_completos_listos.csv' en la carpeta '{latest_dir.name}'.")
        st.stop()
    
    st.sidebar.success(f"📅 Datos de: {latest_dir.name}")
    return pd.read_csv(filepath)

df = load_data()
quality = DataQuality()

# El resto del código no necesita cambios
# ... (Sidebar, Filtros, Gráficas, etc.)
# Sidebar - Filtros (sin cambios)
st.sidebar.header("🔍 Filtros")
tiendas_seleccionadas = st.sidebar.multiselect("Tiendas", options=df['tienda'].unique(), default=df['tienda'].unique())
tipos_seleccionados = st.sidebar.multiselect("Tipo de Vino", options=df['tipo_vino'].unique(), default=df['tipo_vino'].unique())
precio_min = float(df['precio_actual'].min())
precio_max = float(df['precio_actual'].max())
rango_precio = st.sidebar.slider("Rango de Precio", min_value=precio_min, max_value=precio_max, value=(precio_min, precio_max))

# Aplicar filtros (sin cambios)
df_filtrado = df[
    (df['tienda'].isin(tiendas_seleccionadas)) &
    (df['tipo_vino'].isin(tipos_seleccionados)) &
    (df['precio_actual'].between(rango_precio[0], rango_precio[1]))
]

# El resto del código del dashboard se mantiene igual...
df_precios = quality.get_dataset_for_analysis(df_filtrado, 'precio')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Productos", len(df_filtrado))
with col2:
    st.metric("Precio Promedio", f"${df_precios['precio_actual'].mean():.2f}")
with col3:
    st.metric("Con Descuento", f"{(df_filtrado['tiene_descuento'].sum() / len(df_filtrado) * 100):.1f}%")
with col4:
    st.metric("Tiendas Activas", df_filtrado['tienda'].nunique())
st.markdown("---")

# SECCIÓN 1: Análisis de Precios (sin cambios)
st.header("💰 1. Análisis de Precios")
tab1, tab2, tab3 = st.tabs(["Distribución", "Por Tienda", "Por Tipo"])
with tab1:
    st.subheader("Distribución de Precios en el Mercado")
    fig = px.histogram(df_precios, x='precio_actual', nbins=50, title="¿Dónde se concentra la oferta?", labels={'precio_actual': 'Precio (MXN)', 'count': 'Cantidad de Productos'}, color_discrete_sequence=['#8B0000'])
    fig.add_vline(x=df_precios['precio_actual'].median(), line_dash="dash", annotation_text=f"Mediana: ${df_precios['precio_actual'].median():.2f}")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.subheader("Posicionamiento de Precios por Tienda")
    fig = px.box(df_precios, x='tienda', y='precio_actual', title="Rangos de Precio por Competidor", labels={'precio_actual': 'Precio (MXN)', 'tienda': 'Tienda'}, color='tienda')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Comparativa de Precios")
    comparativa = df_precios.groupby('tienda').agg({'precio_actual': ['mean', 'median', 'min', 'max', 'count']}).round(2)
    comparativa.columns = ['Promedio', 'Mediana', 'Mínimo', 'Máximo', 'Productos']
    st.dataframe(comparativa.sort_values('Promedio'), use_container_width=True)
with tab3:
    st.subheader("Precios por Tipo de Vino")
    precio_tipo = df_precios.groupby('tipo_vino')['precio_actual'].agg(['mean', 'count']).reset_index()
    precio_tipo = precio_tipo[precio_tipo['count'] > 5]
    fig = px.bar(precio_tipo, x='tipo_vino', y='mean', title="Precio Promedio por Tipo de Vino", labels={'mean': 'Precio Promedio (MXN)', 'tipo_vino': 'Tipo'}, color='mean', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# SECCIÓN 2: Análisis de Catálogo (sin cambios)
st.header("📚 2. Análisis de Catálogo")
df_catalogo = quality.get_dataset_for_analysis(df_filtrado, 'catalogo')
col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribución por País de Origen")
    pais_count = df_catalogo['pais_origen'].value_counts().head(10)
    fig = px.bar(x=pais_count.values, y=pais_count.index, orientation='h', title="Top 10 Países en el Mercado", labels={'x': 'Cantidad de Productos', 'y': 'País'}, color=pais_count.values, color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Distribución por Tipo de Vino")
    tipo_count = df_catalogo['tipo_vino'].value_counts()
    fig = px.pie(values=tipo_count.values, names=tipo_count.index, title="Composición del Mercado", color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# Footer
st.caption("🍷 Wine Market Analysis Dashboard")