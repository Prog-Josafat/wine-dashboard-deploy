# dashboard.py deploy
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import hmac

st.set_page_config(
    page_title="Wine Market Analysis",
    page_icon="🍷",
    layout="wide"
)

sys.path.insert(0, str(Path(__file__).parent))

from wine_scraper.utils.data_consolidator import DataConsolidator
from wine_scraper.utils.data_quality import DataQuality

def check_password():
    """Retorna True si el usuario ha iniciado sesión, de lo contrario muestra el formulario."""
    def login_form():
        st.markdown("## 🔐 Acceso al Dashboard")
        st.markdown("---")
        with st.form("login_form"):
            username = st.text_input("Usuario", key="username_input")
            password = st.text_input("Contraseña", type="password", key="password_input")
            submit = st.form_submit_button("Iniciar Sesión")
            if submit:
                users = st.secrets.get("passwords", {})
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


st.title("🍷 Análisis de Mercado de Vinos - México")
st.markdown("---")

@st.cache_data
def load_data():
    """Carga datos de la carpeta consolidada más reciente"""
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

st.sidebar.header("🔍 Filtros")

tiendas_seleccionadas = st.sidebar.multiselect(
    "Tiendas",
    options=df['tienda'].unique(),
    default=df['tienda'].unique()
)

tipos_seleccionados = st.sidebar.multiselect(
    "Tipo de Vino",
    options=df['tipo_vino'].unique(),
    default=df['tipo_vino'].unique()
)

precio_min = float(df['precio_actual'].min())
precio_max = float(df['precio_actual'].max())
rango_precio = st.sidebar.slider(
    "Rango de Precio",
    min_value=precio_min,
    max_value=precio_max,
    value=(precio_min, precio_max)
)

df_filtrado = df[
    (df['tienda'].isin(tiendas_seleccionadas)) &
    (df['tipo_vino'].isin(tipos_seleccionados)) &
    (df['precio_actual'].between(rango_precio[0], rango_precio[1]))
]

df_precios = quality.get_dataset_for_analysis(df_filtrado, 'precio')
df_catalogo = quality.get_dataset_for_analysis(df_filtrado, 'catalogo')

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

st.header("📚 2. Análisis de Catálogo")
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

st.header("💎 3. Oportunidades de Nicho")
st.markdown("**¿Qué combinaciones tienen poca oferta?** Estas son tus oportunidades.")

combinaciones = df_catalogo.groupby(['tipo_vino', 'pais_origen']).size().reset_index(name='cantidad')
pivot = combinaciones.pivot(index='tipo_vino', columns='pais_origen', values='cantidad').fillna(0)
fig = px.imshow(pivot, title="Mapa de Disponibilidad: Tipo de Vino x País", labels={'x': 'País', 'y': 'Tipo de Vino', 'color': 'Productos'}, color_continuous_scale='YlOrRd', aspect='auto')
st.plotly_chart(fig, use_container_width=True)

st.subheader("🎯 Top Oportunidades (Menos de 5 productos)")
oportunidades = combinaciones[combinaciones['cantidad'] < 5].sort_values('cantidad')
if len(oportunidades) > 0:
    st.dataframe(oportunidades, use_container_width=True)
    st.markdown("**💡 Estrategia:** Estas combinaciones tienen poca competencia. Si puedes conseguir proveedores de estos nichos, tendrás ventaja competitiva.")
else:
    st.info("No hay brechas evidentes en el rango de filtros seleccionado")
st.markdown("---")

st.header("🏪 4. Matriz de Competitividad")
competitividad = df_precios.groupby('tienda').agg({'precio_actual': 'mean', 'nombre': 'count', 'tiene_descuento': 'sum', 'pais_origen': 'nunique'}).reset_index()
competitividad.columns = ['Tienda', 'Precio_Promedio', 'SKUs', 'Con_Descuento', 'Paises']
competitividad['%_Descuento'] = (competitividad['Con_Descuento'] / competitividad['SKUs'] * 100).round(1)
fig = px.scatter(competitividad, x='Precio_Promedio', y='SKUs', size='Paises', color='%_Descuento', text='Tienda', title="Matriz de Posicionamiento Competitivo", labels={'Precio_Promedio': 'Precio Promedio (MXN)', 'SKUs': 'Variedad de Catálogo (SKUs)', '%_Descuento': '% Productos con Descuento'}, color_continuous_scale='RdYlGn_r')
fig.update_traces(textposition='top center')
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
**💡 Interpretación:**
- **Eje X (Precio):** Izquierda = Económico, Derecha = Premium
- **Eje Y (SKUs):** Arriba = Mayor variedad
- **Tamaño:** Más grande = Mayor diversidad geográfica
- **Color:** Rojo = Muchos descuentos, Verde = Precios estables
""")
st.subheader("📊 Índice de Competitividad")
st.dataframe(competitividad.sort_values('SKUs', ascending=False), use_container_width=True)
st.markdown("---")

st.header("🎯 5. Recomendaciones Estratégicas")
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Basado en Precios")
    precio_promedio_mercado = df_precios['precio_actual'].mean()
    precio_mediana = df_precios['precio_actual'].median()
    st.write(f"**Precio promedio del mercado:** ${precio_promedio_mercado:.2f}")
    st.write(f"**Precio mediano:** ${precio_mediana:.2f}")
    st.markdown("""
    **Sugerencia de posicionamiento:**
    - Para estrategia de valor: Precio ~5% bajo la mediana
    - Para estrategia premium: Precio ~15% sobre el promedio
    - Para estrategia de volumen: Precio en el percentil 25
    """)
with col2:
    st.subheader("📚 Basado en Catálogo")
    top_paises = df_catalogo['pais_origen'].value_counts().head(5).index.tolist()
    top_tipos = df_catalogo['tipo_vino'].value_counts().head(3).index.tolist()
    st.markdown(f"""
    **Productos esenciales (alta demanda):**
    - Países: {', '.join(top_paises)}
    - Tipos: {', '.join(top_tipos)}
    
    **Para diferenciarte:** Busca los nichos en la sección anterior.
    """)
st.markdown("---")

st.caption("🍷 Wine Market Analysis Dashboard | Datos actualizados: " + df['fecha_scraping'].max())