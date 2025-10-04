# dashboard_deploy.py (Deploy - Versión Final con TODAS las Etiquetas Corregidas)
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import hmac

# Configuración de página
st.set_page_config(page_title="Wine Market Analysis", page_icon="🍷", layout="wide")

# Agregar path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))
from wine_scraper.utils import DataQuality, DataConsolidator

# --- Sistema de Autenticación ---
def check_password():
    def login_form():
        st.markdown("## 🔐 Acceso al Dashboard")
        st.markdown("---")
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submit = st.form_submit_button("Iniciar Sesión")
            if submit:
                users = st.secrets.get("passwords", {})
                if username in users and hmac.compare_digest(password, users[username]):
                    st.session_state["password_correct"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("❌ Usuario o contraseña incorrectos")
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if st.session_state.password_correct:
        return True
    login_form()
    return False

if not check_password():
    st.stop()

# --- Comienza el Dashboard ---
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
    consolidator = DataConsolidator('./data', create_new_dir=False)
    latest_dir = consolidator.get_latest_consolidated_dir()
    if not latest_dir:
        st.error("❌ No se encontró ninguna carpeta consolidada.")
        st.stop()
    filepath = latest_dir / 'datos_completos_listos.csv'
    if not filepath.exists():
        st.error(f"❌ No se encontró 'datos_completos_listos.csv' en {latest_dir.name}.")
        st.stop()
    st.sidebar.success(f"📅 Datos de: {latest_dir.name}")
    return pd.read_csv(filepath)

df = load_data()
quality = DataQuality()

# --- Sidebar - Filtros ---
st.sidebar.header("🔍 Filtros")
tiendas_seleccionadas = st.sidebar.multiselect("Tiendas", options=df['tienda'].unique(), default=df['tienda'].unique())
tipos_seleccionados = st.sidebar.multiselect("Tipo de Vino", options=df['tipo_vino'].unique(), default=df['tipo_vino'].unique())
precio_min = float(df['precio_actual'].min())
precio_max = float(df['precio_actual'].max())
rango_precio = st.sidebar.slider("Rango de Precio", min_value=precio_min, max_value=precio_max, value=(precio_min, precio_max))

df_filtrado_temp = df[(df['precio_actual'].between(rango_precio[0], rango_precio[1]))]
paises_disponibles = sorted(df_filtrado_temp['pais_origen'].unique())
paises_seleccionados = st.sidebar.multiselect("País de Origen", options=paises_disponibles, default=paises_disponibles)

segmentos_disponibles = sorted(df_filtrado_temp['segmento_precio'].unique())
segmentos_seleccionados = st.sidebar.multiselect("Segmento de Precio", options=segmentos_disponibles, default=segmentos_disponibles)

# --- Lógica de Filtrado Completa ---
df_filtrado = df[
    (df['tienda'].isin(tiendas_seleccionadas)) &
    (df['tipo_vino'].isin(tipos_seleccionados)) &
    (df['precio_actual'].between(rango_precio[0], rango_precio[1])) &
    (df['pais_origen'].isin(paises_seleccionados)) &
    (df['segmento_precio'].isin(segmentos_seleccionados))
]
df_precios = quality.get_dataset_for_analysis(df_filtrado, 'precio')
df_catalogo = quality.get_dataset_for_analysis(df_filtrado, 'catalogo')

# KPIs principales
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Productos Filtrados", len(df_filtrado))
with col2: st.metric("Precio Promedio", f"${df_precios['precio_actual'].mean():.2f}")
with col3: st.metric("Con Descuento", f"{(df_filtrado['tiene_descuento'].sum() / len(df_filtrado) * 100 if len(df_filtrado) > 0 else 0):.1f}%")
with col4: st.metric("Tiendas Activas", df_filtrado['tienda'].nunique())
st.markdown("---")

# SECCIÓN 1: Análisis de Precios
st.header("💰 1. Análisis de Precios")
tab1, tab2, tab3 = st.tabs(["Distribución", "Por Tienda", "Por Tipo"])
with tab1:
    fig = px.histogram(df_precios, x='precio_actual', nbins=50, title="¿Dónde se concentra la oferta?",
                     labels={'precio_actual': 'Precio Actual (MXN)', 'count': 'Cantidad de Productos'},
                     color_discrete_sequence=['#8B0000'])
    fig.add_vline(x=df_precios['precio_actual'].median(), line_dash="dash", annotation_text=f"Mediana: ${df_precios['precio_actual'].median():.2f}")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    fig = px.box(df_precios, x='tienda', y='precio_actual', title="Rangos de Precio por Competidor",
                 labels={'precio_actual': 'Precio Actual (MXN)', 'tienda': 'Tienda'},
                 color='tienda')
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    precio_tipo = df_precios.groupby('tipo_vino')['precio_actual'].agg(['mean', 'count']).reset_index()
    precio_tipo = precio_tipo[precio_tipo['count'] > 5]
    fig = px.bar(precio_tipo, x='tipo_vino', y='mean', title="Precio Promedio por Tipo de Vino",
                 labels={'mean': 'Precio Promedio (MXN)', 'tipo_vino': 'Tipo de Vino'},
                 color='mean', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# SECCIÓN 2: Análisis de Catálogo
st.header("📚 2. Análisis de Catálogo")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribución por País de Origen")
    pais_count = df_catalogo['pais_origen'].value_counts().head(10)
    pais_df = pais_count.reset_index()
    pais_df.columns = ['País', 'Cantidad']
    fig = px.bar(pais_df, x='Cantidad', y='País', orientation='h', title="Top 10 Países en el Mercado",
                 labels={'Cantidad': 'Cantidad de Productos', 'País': 'País de Origen'},
                 color='Cantidad', color_continuous_scale='Reds', text='Cantidad')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Distribución por Tipo de Vino")
    tipo_count = df_catalogo['tipo_vino'].value_counts()
    fig = px.pie(values=tipo_count.values, names=tipo_count.index, title="Composición del Mercado por Tipo de Vino",
                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# SECCIÓN 3: Oportunidades de Nicho
st.header("💎 3. Oportunidades de Nicho")
combinaciones = df_catalogo.groupby(['tipo_vino', 'pais_origen']).size().reset_index(name='cantidad')
pivot = combinaciones.pivot(index='tipo_vino', columns='pais_origen', values='cantidad').fillna(0)
fig = px.imshow(pivot, title="Mapa de Disponibilidad: Tipo de Vino vs. País de Origen",
                labels={'x': 'País de Origen', 'y': 'Tipo de Vino', 'color': 'Cantidad de Productos'},
                color_continuous_scale='YlOrRd', aspect='auto')
st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

# SECCIÓN 4: Mapa de Competitividad
st.header("🏪 4. Mapa de Competitividad")
df_competencia = df_filtrado.groupby('tienda').agg(num_vinos=('nombre', 'count'), precio_promedio=('precio_actual', 'mean')).reset_index()
fig_competidores = px.scatter(df_competencia, x='num_vinos', y='precio_promedio', size='num_vinos', color='precio_promedio', text='tienda',
                              title='Posicionamiento de Tiendas: Catálogo vs. Precio',
                              labels={'num_vinos': 'Amplitud de Catálogo (No. de Vinos)', 'precio_promedio': 'Precio Promedio (MXN)'},
                              color_continuous_scale='RdYlGn_r', size_max=60)
fig_competidores.update_traces(textposition='top center')
st.plotly_chart(fig_competidores, use_container_width=True)
st.markdown("---")

# SECCIÓN 5: RECOMENDACIONES ESTRATÉGICAS
st.header("🎯 5. Recomendaciones Estratégicas Dinámicas")
if not df_filtrado.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Oportunidades de Precio y Formato")
        precio_p25, precio_p75 = df_precios['precio_actual'].quantile(0.25), df_precios['precio_actual'].quantile(0.75)
        st.metric(label="Rango de Precios Clave (Sweet Spot)", value=f"${precio_p25:,.2f} - ${precio_p75:,.2f}")
        st.markdown(f"**Recomendación:** El 50% de los vinos se encuentra en este rango. Posicionar productos aquí asegura competir en el segmento más grande.")
        
        df_tamanos_realistas = df_precios[df_precios['ml_botella'] < 5000]
        if not df_tamanos_realistas.empty:
            moda_tamano = df_tamanos_realistas['ml_botella'].mode().iloc[0]
            media_tamano = df_tamanos_realistas['ml_botella'].mean()
            mediana_tamano = df_tamanos_realistas['ml_botella'].median()
            st.metric(label="Tamaño de Botella Más Común (Moda)", value=f"{int(moda_tamano)} ml")
            st.markdown(f"**Recomendación:** El formato de **{int(moda_tamano)} ml** es el estándar de facto en el mercado. Tu catálogo debe tener una fuerte presencia de este tamaño.")
            st.caption(f"Promedio: {media_tamano:.0f} ml | Mediana: {int(mediana_tamano)} ml")
        else:
            st.info("No hay datos de tamaño de botella para analizar.")
            
    with col2:
        st.subheader("📚 Oportunidades de Catálogo")
        oportunidades = df_catalogo.groupby(['tipo_vino', 'pais_origen']).size().reset_index(name='cantidad')
        oportunidades = oportunidades[oportunidades['cantidad'] < 5].sort_values('cantidad')
        if not oportunidades.empty:
            top_oportunidad = oportunidades.iloc[0]
            st.metric(label="Nicho con Menor Competencia", value=f"{top_oportunidad['tipo_vino']} de {top_oportunidad['pais_origen']}")
            st.markdown(f"**Recomendación:** Existe una baja oferta para esta combinación. Explorar proveedores para este nicho podría darte una ventaja competitiva.")
        else:
            st.info("No se detectaron nichos claros.")
        try:
            uva_dominante = df_catalogo[~df_catalogo['uva_varietal'].isin(['No especificado', 'Tinto', 'Blanco'])].uva_varietal.mode()[0]
            st.metric(label="Uva Más Popular del Mercado", value=uva_dominante)
            st.markdown(f"**Recomendación:** Asegúrate de tener una sólida oferta de vinos **{uva_dominante}**, ya que es la uva con mayor presencia.")
        except IndexError:
            st.info("No hay una uva dominante con los filtros actuales.")
else:
    st.warning("No hay datos con los filtros seleccionados para generar recomendaciones.")
st.markdown("---")

# SECCIÓN 6: CUOTA DE MERCADO POR UVA (TREEMAP)
st.header("🍇 6. Cuota de Mercado por Tipo de Uva")
df_uvas_treemap = df_filtrado[~df_filtrado['uva_varietal'].isin(['No especificado', 'Tinto', 'Blanco'])].copy()
df_uvas_treemap = df_uvas_treemap['uva_varietal'].value_counts().nlargest(20).reset_index()
df_uvas_treemap.columns = ['uva_varietal', 'cantidad']
fig = px.treemap(df_uvas_treemap, path=['uva_varietal'], values='cantidad', title='Distribución del Catálogo por Tipo de Uva (Top 20)',
                 color='cantidad', color_continuous_scale='Reds',
                 labels={'cantidad': 'Cantidad de Vinos'})
st.plotly_chart(fig, use_container_width=True)
st.markdown("**Insight:** Los rectángulos más grandes representan las uvas con mayor dominancia en el mercado. Úsalo para balancear tu inventario.")
st.markdown("---")

# SECCIÓN 7: DENSIDAD DE PRECIOS POR COMPETIDOR (VIOLIN PLOT)
st.header("🎻 7. Densidad de Precios por Competidor")
fig = px.violin(df_filtrado, x='tienda', y='precio_actual', color='tienda', box=True,
                title='Distribución y Densidad de Precios por Tienda',
                labels={'precio_actual': 'Precio Actual (MXN)', 'tienda': 'Tienda'})
st.plotly_chart(fig, use_container_width=True)
st.markdown("**Insight:** La parte ancha del 'violín' indica dónde se concentra la mayor cantidad de vinos de una tienda. Un violín ancho y corto significa una estrategia de precios muy enfocada.")
st.markdown("---")

# SECCIÓN 8: ANÁLISIS DE DESCUENTOS (STACKED BAR)
st.header("📊 8. Actividad de Descuentos por Categoría de Vino")
df_descuentos = df_filtrado.groupby(['tipo_vino', 'tiene_descuento']).size().reset_index(name='cantidad')
df_descuentos['tiene_descuento'] = df_descuentos['tiene_descuento'].map({True: 'Con Descuento', False: 'Sin Descuento'})
fig = px.bar(df_descuentos, x='tipo_vino', y='cantidad', color='tiene_descuento',
             title='Número de Vinos con y sin Descuento por Tipo',
             labels={'tipo_vino': 'Tipo de Vino', 'cantidad': 'Número de Productos', 'tiene_descuento': 'Condición'},
             color_discrete_map={'Con Descuento': '#E53935', 'Sin Descuento': '#BDBDBD'},
             text='cantidad')
fig.update_traces(textposition='inside', textfont_size=12)
st.plotly_chart(fig, use_container_width=True)
st.markdown("**Insight:** Observa qué categorías tienen una barra roja más grande. Esto puede indicar alta competencia o una estrategia para atraer volumen en ese segmento.")

# --- Footer ---
st.markdown("---")
st.caption("🍷 Wine Market Analysis Dashboard | Datos actualizados: " + df['fecha_scraping'].max())