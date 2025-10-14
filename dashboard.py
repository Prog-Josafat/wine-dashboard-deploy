# dashboard_deploy.py (Deploy)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
import hmac
from glob import glob

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wine_scraper.utils import DataQuality, DataConsolidator
except ImportError:
    st.warning("No se encontraron 'DataQuality' y 'DataConsolidator'. Usando clases de ejemplo.")
    class DataQuality:
        def get_dataset_for_analysis(self, df, analysis_type): return df.copy()
    class DataConsolidator:
        def __init__(self, base_path, create_new_dir=False): self.base_path = Path(base_path); self.consolidated_path = self.base_path / 'consolidated'
        def get_latest_consolidated_dir(self):
            if not self.consolidated_path.exists(): return None
            date_dirs = [p for p in self.consolidated_path.iterdir() if p.is_dir() and p.name.isdigit()]
            if not date_dirs: return None
            return max(date_dirs, key=lambda p: p.name)

def get_available_dates(base_path='./data/consolidated'):
    if not Path(base_path).exists(): return []
    date_dirs = [Path(p).name for p in glob(f"{base_path}/*") if Path(p).is_dir()]
    return sorted([d for d in date_dirs if d.isdigit()], reverse=True)

st.set_page_config(page_title="Wine Market Analysis", page_icon="🍷", layout="wide")

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

with st.sidebar:
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
    if not latest_dir: st.error("❌ No se encontró carpeta de datos en './data/consolidated/'."); st.stop()
    filepath = latest_dir / 'datos_completos_listos.csv'
    if not filepath.exists(): st.error(f"❌ No se encontró 'datos_completos_listos.csv' en '{latest_dir.name}'."); st.stop()
    st.sidebar.success(f"📅 Datos de: {latest_dir.name}")
    return pd.read_csv(filepath)

df = load_data()
quality = DataQuality()

st.sidebar.markdown("---")

alertas = []
if 'precio_anterior' in df.columns:
    df_alertas = df.dropna(subset=['precio_actual', 'precio_anterior'])
    df_alertas = df_alertas[df_alertas['precio_anterior'] > 0]
    aumentos = df_alertas[((df_alertas['precio_actual'] - df_alertas['precio_anterior']) / df_alertas['precio_anterior']) > 0.2]
    if len(aumentos) > 0: alertas.append({'tipo': 'precio', 'nivel': 'warning', 'icono': '📈', 'mensaje': f"{len(aumentos)} productos con aumento >20%", 'detalle': f"Promedio: {aumentos['precio_actual'].mean() - aumentos['precio_anterior'].mean():.2f} MXN"})
    reducciones = df_alertas[((df_alertas['precio_anterior'] - df_alertas['precio_actual']) / df_alertas['precio_anterior']) > 0.2]
    if len(reducciones) > 0: alertas.append({'tipo': 'precio', 'nivel': 'success', 'icono': '📉', 'mensaje': f"{len(reducciones)} productos con reducción >20%", 'detalle': f"Promedio: {reducciones['precio_anterior'].mean() - reducciones['precio_actual'].mean():.2f} MXN"})

combinaciones = df.groupby(['tipo_vino', 'pais_origen']).size().reset_index(name='cantidad')
nichos_criticos = combinaciones[combinaciones['cantidad'] < 3]
if len(nichos_criticos) > 0: alertas.append({'tipo': 'oportunidad', 'nivel': 'success', 'icono': '💎', 'mensaje': f"{len(nichos_criticos)} nichos con <3 competidores", 'detalle': "Oportunidad de entrada con baja competencia"})

Q1, Q3 = df['precio_actual'].quantile(0.25), df['precio_actual'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['precio_actual'] < Q1 - 1.5 * IQR) | (df['precio_actual'] > Q3 + 1.5 * IQR)]
if len(outliers) > 0: alertas.append({'tipo': 'precio', 'nivel': 'warning', 'icono': '⚠️', 'mensaje': f"{len(outliers)} productos con precio atípico", 'detalle': "Pueden ser errores o productos premium/liquidación"})

if 'tiene_descuento' in df.columns:
    descuentos_tienda = df.groupby('tienda').agg(total=('nombre', 'count'), con_descuento=('tiene_descuento', 'sum'))
    descuentos_tienda['%_descuento'] = (descuentos_tienda['con_descuento'] / descuentos_tienda['total']) * 100
    tiendas_alto_descuento = descuentos_tienda[descuentos_tienda['%_descuento'] > 30]
    if len(tiendas_alto_descuento) > 0:
        for tienda in tiendas_alto_descuento.index:
            pct = tiendas_alto_descuento.loc[tienda, '%_descuento']
            alertas.append({'tipo': 'inventario', 'nivel': 'warning', 'icono': '🏷️', 'mensaje': f"{tienda}: {pct:.1f}% en descuento", 'detalle': "Posible problema de rotación o estrategia agresiva"})

segmentos_esperados = ['Económico', 'Medio-Bajo', 'Medio', 'Medio-Alto', 'Premium']
segmentos_faltantes = set(segmentos_esperados) - set(df['segmento_precio'].unique())
if len(segmentos_faltantes) > 0: alertas.append({'tipo': 'catalogo', 'nivel': 'info', 'icono': '📊', 'mensaje': f"Segmentos sin cobertura: {', '.join(segmentos_faltantes)}", 'detalle': "Considera ampliar el catálogo"})

if 'calidad_datos' in df.columns:
    tiendas_baja_calidad = df[df['calidad_datos'] == 'parcial'].groupby('tienda').size()
    if len(tiendas_baja_calidad) > 0:
        total_tienda = df.groupby('tienda').size()
        pct_baja = (tiendas_baja_calidad / total_tienda * 100).round(1)
        for tienda, pct in pct_baja.items():
            if pct > 20: alertas.append({'tipo': 'datos', 'nivel': 'error', 'icono': '❌', 'mensaje': f"{tienda}: {pct}% datos incompletos", 'detalle': "Revisar scraper o estructura del sitio"})

productos_sin_origen = df[df['pais_origen'].isin(['No especificado', None])].shape[0]
if productos_sin_origen > 0:
    pct_sin_origen = (productos_sin_origen / len(df)) * 100
    alertas.append({'tipo': 'datos', 'nivel': 'warning', 'icono': '🌍', 'mensaje': f"{pct_sin_origen:.1f}% productos sin origen", 'detalle': "Dificulta análisis de mercado por región"})

cuota_tienda = df['tienda'].value_counts(normalize=True) * 100
if not cuota_tienda.empty:
    tienda_dominante = cuota_tienda.iloc[0]
    if tienda_dominante > 40: alertas.append({'tipo': 'mercado', 'nivel': 'info', 'icono': '🏛️', 'mensaje': f"{cuota_tienda.index[0]} domina ({tienda_dominante:.1f}%)", 'detalle': "Alta concentración de mercado"})

precio_tipo = df.groupby('tipo_vino')['precio_actual'].mean().sort_values(ascending=False)
if len(precio_tipo) > 0 and precio_tipo.iloc[0] > df['precio_actual'].mean() * 2:
    tipo_caro = precio_tipo.index[0]
    precio_promedio = precio_tipo.iloc[0]
    alertas.append({'tipo': 'mercado', 'nivel': 'info', 'icono': '💰', 'mensaje': f"{tipo_caro}: precio 2x > promedio", 'detalle': f"Promedio: ${precio_promedio:,.2f} MXN - Segmento premium"})

if 'pais_origen' in df.columns:
    paises_precio = df.groupby('pais_origen').agg(cantidad=('nombre', 'count'), precio_promedio=('precio_actual', 'mean'))
    paises_nicho = paises_precio[(paises_precio['cantidad'] < 10) & (paises_precio['precio_promedio'] > df['precio_actual'].median())]
    if len(paises_nicho) > 0:
        for pais in paises_nicho.index[:3]: alertas.append({'tipo': 'oportunidad', 'nivel': 'success', 'icono': '🌟', 'mensaje': f"Nicho premium: {pais}", 'detalle': f"Bajo volumen ({paises_nicho.loc[pais, 'cantidad']} prod.) pero alto precio (${paises_nicho.loc[pais, 'precio_promedio']:,.0f})"})

if 'uva_varietal' in df.columns:
    uvas_validas = df[~df['uva_varietal'].isin(['No especificado', 'Tinto', 'Blanco'])]
    if len(uvas_validas) > 0:
        uvas_count = uvas_validas['uva_varietal'].value_counts()
        uvas_raras = uvas_count[uvas_count < 5]
        if len(uvas_raras) > 0: alertas.append({'tipo': 'catalogo', 'nivel': 'info', 'icono': '🍇', 'mensaje': f"{len(uvas_raras)} uvas con <5 productos", 'detalle': "Oportunidad de diferenciación con uvas poco comunes"})

st.sidebar.subheader(f"🔔 Panel de Alertas ({len(alertas)})")
if alertas:
    criticas = [a for a in alertas if a['nivel'] == 'error']
    advertencias = [a for a in alertas if a['nivel'] == 'warning']
    oportunidades = [a for a in alertas if a['nivel'] == 'success' or a['tipo'] == 'oportunidad']
    info = [a for a in alertas if a['nivel'] == 'info' and a['tipo'] != 'oportunidad']
    if criticas:
        with st.sidebar.expander(f"**🔴 Críticas ({len(criticas)})**"):
            for alerta in criticas: st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**"); st.caption(f"_{alerta['detalle']}_")
    if advertencias:
        with st.sidebar.expander(f"**🟡 Advertencias ({len(advertencias)})**"):
            for alerta in advertencias: st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**"); st.caption(f"_{alerta['detalle']}_")
    if oportunidades:
        with st.sidebar.expander(f"**🟢 Oportunidades ({len(oportunidades)})**"):
            for alerta in oportunidades: st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**"); st.caption(f"_{alerta['detalle']}_")
    if info:
        with st.sidebar.expander(f"**🔵 Información ({len(info)})**"):
            for alerta in info: st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**"); st.caption(f"_{alerta['detalle']}_")
else:
    st.sidebar.success("✅ Sin alertas pendientes")

st.sidebar.markdown("---")

st.sidebar.header("🔍 Filtros de Mercado")
with st.sidebar.expander("🏪 Tiendas"):
    tiendas_seleccionadas = st.multiselect("Selecciona Tiendas", options=df['tienda'].unique(), default=df['tienda'].unique(), label_visibility="collapsed")
with st.sidebar.expander("🍷 Tipo de Vino"):
    tipos_seleccionados = st.multiselect("Selecciona Tipos de Vino", options=df['tipo_vino'].unique(), default=df['tipo_vino'].unique(), label_visibility="collapsed")
with st.sidebar.expander("🌍 País de Origen"):
    paises_disponibles = sorted(df['pais_origen'].dropna().unique())
    paises_seleccionados = st.multiselect("Selecciona Países de Origen", options=paises_disponibles, default=paises_disponibles, label_visibility="collapsed")
with st.sidebar.expander("💰 Segmento de Precio"):
    segmentos_disponibles = sorted(df['segmento_precio'].dropna().unique())
    segmentos_seleccionados = st.multiselect("Selecciona Segmentos de Precio", options=segmentos_disponibles, default=segmentos_disponibles, label_visibility="collapsed")
with st.sidebar.expander("💲 Rango de Precio (MXN)"):
    precio_min, precio_max = float(df['precio_actual'].min()), float(df['precio_actual'].max())
    rango_precio = st.slider("Selecciona un Rango de Precio", min_value=precio_min, max_value=precio_max, value=(precio_min, precio_max), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.header("🔬 Análisis Comparativo Temporal")
available_dates = get_available_dates()
if len(available_dates) >= 2:
    date1 = st.sidebar.selectbox("Fecha base (anterior):", available_dates, index=1)
    date2 = st.sidebar.selectbox("Fecha a comparar (nueva):", available_dates, index=0)
    run_comparison = st.sidebar.button("📊 Comparar Periodos")
else:
    st.sidebar.info("Se necesitan al menos dos carpetas de datos para poder comparar.")
    run_comparison = False

df_filtrado = df[(df['tienda'].isin(tiendas_seleccionadas)) & (df['tipo_vino'].isin(tipos_seleccionados)) & (df['precio_actual'].between(rango_precio[0], rango_precio[1])) & (df['pais_origen'].isin(paises_seleccionados)) & (df['segmento_precio'].isin(segmentos_seleccionados))]
if df_filtrado.empty: st.error("No hay datos que coincidan con los filtros. Amplía tu selección."); st.stop()
df_precios = quality.get_dataset_for_analysis(df_filtrado, 'precio')
df_catalogo = quality.get_dataset_for_analysis(df_filtrado, 'catalogo')

col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Productos Filtrados", f"{len(df_filtrado):,}")
with col2: st.metric("Precio Promedio", f"${df_precios['precio_actual'].mean():,.2f}")
with col3: st.metric("Con Descuento", f"{df_filtrado['tiene_descuento'].mean() * 100:.1f}%")
with col4: st.metric("Tiendas Activas", df_filtrado['tienda'].nunique())
st.markdown("---")


st.header("💰 1. Análisis de Precios")
tab1, tab2, tab3 = st.tabs(["Distribución", "Por Tienda", "Por Tipo"])
with tab1:
    fig = px.histogram(df_precios, x='precio_actual', nbins=50, title="¿Dónde se concentra la oferta?",
                       labels={'precio_actual': 'Precio Actual (MXN)', 'count': 'Cantidad de Productos'},
                       color_discrete_sequence=['#8B0000'])
    fig.add_vline(x=df_precios['precio_actual'].median(), line_dash="dash", annotation_text=f"Mediana: ${df_precios['precio_actual'].median():,.2f}")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    fig = px.box(df_precios, x='tienda', y='precio_actual', title="Rangos de Precio por Competidor",
                 labels={'precio_actual': 'Precio Actual (MXN)', 'tienda': 'Tienda'},
                 color='tienda')
    st.plotly_chart(fig, use_container_width=True)
with tab3:
    precio_tipo = df_precios.groupby('tipo_vino')['precio_actual'].agg(['mean', 'count']).reset_index()
    precio_tipo = precio_tipo[precio_tipo['count'] > 5].sort_values('mean', ascending=False)
    fig = px.bar(precio_tipo, x='tipo_vino', y='mean', title="Precio Promedio por Tipo de Vino (con más de 5 productos)",
                 labels={'mean': 'Precio Promedio (MXN)', 'tipo_vino': 'Tipo de Vino'},
                 color='mean', color_continuous_scale='Reds', text='mean')
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

st.header("📚 2. Análisis de Catálogo")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribución por País de Origen")
    pais_count = df_catalogo['pais_origen'].value_counts().head(10)
    fig = px.bar(pais_count, y=pais_count.index, x=pais_count.values, orientation='h', title="Top 10 Países en el Mercado",
                 labels={'x': 'Cantidad de Productos', 'y': 'País de Origen'},
                 color=pais_count.values, color_continuous_scale='Reds', text=pais_count.values)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Distribución por Tipo de Vino")
    tipo_count = df_catalogo['tipo_vino'].value_counts()
    fig = px.pie(values=tipo_count.values, names=tipo_count.index, title="Composición del Mercado por Tipo de Vino",
                 color_discrete_sequence=px.colors.sequential.RdBu, hole=0.3)
    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

st.header("💎 3. Oportunidades de Nicho")
combinaciones = df_catalogo.groupby(['tipo_vino', 'pais_origen']).size().reset_index(name='cantidad')
pivot = combinaciones.pivot_table(index='tipo_vino', columns='pais_origen', values='cantidad').fillna(0)
fig = px.imshow(pivot, title="Mapa de Calor: Tipo de Vino vs. País de Origen",
                labels={'x': 'País de Origen', 'y': 'Tipo de Vino', 'color': 'Cantidad de Productos'},
                color_continuous_scale='YlOrRd', aspect='auto')
st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

st.header("🏪 4. Mapa de Competitividad")
df_competencia = df_filtrado.groupby('tienda').agg(num_vinos=('nombre', 'count'), precio_promedio=('precio_actual', 'mean')).reset_index()
fig_competidores = px.scatter(df_competencia, x='num_vinos', y='precio_promedio', size='num_vinos', color='precio_promedio', text='tienda',
                              title='Posicionamiento de Tiendas: Catálogo vs. Precio',
                              labels={'num_vinos': 'Amplitud de Catálogo (No. de Vinos)', 'precio_promedio': 'Precio Promedio (MXN)'},
                              color_continuous_scale='RdYlGn_r', size_max=60)
fig_competidores.update_traces(textposition='top center')
st.plotly_chart(fig_competidores, use_container_width=True)
st.markdown("---")

st.header("🎯 5. Recomendaciones Estratégicas Dinámicas")
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Oportunidades de Precio y Formato")
    precio_p25, precio_p75 = df_precios['precio_actual'].quantile(0.25), df_precios['precio_actual'].quantile(0.75)
    st.metric(label="Rango de Precios Clave (Sweet Spot)", value=f"${precio_p25:,.2f} - ${precio_p75:,.2f}")
    st.markdown(f"**Recomendación:** El 50% de los vinos se encuentra en este rango. Posicionar productos aquí asegura competir en el segmento más grande.")
    
    if 'ml_botella' in df_precios.columns:
        df_tamanos_realistas = df_precios[df_precios['ml_botella'] < 5000]
        if not df_tamanos_realistas.empty:
            moda_tamano = df_tamanos_realistas['ml_botella'].mode().iloc[0]
            st.metric(label="Tamaño de Botella Más Común (Moda)", value=f"{int(moda_tamano)} ml")
            st.markdown(f"**Recomendación:** El formato de **{int(moda_tamano)} ml** es el estándar de facto. Tu catálogo debe tener una fuerte presencia de este tamaño.")
        else:
            st.info("No hay datos de tamaño de botella para analizar.")
    else:
        st.info("La columna 'ml_botella' no se encuentra en los datos.")
        
with col2:
    st.subheader("📚 Oportunidades de Catálogo")
    oportunidades = combinaciones[combinaciones['cantidad'] < 5].sort_values('cantidad')
    if not oportunidades.empty:
        top_oportunidad = oportunidades.iloc[0]
        st.metric(label="Nicho con Menor Competencia", value=f"{top_oportunidad['tipo_vino']} de {top_oportunidad['pais_origen']}")
        st.markdown(f"**Recomendación:** Existe una baja oferta para esta combinación. Explorar proveedores para este nicho podría darte una ventaja competitiva.")
    else:
        st.info("No se detectaron nichos claros.")
    try:
        if 'uva_varietal' in df_catalogo.columns:
            uva_dominante = df_catalogo[~df_catalogo['uva_varietal'].isin(['No especificado', 'Tinto', 'Blanco'])].uva_varietal.mode()[0]
            st.metric(label="Uva Más Popular del Mercado", value=uva_dominante)
            st.markdown(f"**Recomendación:** Asegúrate de tener una sólida oferta de vinos **{uva_dominante}**, ya que es la uva con mayor presencia.")
    except IndexError:
        st.info("No hay una uva dominante con los filtros actuales.")
st.markdown("---")

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

st.header("🎻 7. Densidad de Precios por Competidor")
fig = px.violin(df_filtrado, x='tienda', y='precio_actual', color='tienda', box=True,
                title='Distribución y Densidad de Precios por Tienda',
                labels={'precio_actual': 'Precio Actual (MXN)', 'tienda': 'Tienda'})
st.plotly_chart(fig, use_container_width=True)
st.markdown("**Insight:** La parte ancha del 'violín' indica dónde se concentra la mayor cantidad de vinos de una tienda. Un violín ancho y corto significa una estrategia de precios muy enfocada.")
st.markdown("---")

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
st.markdown("---")

st.header("💰 9. Simulador de Rentabilidad")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Configuración de Costos")
    margen_objetivo = st.slider("Margen de ganancia objetivo (%)", 10, 100, 40)
    costo_operativo = st.number_input("Costo operativo por botella (MXN)", 10, 200, 50)

with col2:
    st.subheader("Precio de Venta Sugerido")

    precio_competencia = df_precios.groupby('tipo_vino')['precio_actual'].median()
    
    for tipo in precio_competencia.index[:5]:
        precio_mercado = precio_competencia[tipo]
        precio_sugerido = precio_mercado * 0.95 
        
        costo_total = costo_operativo
        ganancia = precio_sugerido - costo_total
        margen_real = (ganancia / precio_sugerido) * 100 if precio_sugerido > 0 else 0
        
        st.metric(
            label=f"{tipo}",
            value=f"${precio_sugerido:.2f}",
            delta=f"{margen_real:.1f}% margen"
        )

st.markdown("---")

st.header("🎯 10. Índice de Saturación del Mercado")

saturacion = df_catalogo.groupby(['tipo_vino', 'pais_origen']).agg({
    'tienda': 'nunique',
    'nombre': 'count'
}).reset_index()

saturacion.columns = ['tipo_vino', 'pais_origen', 'tiendas', 'productos']
saturacion['indice_saturacion'] = saturacion['productos'] / saturacion['tiendas']
saturacion = saturacion.sort_values('indice_saturacion')

top_nichos = saturacion.head(10)

fig = px.bar(
    top_nichos,
    x='indice_saturacion',
    y=top_nichos['tipo_vino'] + ' - ' + top_nichos['pais_origen'],
    orientation='h',
    title='Top 10 Nichos con Menor Saturación (Mejor Oportunidad)',
    labels={'indice_saturacion': 'Índice de Saturación', 'y': 'Nicho'},
    color='indice_saturacion',
    color_continuous_scale='Greens_r'
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**💡 Interpretación:**
- **Índice bajo (< 5)**: Nicho poco competido, buena oportunidad
- **Índice medio (5-15)**: Competencia moderada
- **Índice alto (> 15)**: Mercado saturado, evitar o diferenciarse mucho
""")

st.markdown("---")

st.header("📊 11. Elasticidad de Demanda por Segmento")

elasticidad = df_precios.groupby('segmento_precio').agg({
    'precio_actual': ['min', 'max', 'mean', 'std', 'count']
}).reset_index()

elasticidad.columns = ['segmento', 'min', 'max', 'mean', 'std', 'count']
elasticidad['coef_variacion'] = (elasticidad['std'] / elasticidad['mean']) * 100

fig = px.scatter(
    elasticidad,
    x='mean',
    y='count',
    size='coef_variacion',
    color='coef_variacion',
    text='segmento',
    title='Volumen vs Precio Promedio (Tamaño = Variabilidad de Precio)',
    labels={'mean': 'Precio Promedio', 'count': 'Cantidad de Productos', 'coef_variacion': 'Variabilidad (%)'},
    color_continuous_scale='RdYlGn_r'
)
fig.update_traces(textposition='top center')
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**💡 Decisión Estratégica:**
- Burbujas grandes = Alta variabilidad en precios = Clientes menos sensibles al precio
- Burbujas pequeñas = Precios muy similares = Guerra de precios, clientes muy sensibles
""")

st.markdown("---")

st.header("🔄 12. Indicadores de Rotación de Inventario")

df_rotacion = df_filtrado.groupby('tienda').agg({
    'tiene_descuento': 'sum',
    'nombre': 'count'
}).reset_index()

df_rotacion.columns = ['tienda', 'con_descuento', 'total']
df_rotacion['%_descuento'] = (df_rotacion['con_descuento'] / df_rotacion['total']) * 100

fig = px.bar(
    df_rotacion.sort_values('%_descuento', ascending=False),
    x='tienda',
    y='%_descuento',
    title='% de Productos con Descuento por Tienda',
    labels={'%_descuento': '% con Descuento', 'tienda': 'Tienda'},
    color='%_descuento',
    color_continuous_scale='Reds',
    text='%_descuento'
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**💡 Insight de Inventario:**
- **> 30% con descuento**: Inventario con problemas de rotación o estrategia agresiva
- **15-30%**: Normal, promociones estacionales
- **< 15%**: Inventario rotando bien, marca fuerte

**Recomendación:** Apunta a mantener descuentos en < 20% para no entrenar al cliente a esperar ofertas.
""")

st.markdown("---")

st.header("⚖️ 13. Calculadora de Punto de Equilibrio")

col1, col2, col3 = st.columns(3)

with col1:
    costos_fijos_mes = st.number_input("Costos fijos mensuales (renta, luz, etc.)", 10000, 200000, 50000)

with col2:
    precio_venta_promedio = st.number_input("Precio de venta promedio", 100, 2000, int(df_precios['precio_actual'].mean()))

with col3:
    costo_variable_botella = st.number_input("Costo variable por botella", 50, 1500, int(precio_venta_promedio * 0.6))

margen_contribucion = precio_venta_promedio - costo_variable_botella

if margen_contribucion > 0:
    botellas_equilibrio = costos_fijos_mes / margen_contribucion
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Botellas a Vender/Mes", f"{int(botellas_equilibrio)}")
    col2.metric("Botellas por Día", f"{int(botellas_equilibrio / 30)}")
    col3.metric("Margen por Botella", f"${margen_contribucion:.2f}")
    
    st.subheader("Proyección de Ganancia por Volumen")
    
    volumenes = list(range(0, int(botellas_equilibrio * 3), max(1, int(botellas_equilibrio / 10))))
    ganancias = [(v * margen_contribucion) - costos_fijos_mes for v in volumenes]
    
    fig = px.line(
        x=volumenes,
        y=ganancias,
        title='Ganancia/Pérdida según Volumen de Ventas',
        labels={'x': 'Botellas Vendidas/Mes', 'y': 'Ganancia (MXN)'}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Punto de Equilibrio")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("El margen de contribución es negativo. Ajusta tus precios o costos.")

st.markdown("---")

st.header("🏛️ 14. Concentración del Mercado")

market_share = df_filtrado['tienda'].value_counts(normalize=True) * 100

hhi = (market_share ** 2).sum()

col1, col2 = st.columns(2)

with col1:
    st.metric("Índice HHI", f"{hhi:.0f}")
    
    if hhi < 1500:
        st.success("✅ Mercado COMPETITIVO - Buena oportunidad para entrar")
    elif hhi < 2500:
        st.warning("⚠️ Mercado MODERADAMENTE CONCENTRADO - Hay jugadores dominantes")
    else:
        st.error("❌ Mercado ALTAMENTE CONCENTRADO - Difícil competir")

with col2:
    fig = px.pie(
        values=market_share.values,
        names=market_share.index,
        title='Cuota de Mercado por Tienda (% de productos)',
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**💡 Interpretación HHI:**
- **< 1,500**: Mercado no concentrado (muchos competidores)
- **1,500 - 2,500**: Moderadamente concentrado (algunos líderes)
- **> 2,500**: Altamente concentrado (oligopolio)
""")

st.markdown("---")

st.header("🎁 15. Tu Catálogo Inicial Recomendado")

st.markdown("Basado en datos del mercado, estos son los 20 productos que deberías tener desde el día 1:")

recomendaciones = df_catalogo.groupby(['tipo_vino', 'pais_origen', 'uva_varietal']).agg({
    'precio_actual': ['mean', 'count'],
    'tienda': 'nunique'
}).reset_index()

recomendaciones.columns = ['tipo_vino', 'pais_origen', 'uva', 'precio_promedio', 'frecuencia', 'tiendas']

recomendaciones['score'] = (
    recomendaciones['frecuencia'] * 0.4 +
    recomendaciones['tiendas'] * 0.3 + 
    (1 / (recomendaciones['precio_promedio'] / 500)) * 0.3
)

top_20 = recomendaciones.sort_values('score', ascending=False).head(20)

st.dataframe(
    top_20[['tipo_vino', 'pais_origen', 'uva', 'precio_promedio', 'frecuencia']].style.format({
        'precio_promedio': '${:.2f}'
    }),
    use_container_width=True
)

st.download_button(
    label="📥 Descargar Catálogo Recomendado (CSV)",
    data=top_20.to_csv(index=False).encode('utf-8'),
    file_name='catalogo_inicial_recomendado.csv',
    mime='text/csv'
)

st.markdown("---")

if run_comparison:
    st.header(f"📈 Comparativa de Mercado: {date1} vs. {date2}")

    @st.cache_data
    def load_specific_data(date_str):
        filepath = Path(f'./data/consolidated/{date_str}/datos_completos_listos.csv')
        if filepath.exists():
            return pd.read_csv(filepath)
        return pd.DataFrame()

    df_base = load_specific_data(date1)
    df_compare = load_specific_data(date2)

    if df_base.empty or df_compare.empty:
        st.error("No se pudieron cargar los datos para una o ambas fechas.")
    else:
        st.subheader("💰 Evolución de Precios")
        df_base['product_id'] = df_base['nombre'] + " | " + df_base['tienda']
        df_compare['product_id'] = df_compare['nombre'] + " | " + df_compare['tienda']
        df_merged = pd.merge(df_base[['product_id', 'precio_actual', 'nombre']], df_compare[['product_id', 'precio_actual']], on='product_id', suffixes=('_anterior', '_nuevo'))
        df_merged['cambio_precio'] = df_merged['precio_actual_nuevo'] - df_merged['precio_actual_anterior']
        df_merged['cambio_pct'] = (df_merged['cambio_precio'] / df_merged['precio_actual_anterior']) * 100
        df_con_cambios = df_merged[df_merged['cambio_precio'] != 0].sort_values('cambio_pct', ascending=False)
        
        st.metric("Productos Comunes Analizados", f"{len(df_merged)} vinos")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 📺 Vinos con Mayor Aumento de Precio")
            st.dataframe(df_con_cambios[['nombre', 'precio_actual_anterior', 'precio_actual_nuevo', 'cambio_pct']].head(10).style.format({'cambio_pct': '{:.2f}%'}))
        with col2:
            st.markdown("##### 📻 Vinos con Mayor Reducción de Precio")
            st.dataframe(df_con_cambios[['nombre', 'precio_actual_anterior', 'precio_actual_nuevo', 'cambio_pct']].tail(10).sort_values('cambio_pct').style.format({'cambio_pct': '{:.2f}%'}))
            
        st.subheader("📦 Evolución del Catálogo de Productos")
        set_base = set(df_base['product_id'])
        set_compare = set(df_compare['product_id'])
        vinos_nuevos = set_compare - set_base
        vinos_descontinuados = set_base - set_compare
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Vinos Nuevos en el Catálogo", f"{len(vinos_nuevos)}")
        col2.metric("Vinos Descontinuados", f"{len(vinos_descontinuados)}")
        col3.metric("Cambio Neto en Productos", f"{len(set_compare) - len(set_base)}")

        st.subheader("🏪 Cambio en Catálogo por Tienda")
        cat_base = df_base['tienda'].value_counts().reset_index()
        cat_base.columns = ['tienda', 'vinos_anterior']
        cat_compare = df_compare['tienda'].value_counts().reset_index()
        cat_compare.columns = ['tienda', 'vinos_nuevo']
        df_tiendas = pd.merge(cat_base, cat_compare, on='tienda', how='outer').fillna(0)
        df_tiendas['cambio'] = df_tiendas['vinos_nuevo'] - df_tiendas['vinos_anterior']

        fig = px.bar(df_tiendas, x='tienda', y='cambio', title='Cambio Neto en el Número de Vinos por Tienda',
                     labels={'tienda': 'Tienda', 'cambio': 'Cambio Neto en No. de Vinos'},
                     color='cambio', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🍷 Wine Market Analysis Dashboard | Datos actualizados: " + df['fecha_scraping'].max())