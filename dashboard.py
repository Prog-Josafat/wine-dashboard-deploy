# dashboard_deploy.py (Deploy - IA)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
import hmac
from glob import glob
import google.generativeai as genai
import os
from dotenv import load_dotenv

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wine_scraper.utils import DataQuality, DataConsolidator
    from wine_scraper.utils.gemini_analyzer import GeminiAnalyzer
    MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Módulos no disponibles: {e}. Usando clases de ejemplo. Funciones IA deshabilitadas.")
    MODULES_AVAILABLE = False
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

st.set_page_config(page_title="Wine Market Analysis AI", page_icon="🍷", layout="wide")

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

gemini = None
gemini_error_message = None
if MODULES_AVAILABLE:
    try:
        gemini = GeminiAnalyzer()
    except ValueError as e:
        gemini_error_message = str(e)
    except Exception as e:
        gemini_error_message = f"Error inesperado al inicializar Gemini: {str(e)}"
        MODULES_AVAILABLE = False
else:
    gemini_error_message = "Módulo GeminiAnalyzer no disponible (ImportError)."


st.title("🍷 Análisis de Mercado de Vinos - México (AI Enhanced)")
st.markdown("---")

@st.cache_data
def load_data():
    consolidator = DataConsolidator('./data', create_new_dir=False)
    latest_dir = consolidator.get_latest_consolidated_dir()
    if not latest_dir: st.error("❌ No se encontró carpeta de datos..."); st.stop()
    filepath = latest_dir / 'datos_completos_listos.csv'
    if not filepath.exists(): st.error(f"❌ No se encontró '{filepath.name}'..."); st.stop()
    return pd.read_csv(filepath), latest_dir.name

df, data_date = load_data()
quality = DataQuality()

with st.sidebar:
    st.markdown(f"**👤 Usuario:** {st.session_state.get('username', 'N/A')}")
    if st.button("🚪 Cerrar Sesión"):
        st.session_state["password_correct"] = False
        st.session_state["username"] = None
        st.rerun()
    st.markdown("---")


    st.success(f"📅 Datos de: {data_date}")

    if MODULES_AVAILABLE and gemini:
        st.success("🤖 Gemini AI: Activado")
    elif gemini_error_message:
         st.error(f"❌ {gemini_error_message}")
         st.info("💡 Verifica GEMINI_API_KEY en Secrets.")
    else:
         st.warning("⚠️ Funciones IA no disponibles.")

    st.markdown("---")

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

    st.subheader(f"🔔 Panel de Alertas ({len(alertas)})")
    if alertas:
        criticas = [a for a in alertas if a['nivel'] == 'error']
        advertencias = [a for a in alertas if a['nivel'] == 'warning']
        oportunidades = [a for a in alertas if a['nivel'] == 'success' or a['tipo'] == 'oportunidad']
        info = [a for a in alertas if a['nivel'] == 'info' and a['tipo'] != 'oportunidad']

        if criticas:
            with st.expander(f"**🔴 Críticas ({len(criticas)})**"):
                for alerta in criticas: st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**"); st.caption(f"_{alerta['detalle']}_")
        if advertencias:
            with st.expander(f"**🟡 Advertencias ({len(advertencias)})**"):
                for alerta in advertencias: st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**"); st.caption(f"_{alerta['detalle']}_")
        if oportunidades:
            with st.expander(f"**🟢 Oportunidades ({len(oportunidades)})**"):
                for alerta in oportunidades: st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**"); st.caption(f"_{alerta['detalle']}_")
        if info:
            with st.expander(f"**🔵 Información ({len(info)})**"):
                for alerta in info: st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**"); st.caption(f"_{alerta['detalle']}_")
    else:
        st.success("✅ Sin alertas pendientes")

    st.markdown("---")

    st.header("🔍 Filtros de Mercado")
    with st.expander("🏪 Tiendas"):
        tiendas_seleccionadas = st.multiselect("Selecciona Tiendas", options=df['tienda'].unique(), default=df['tienda'].unique(), label_visibility="collapsed")
    with st.expander("🍷 Tipo de Vino"):
        tipos_seleccionados = st.multiselect("Selecciona Tipos de Vino", options=df['tipo_vino'].unique(), default=df['tipo_vino'].unique(), label_visibility="collapsed")
    with st.expander("🌍 País de Origen"):
        paises_disponibles = sorted(df['pais_origen'].dropna().unique())
        paises_seleccionados = st.multiselect("Selecciona Países de Origen", options=paises_disponibles, default=paises_disponibles, label_visibility="collapsed")
    with st.expander("💰 Segmento de Precio"):
        segmentos_disponibles = sorted(df['segmento_precio'].dropna().unique())
        segmentos_seleccionados = st.multiselect("Selecciona Segmentos de Precio", options=segmentos_disponibles, default=segmentos_disponibles, label_visibility="collapsed")
    with st.expander("💲 Rango de Precio (MXN)"):
        precio_min = float(df['precio_actual'].min())
        precio_max = float(df['precio_actual'].max())
        rango_precio = st.slider("Selecciona un Rango de Precio", min_value=precio_min, max_value=precio_max, value=(precio_min, precio_max), label_visibility="collapsed")

df_filtrado = df[
    (df['tienda'].isin(tiendas_seleccionadas)) &
    (df['tipo_vino'].isin(tipos_seleccionados)) &
    (df['precio_actual'].between(rango_precio[0], rango_precio[1])) &
    (df['pais_origen'].isin(paises_seleccionados)) &
    (df['segmento_precio'].isin(segmentos_seleccionados))
]
if df_filtrado.empty:
    st.error("No hay datos que coincidan con los filtros seleccionados. Por favor, amplía tu selección.")
    st.stop()
df_precios = quality.get_dataset_for_analysis(df_filtrado, 'precio')
df_catalogo = quality.get_dataset_for_analysis(df_filtrado, 'catalogo')

tab_resumen, tab_precios, tab_catalogo, tab_ia, tab_herramientas, tab_comparativa = st.tabs([
    "📊 Resumen y Catálogo",
    "💰 Precios y Competencia",
    "🍇 Uvas y Nichos",
    "🤖 Análisis con IA",
    "⚖️ Herramientas",
    "🔬 Análisis Comparativo"
])

with tab_resumen:
    st.header("Resumen General del Mercado")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Productos Filtrados", f"{len(df_filtrado):,}")
    with col2: st.metric("Precio Promedio", f"${df_precios['precio_actual'].mean():,.2f}")
    with col3: st.metric("Con Descuento", f"{df_filtrado['tiene_descuento'].mean() * 100:.1f}%")
    with col4: st.metric("Tiendas Activas", df_filtrado['tienda'].nunique())
    st.markdown("---")

    st.header("📚 Análisis Básico de Catálogo")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución por País de Origen")
        pais_count = df_catalogo['pais_origen'].value_counts().head(10)
        pais_df = pais_count.reset_index(); pais_df.columns = ['País', 'Cantidad']
        fig = px.bar(pais_df, x='Cantidad', y='País', orientation='h', title="Top 10 Países",
                     labels={'Cantidad': 'Productos', 'País': 'País'}, color='Cantidad',
                     color_continuous_scale='Reds', text='Cantidad')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Distribución por Tipo de Vino")
        tipo_count = df_catalogo['tipo_vino'].value_counts()
        fig = px.pie(values=tipo_count.values, names=tipo_count.index, title="Composición por Tipo",
                     color_discrete_sequence=px.colors.sequential.RdBu, hole=0.3)
        st.plotly_chart(fig, use_container_width=True)

with tab_precios:
    st.header("Análisis Detallado de Precios y Competencia")

    st.subheader("💰 Análisis de Precios")
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Distribución General", "Comparativa por Tienda", "Precio por Tipo"])
    with sub_tab1:
        fig = px.histogram(df_precios, x='precio_actual', nbins=50, title="Distribución de Precios", labels={'precio_actual': 'Precio (MXN)', 'count': 'Cantidad'}, color_discrete_sequence=['#8B0000'])
        fig.add_vline(x=df_precios['precio_actual'].median(), line_dash="dash", annotation_text=f"Mediana: ${df_precios['precio_actual'].median():,.2f}")
        st.plotly_chart(fig, use_container_width=True)
    with sub_tab2:
        fig_box = px.box(df_precios, x='tienda', y='precio_actual', title="Rangos de Precio por Tienda", labels={'precio_actual': 'Precio (MXN)', 'tienda': 'Tienda'}, color='tienda')
        st.plotly_chart(fig_box, use_container_width=True)
        st.subheader("🎻 7. Densidad de Precios por Competidor")
        fig_violin = px.violin(df_filtrado, x='tienda', y='precio_actual', color='tienda', box=True, title='Distribución y Densidad de Precios', labels={'precio_actual': 'Precio (MXN)', 'tienda': 'Tienda'})
        st.plotly_chart(fig_violin, use_container_width=True)
    with sub_tab3:
        precio_tipo = df_precios.groupby('tipo_vino')['precio_actual'].agg(['mean', 'count']).reset_index()
        precio_tipo = precio_tipo[precio_tipo['count'] > 5].sort_values('mean', ascending=False)
        fig = px.bar(precio_tipo, x='tipo_vino', y='mean', title="Precio Promedio por Tipo ( >5 prod.)", labels={'mean': 'Precio Promedio (MXN)', 'tipo_vino': 'Tipo'}, color='mean', color_continuous_scale='Reds', text='mean')
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    st.subheader("🏪 Mapa de Competitividad")
    df_competencia = df_filtrado.groupby('tienda').agg(num_vinos=('nombre', 'count'), precio_promedio=('precio_actual', 'mean')).reset_index()
    fig_competidores = px.scatter(df_competencia, x='num_vinos', y='precio_promedio', size='num_vinos', color='precio_promedio', text='tienda', title='Posicionamiento: Catálogo vs. Precio', labels={'num_vinos': 'No. Vinos', 'precio_promedio': 'Precio Promedio (MXN)'}, color_continuous_scale='RdYlGn_r', size_max=60)
    fig_competidores.update_traces(textposition='top center')
    st.plotly_chart(fig_competidores, use_container_width=True)
    st.markdown("---")

    st.subheader("📊 Elasticidad por Segmento")
    elasticidad = df_precios.groupby('segmento_precio').agg(precio_min=('precio_actual', 'min'), precio_max=('precio_actual', 'max'), precio_mean=('precio_actual', 'mean'), precio_std=('precio_actual', 'std'), count=('precio_actual', 'count')).reset_index()
    elasticidad.columns = ['segmento', 'min', 'max', 'mean', 'std', 'count']
    elasticidad['coef_variacion'] = (elasticidad['std'] / elasticidad['mean']) * 100
    fig_elasticidad = px.scatter(elasticidad, x='mean', y='count', size='coef_variacion', color='coef_variacion', text='segmento', title='Volumen vs. Precio Promedio (Tamaño = Variabilidad)', labels={'mean': 'Precio Promedio', 'count': 'Cantidad', 'coef_variacion': 'Variabilidad (%)'}, color_continuous_scale='RdYlGn_r')
    fig_elasticidad.update_traces(textposition='top center')
    st.plotly_chart(fig_elasticidad, use_container_width=True)
    st.markdown("""**💡 Decisión Estratégica:** Burbujas grandes = Alta variabilidad = Clientes menos sensibles al precio. Burbujas pequeñas = Precios consistentes = Guerra de precios.""")
    st.markdown("---")

    st.subheader("🔄 Indicadores de Rotación (Descuentos)")
    df_rotacion = df_filtrado.groupby('tienda').agg(con_descuento=('tiene_descuento', 'sum'), total=('nombre', 'count')).reset_index()
    df_rotacion['%_descuento'] = (df_rotacion['con_descuento'] / df_rotacion['total']) * 100
    fig_rotacion = px.bar(df_rotacion.sort_values('%_descuento', ascending=False), x='tienda', y='%_descuento', title='% Productos con Descuento por Tienda', labels={'%_descuento': '% Descuento', 'tienda': 'Tienda'}, color='%_descuento', color_continuous_scale='Reds', text='%_descuento')
    fig_rotacion.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_rotacion, use_container_width=True)
    st.markdown("""**💡 Insight de Inventario:** > 30% descuento: Posible baja rotación/agresividad. < 15%: Buena rotación/marca fuerte.""")

with tab_catalogo:
    st.header("Análisis de Variedad y Oportunidades")

    st.subheader("🍇 Cuota de Mercado por Tipo de Uva")
    if 'uva_varietal' in df_filtrado.columns:
        df_uvas_treemap = df_filtrado[~df_filtrado['uva_varietal'].isin(['No especificado', 'Tinto', 'Blanco'])].copy()
        if not df_uvas_treemap.empty:
            df_uvas_treemap_counts = df_uvas_treemap['uva_varietal'].value_counts().nlargest(20).reset_index()
            df_uvas_treemap_counts.columns = ['uva_varietal', 'cantidad']
            fig_treemap = px.treemap(df_uvas_treemap_counts, path=['uva_varietal'], values='cantidad', title='Distribución por Uva (Top 20)', color='cantidad', color_continuous_scale='Reds', labels={'cantidad': 'Cantidad'})
            st.plotly_chart(fig_treemap, use_container_width=True)
            st.markdown("**Insight:** Rectángulos grandes = uvas dominantes. Balancea inventario.")
        else: st.info("No hay datos de uvas para mostrar.")
    else: st.info("Columna 'uva_varietal' no encontrada.")
    st.markdown("---")

    st.subheader("💎 Oportunidades de Nicho")
    combinaciones = df_catalogo.groupby(['tipo_vino', 'pais_origen']).size().reset_index(name='cantidad')
    pivot = combinaciones.pivot_table(index='tipo_vino', columns='pais_origen', values='cantidad').fillna(0)
    fig_heatmap = px.imshow(pivot, title="Mapa de Calor: Tipo vs. País", labels={'x': 'País', 'y': 'Tipo', 'color': 'Cantidad'}, color_continuous_scale='YlOrRd', aspect='auto')
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown("---")

    st.subheader("🎯 Índice de Saturación del Mercado")
    saturacion = df_catalogo.groupby(['tipo_vino', 'pais_origen']).agg(tiendas=('tienda', 'nunique'), productos=('nombre', 'count')).reset_index()
    saturacion['indice_saturacion'] = (saturacion['productos'] / saturacion['tiendas']).fillna(0)
    saturacion = saturacion[saturacion['productos'] > 0].sort_values('indice_saturacion').head(10)
    fig_saturacion = px.bar(saturacion, x='indice_saturacion', y=saturacion['tipo_vino'] + ' - ' + saturacion['pais_origen'], orientation='h', title='Top 10 Nichos Menos Saturados', labels={'indice_saturacion': 'Índice (Prod/Tienda)', 'y': 'Nicho'}, color='indice_saturacion', color_continuous_scale='Greens_r')
    fig_saturacion.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_saturacion, use_container_width=True)
    st.markdown("""**💡 Interpretación:** Índice bajo (< 5): Poca competencia. Índice alto (> 15): Mercado saturado.""")
    st.markdown("---")

    st.subheader("🎁 Catálogo Inicial Recomendado")
    st.markdown("Basado en popularidad, disponibilidad y precio:")
    if 'uva_varietal' in df_catalogo.columns:
        recomendaciones = df_catalogo.groupby(['tipo_vino', 'pais_origen', 'uva_varietal']).agg(precio_promedio=('precio_actual', 'mean'), frecuencia=('nombre', 'count'), tiendas=('tienda', 'nunique')).reset_index()
        recomendaciones['score'] = (recomendaciones['frecuencia'] * 0.4) + \
                                   (recomendaciones['tiendas'] * 0.3) + \
                                   ((1 / (recomendaciones['precio_promedio'] / 500 + 1e-6)) * 0.3)
        top_20 = recomendaciones.sort_values('score', ascending=False).head(20)
        st.dataframe(top_20[['tipo_vino', 'pais_origen', 'uva_varietal', 'precio_promedio', 'frecuencia']].rename(columns={'uva_varietal': 'uva'}).style.format({'precio_promedio': '${:,.2f}'}), use_container_width=True)
        csv_data = top_20.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Descargar Recomendación (CSV)", data=csv_data, file_name='catalogo_recomendado.csv', mime='text/csv')
    else: st.info("Falta columna 'uva_varietal' para generar recomendación.")

with tab_ia:
    st.header("🤖 Análisis y Herramientas con Inteligencia Artificial")

    if gemini:
        st.subheader("💡 Análisis Inteligente de Alertas Globales")
        st.markdown("Pulsa para obtener un resumen estratégico basado en **todas** las alertas del mercado completo.")
        if st.button("🧠 Analizar Alertas Globales con IA", key="analisis_alertas_main"):
             with st.spinner("Analizando alertas globales..."):
                 try:
                     analisis_global_ia = gemini.analizar_alertas_inteligentes(df, alertas)
                     st.markdown(analisis_global_ia)
                 except Exception as e: st.error(f"Error al generar análisis IA: {e}")
        st.markdown("---")

        st.subheader("🛠️ Herramientas Específicas de IA")
        sub_tab_ia1, sub_tab_ia2, sub_tab_ia3 = st.tabs(["🎯 Análisis de Producto", "📦 Catálogo Inicial", "💡 Insights del Segmento"])

        with sub_tab_ia1:
            st.markdown("##### Analizar Posicionamiento de un Producto")
            col1, col2 = st.columns([3,1])
            with col1: producto_buscar = st.text_input("Nombre (o parte):", placeholder="Ej: Casa Madero Cabernet")
            with col2: analizar_prod_btn = st.button("🔍 Analizar", key="analizar_producto_main", type="primary", use_container_width=True)
            if analizar_prod_btn:
                if producto_buscar:
                    with st.spinner("Analizando producto..."):
                        try: analisis = gemini.analizar_producto_especifico(producto_buscar, df_filtrado); st.markdown(analisis)
                        except Exception as e: st.error(f"Error: {e}")
                else: st.warning("Ingresa un nombre.")
        
        with sub_tab_ia2:
            st.markdown("##### Recomendación de Catálogo Inicial")
            col1, col2 = st.columns([3,1])
            with col1: presupuesto = st.number_input("Presupuesto (MXN):", 10000, 1000000, 50000, 5000)
            with col2: generar_cat_btn = st.button("📦 Generar", key="generar_catalogo_main", type="primary", use_container_width=True)
            if generar_cat_btn:
                with st.spinner("Generando recomendación..."):
                     try: recomendacion = gemini.generar_recomendacion_catalogo(df_filtrado, presupuesto); st.markdown(recomendacion)
                     except Exception as e: st.error(f"Error: {e}")
        
        with sub_tab_ia3:
            st.markdown("##### Obtener Insights del Mercado Filtrado")
            st.caption("Análisis basado en alertas globales y datos filtrados.")
            if st.button("🧠 Generar Análisis del Segmento", key="analisis_segmento_main", type="primary", use_container_width=True):
                with st.spinner("Analizando segmento..."):
                    try:
                        analisis_segmento = gemini.analizar_alertas_inteligentes(df_filtrado, alertas)
                        st.markdown("---"); st.markdown(analisis_segmento); st.markdown("---")
                    except Exception as e: st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Las herramientas IA no están disponibles.")
        st.info("Verifica tu API Key de Gemini en los Secrets de Streamlit Cloud.")


with tab_herramientas:
    st.header("Calculadoras y Simuladores")
    st.subheader("💰 Simulador de Rentabilidad")
    with st.expander("Expandir Simulador"):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Configuración de Costos")
            margen_objetivo = st.slider("Margen objetivo (%)", 10, 100, 40)
            costo_operativo = st.number_input("Costo operativo/botella (MXN)", 10.0, 200.0, 50.0, 5.0)
        with col2:
            st.caption("Precios Sugeridos (Top 5 Tipos por Mediana)")
            precio_competencia = df_precios.groupby('tipo_vino')['precio_actual'].median()
            num_tipos_mostrar = min(5, len(precio_competencia))
            if num_tipos_mostrar > 0:
                 for tipo in precio_competencia.nlargest(num_tipos_mostrar).index:
                     precio_mercado = precio_competencia[tipo]
                     precio_sugerido = precio_mercado * 0.95
                     ganancia = precio_sugerido - costo_operativo
                     margen_real = (ganancia / precio_sugerido) * 100 if precio_sugerido > 0 else 0
                     st.metric(label=f"{tipo}", value=f"${precio_sugerido:,.2f}", delta=f"{margen_real:.1f}% margen")
            else:
                 st.info("No hay suficientes tipos de vino para mostrar precios sugeridos.")
    st.markdown("---")

    st.subheader("⚖️ Calculadora de Punto de Equilibrio")
    with st.expander("Expandir Calculadora"):
        col1, col2, col3 = st.columns(3)
        with col1: costos_fijos_mes = st.number_input("Costos fijos/mes", 10000, 200000, 50000, 1000, key="costos_fijos")
        with col2: precio_venta_promedio = st.number_input("Precio venta promedio", 100.0, 5000.0, float(df_precios['precio_actual'].mean()), key="precio_venta")
        with col3: costo_variable_botella = st.number_input("Costo variable/botella", 50.0, 4500.0, float(precio_venta_promedio * 0.6), key="costo_variable")
        margen_contribucion = precio_venta_promedio - costo_variable_botella
        if margen_contribucion > 0:
            botellas_equilibrio = costos_fijos_mes / margen_contribucion
            col1, col2, col3 = st.columns(3)
            col1.metric("Botellas/Mes (Equilibrio)", f"{botellas_equilibrio:,.0f}")
            col2.metric("Botellas/Día", f"{(botellas_equilibrio / 30):,.1f}")
            col3.metric("Margen/Botella", f"${margen_contribucion:,.2f}")
            max_vol = int(botellas_equilibrio * 2.5) if botellas_equilibrio > 0 else 100
            step = max(1, int(max_vol / 15))
            volumenes = list(range(0, max_vol + step, step))
            ganancias = [(v * margen_contribucion) - costos_fijos_mes for v in volumenes]
            fig_pe = px.line(x=volumenes, y=ganancias, title='Ganancia/Pérdida vs Volumen', labels={'x': 'Botellas/Mes', 'y': 'Ganancia (MXN)'})
            fig_pe.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Punto Equilibrio")
            fig_pe.add_vline(x=botellas_equilibrio, line_dash="dash", line_color="red")
            st.plotly_chart(fig_pe, use_container_width=True)
        else: st.error("Margen de contribución negativo. Ajusta precios/costos.")
    st.markdown("---")

    st.subheader("🏛️ Concentración del Mercado (HHI)")
    market_share = df_filtrado['tienda'].value_counts(normalize=True) * 100
    hhi = (market_share ** 2).sum()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Índice HHI", f"{hhi:,.0f}")
        if hhi < 1500: st.success("✅ COMPETITIVO")
        elif hhi < 2500: st.warning("⚠️ MODERADO")
        else: st.error("❌ CONCENTRADO")
        st.caption("(<1500 Comp., 1500-2500 Mod., >2500 Conc.)")
    with col2:
        fig_hhi = px.pie(values=market_share.values, names=market_share.index, title='Cuota de Mercado por Tienda', hole=0.4)
        st.plotly_chart(fig_hhi, use_container_width=True)

with tab_comparativa:
    st.header("🔬 Análisis Comparativo Temporal")

    available_dates_comp = get_available_dates()
    if len(available_dates_comp) >= 2:
        col1, col2, col3 = st.columns([2,2,1])
        with col1: date1_comp = st.selectbox("Fecha base (anterior):", available_dates_comp, index=1, key="comp_date1")
        with col2: date2_comp = st.selectbox("Fecha a comparar (nueva):", available_dates_comp, index=0, key="comp_date2")
        with col3:
            st.write("")
            st.write("")
            run_comparison_main = st.button("📊 Comparar", key="compare_main", type="primary", use_container_width=True)

        if run_comparison_main:
            @st.cache_data
            def load_specific_data(date_str):
                filepath = Path(f'./data/consolidated/{date_str}/datos_completos_listos.csv')
                if filepath.exists(): return pd.read_csv(filepath)
                return pd.DataFrame()

            df_base = load_specific_data(date1_comp)
            df_compare = load_specific_data(date2_comp)

            if df_base.empty or df_compare.empty:
                st.error("No se pudieron cargar datos para una o ambas fechas.")
            else:
                st.subheader("💰 Evolución de Precios")
                df_base['product_id'] = df_base['nombre'] + " | " + df_base['tienda']
                df_compare['product_id'] = df_compare['nombre'] + " | " + df_compare['tienda']
                df_merged = pd.merge(df_base[['product_id', 'nombre', 'precio_actual']], df_compare[['product_id', 'precio_actual']], on='product_id', suffixes=('_anterior', '_nuevo'))
                df_merged['cambio_precio'] = df_merged['precio_actual_nuevo'] - df_merged['precio_actual_anterior']
                df_merged['cambio_pct'] = (df_merged['cambio_precio'] / df_merged['precio_actual_anterior'].replace(0, pd.NA)) * 100 
                df_con_cambios = df_merged.dropna(subset=['cambio_pct'])
                df_con_cambios = df_con_cambios[df_con_cambios['cambio_precio'] != 0].sort_values('cambio_pct', ascending=False)

                st.metric("Productos Comunes Analizados", f"{len(df_merged)} vinos")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("##### 📈 Mayor Aumento")
                    st.dataframe(df_con_cambios[['nombre', 'precio_actual_anterior', 'precio_actual_nuevo', 'cambio_pct']].head(10).style.format({'precio_actual_anterior': '${:,.2f}', 'precio_actual_nuevo': '${:,.2f}', 'cambio_pct': '{:.1f}%'}))
                with col_b:
                    st.markdown("##### 📉 Mayor Reducción")
                    st.dataframe(df_con_cambios[['nombre', 'precio_actual_anterior', 'precio_actual_nuevo', 'cambio_pct']].tail(10).sort_values('cambio_pct').style.format({'precio_actual_anterior': '${:,.2f}', 'precio_actual_nuevo': '${:,.2f}', 'cambio_pct': '{:.1f}%'}))
                
                st.subheader("📦 Evolución del Catálogo")
                set_base = set(df_base['product_id'])
                set_compare = set(df_compare['product_id'])
                vinos_nuevos = set_compare - set_base
                vinos_descontinuados = set_base - set_compare
                col_c, col_d, col_e = st.columns(3)
                col_c.metric("Vinos Nuevos", f"{len(vinos_nuevos)}")
                col_d.metric("Vinos Descontinuados", f"{len(vinos_descontinuados)}")
                col_e.metric("Cambio Neto", f"{len(set_compare) - len(set_base)}", delta_color="off")

                st.subheader("🏪 Cambio en Catálogo por Tienda")
                cat_base = df_base['tienda'].value_counts().reset_index().rename(columns={'index': 'tienda', 'tienda': 'vinos_anterior'})
                cat_compare = df_compare['tienda'].value_counts().reset_index().rename(columns={'index': 'tienda', 'tienda': 'vinos_nuevo'})
                df_tiendas = pd.merge(cat_base, cat_compare, on='tienda', how='outer').fillna(0)
                df_tiendas['cambio'] = df_tiendas['vinos_nuevo'] - df_tiendas['vinos_anterior']
                fig_tiendas_comp = px.bar(df_tiendas, x='tienda', y='cambio', title='Cambio Neto Vinos por Tienda', labels={'tienda': 'Tienda', 'cambio': 'Cambio Neto'}, color='cambio', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_tiendas_comp, use_container_width=True)
    else:
        st.info("Se necesitan al menos dos carpetas de datos ('./data/consolidated/YYYYMMDD/') para comparar.")

st.markdown("---")
st.info("Dashboard creado para el análisis estratégico del mercado de vinos en México.")