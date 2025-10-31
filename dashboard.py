# dashboard_deploy.py (Deploy)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
import hmac
from glob import glob
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Bloque para importar clases personalizadas ---
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from wine_scraper.utils import DataQuality, DataConsolidator
    from wine_scraper.utils.gemini_analyzer import GeminiAnalyzer
    from wine_scraper.utils.historical_analyzer import HistoricalAnalyzer
    from wine_scraper.utils.price_simulator import PriceSimulator
    from wine_scraper.utils.predictor import SimplePredictor
    MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Error crítico al importar módulos: {e}")
    st.error("Verifica que los archivos .py estén en 'wine_scraper/utils/' y que __init__.py los incluya si es necesario.")
    MODULES_AVAILABLE = False
    HistoricalAnalyzer = None
    PriceSimulator = None
    GeminiAnalyzer = None
    SimplePredictor = None
    class DataQuality:
        def get_dataset_for_analysis(self, df, analysis_type): return df.copy()
    class DataConsolidator:
        def __init__(self, base_path='data', create_new_dir=False):
            self.base_path = Path(base_path)
            self.consolidated_path = self.base_path / 'consolidated'
        def get_latest_consolidated_dir(self):
            if not self.consolidated_path.exists(): return None
            date_dirs = [p for p in self.consolidated_path.iterdir() if p.is_dir() and p.name.isdigit()]
            if not date_dirs: return None
            return max(date_dirs, key=lambda p: p.name)

def get_available_dates(base_path='data/consolidated'):
    """Obtiene las fechas disponibles de los directorios consolidados."""
    full_base_path = Path(base_path)
    if not full_base_path.exists(): return []
    date_dirs = [p.name for p in full_base_path.iterdir() if p.is_dir() and len(p.name) == 8 and p.name.isdigit()]
    return sorted(date_dirs, reverse=True)

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
    if st.session_state.get("password_correct", False):
        return True
    login_form()
    return False

if not check_password():
    st.stop()

gemini = None
gemini_error_message = None
if MODULES_AVAILABLE and GeminiAnalyzer:
    try:
        gemini = GeminiAnalyzer()
    except ValueError as e:
        gemini_error_message = str(e)
    except Exception as e:
        gemini_error_message = f"Error inesperado al inicializar Gemini: {str(e)}"
elif not MODULES_AVAILABLE:
    gemini_error_message = "Módulo GeminiAnalyzer no disponible (ImportError)."
else:
    gemini_error_message = "Clase GeminiAnalyzer no encontrada."

st.title("🍷 Análisis de Mercado de Vinos - México (AI Enhanced)")
st.markdown("---")

@st.cache_data
def load_data():
    """Carga los datos consolidados más recientes."""
    consolidator = DataConsolidator('data', create_new_dir=False)
    latest_dir = consolidator.get_latest_consolidated_dir()

    if not latest_dir:
        st.error("❌ No se encontró ninguna carpeta de datos consolidada en './data/consolidated/'.")
        st.error("Verifica que la carpeta 'data/consolidated/YYYYMMDD' exista en el repositorio.")
        st.stop()

    filepath = latest_dir / 'datos_completos_listos.csv'
    if not filepath.exists():
        st.error(f"❌ No se encontró el archivo '{filepath.name}' en la carpeta '{latest_dir.name}'.")
        st.error("Verifica que el archivo CSV exista en la carpeta de fecha más reciente.")
        st.stop()

    try:
        df_loaded = pd.read_csv(filepath)
        print(f"✅ Datos cargados desde: {filepath}")
        return df_loaded, latest_dir.name
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo CSV '{filepath.name}': {e}")
        st.stop()

df, data_date = load_data()

if df.empty:
    st.error("El archivo CSV cargado está vacío.")
    st.stop()
if 'precio_actual' not in df.columns or 'tienda' not in df.columns:
    st.error("El archivo CSV no contiene las columnas esperadas ('precio_actual', 'tienda', etc.). Verifica el formato.")
    st.stop()

quality = DataQuality()

historical_analyzer = None
historical_available = False
available_hist_dates = []
if HistoricalAnalyzer:
    try:
        historical_analyzer = HistoricalAnalyzer('data/consolidated')
        available_hist_dates = historical_analyzer.get_available_dates()
        historical_available = len(available_hist_dates) >= 2
        if not historical_available and len(available_hist_dates) == 1:
            st.sidebar.info("ℹ️ Solo hay 1 snapshot. Se necesita historial para análisis temporal y predicciones.")
        elif len(available_hist_dates) == 0:
            st.sidebar.warning("⚠️ No se encontraron snapshots históricos en 'data/consolidated/'.")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error al cargar historial: {e}")
        historical_analyzer = None
        historical_available = False
else:
    st.sidebar.warning("Módulo HistoricalAnalyzer no disponible.")

with st.sidebar:
    st.markdown(f"**👤 Usuario:** {st.session_state.get('username', 'N/A')}")
    if st.button("🚪 Cerrar Sesión"):
        st.session_state["password_correct"] = False
        st.session_state["username"] = None
        st.rerun()
    st.markdown("---")

    st.success(f"📅 Datos de: {data_date}")
    if gemini:
        st.success("🤖 Gemini AI: Activado")
    elif gemini_error_message:
        st.error(f"❌ {gemini_error_message}")
        st.info("💡 Verifica GEMINI_API_KEY en Secrets.")
    st.markdown("---")

    alertas = []
    try:
        if 'precio_anterior' in df.columns and 'precio_actual' in df.columns:
            df_alertas = df.dropna(subset=['precio_actual', 'precio_anterior'])
            df_alertas = df_alertas[df_alertas['precio_anterior'] > 0]
            if not df_alertas.empty:
                aumentos = df_alertas[((df_alertas['precio_actual'] - df_alertas['precio_anterior']) / df_alertas['precio_anterior']) > 0.2]
                if len(aumentos) > 0:
                    alertas.append({'tipo': 'precio', 'nivel': 'warning', 'icono': '📈', 'mensaje': f"{len(aumentos)} productos con aumento >20%", 'detalle': f"Promedio: {aumentos['precio_actual'].mean() - aumentos['precio_anterior'].mean():.2f} MXN"})
                reducciones = df_alertas[((df_alertas['precio_anterior'] - df_alertas['precio_actual']) / df_alertas['precio_anterior']) > 0.2]
                if len(reducciones) > 0:
                    alertas.append({'tipo': 'precio', 'nivel': 'success', 'icono': '📉', 'mensaje': f"{len(reducciones)} productos con reducción >20%", 'detalle': f"Promedio: {reducciones['precio_anterior'].mean() - reducciones['precio_actual'].mean():.2f} MXN"})

        if 'tipo_vino' in df.columns and 'pais_origen' in df.columns:
            combinaciones = df.groupby(['tipo_vino', 'pais_origen'], observed=True).size().reset_index(name='cantidad')
            nichos_criticos = combinaciones[combinaciones['cantidad'] < 3]
            if len(nichos_criticos) > 0:
                alertas.append({'tipo': 'oportunidad', 'nivel': 'success', 'icono': '💎', 'mensaje': f"{len(nichos_criticos)} nichos con <3 competidores", 'detalle': "Oportunidad de entrada con baja competencia"})

        if 'precio_actual' in df.columns:
            Q1, Q3 = df['precio_actual'].quantile(0.25), df['precio_actual'].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outliers = df[(df['precio_actual'] < Q1 - 1.5 * IQR) | (df['precio_actual'] > Q3 + 1.5 * IQR)]
                if len(outliers) > 0:
                    alertas.append({'tipo': 'precio', 'nivel': 'warning', 'icono': '⚠️', 'mensaje': f"{len(outliers)} productos con precio atípico", 'detalle': "Pueden ser errores o productos premium/liquidación"})

        if 'tiene_descuento' in df.columns and 'tienda' in df.columns and 'nombre' in df.columns:
            df['tiene_descuento'] = df['tiene_descuento'].astype(bool)
            descuentos_tienda = df.groupby('tienda', observed=True).agg(
                total=('nombre', 'count'),
                con_descuento=('tiene_descuento', 'sum')
            )
            descuentos_tienda = descuentos_tienda[descuentos_tienda['total'] > 0]
            descuentos_tienda['%_descuento'] = (descuentos_tienda['con_descuento'] / descuentos_tienda['total']) * 100
            tiendas_alto_descuento = descuentos_tienda[descuentos_tienda['%_descuento'] > 30]
            if len(tiendas_alto_descuento) > 0:
                for tienda in tiendas_alto_descuento.index:
                    pct = tiendas_alto_descuento.loc[tienda, '%_descuento']
                    alertas.append({'tipo': 'inventario', 'nivel': 'warning', 'icono': '🏷️', 'mensaje': f"{tienda}: {pct:.1f}% en descuento", 'detalle': "Posible problema de rotación o estrategia agresiva"})

        if 'segmento_precio' in df.columns:
            segmentos_esperados = ['Económico', 'Medio-Bajo', 'Medio', 'Medio-Alto', 'Premium']
            segmentos_faltantes = set(segmentos_esperados) - set(df['segmento_precio'].dropna().unique())
            if len(segmentos_faltantes) > 0:
                alertas.append({'tipo': 'catalogo', 'nivel': 'info', 'icono': '📊', 'mensaje': f"Segmentos sin cobertura: {', '.join(segmentos_faltantes)}", 'detalle': "Considera ampliar el catálogo"})

        if 'calidad_datos' in df.columns and 'tienda' in df.columns:
            tiendas_baja_calidad = df[df['calidad_datos'] == 'parcial'].groupby('tienda', observed=True).size()
            if len(tiendas_baja_calidad) > 0:
                total_tienda = df.groupby('tienda', observed=True).size()
                pct_baja = (tiendas_baja_calidad.reindex(total_tienda.index, fill_value=0) / total_tienda * 100).round(1)
                for tienda, pct in pct_baja.items():
                    if pct > 20:
                        alertas.append({'tipo': 'datos', 'nivel': 'error', 'icono': '❌', 'mensaje': f"{tienda}: {pct}% datos incompletos", 'detalle': "Revisar scraper o estructura del sitio"})

        if 'pais_origen' in df.columns:
            productos_sin_origen = df[df['pais_origen'].isin(['No especificado', '', None])].shape[0]
            if productos_sin_origen > 0:
                pct_sin_origen = (productos_sin_origen / len(df)) * 100
                alertas.append({'tipo': 'datos', 'nivel': 'warning', 'icono': '🌍', 'mensaje': f"{pct_sin_origen:.1f}% productos sin origen", 'detalle': "Dificulta análisis de mercado por región"})

        if 'tienda' in df.columns:
            cuota_tienda = df['tienda'].value_counts(normalize=True) * 100
            if not cuota_tienda.empty:
                tienda_dominante = cuota_tienda.iloc[0]
                if tienda_dominante > 40:
                    alertas.append({'tipo': 'mercado', 'nivel': 'info', 'icono': '🛒', 'mensaje': f"{cuota_tienda.index[0]} domina ({tienda_dominante:.1f}%)", 'detalle': "Alta concentración de mercado"})

        if 'tipo_vino' in df.columns and 'precio_actual' in df.columns:
            precio_tipo = df.groupby('tipo_vino', observed=True)['precio_actual'].mean().sort_values(ascending=False)
            precio_promedio_global = df['precio_actual'].mean()
            if not precio_tipo.empty and precio_promedio_global > 0 and precio_tipo.iloc[0] > precio_promedio_global * 2:
                tipo_caro = precio_tipo.index[0]
                precio_promedio_tipo = precio_tipo.iloc[0]
                alertas.append({'tipo': 'mercado', 'nivel': 'info', 'icono': '💰', 'mensaje': f"{tipo_caro}: precio 2x > promedio", 'detalle': f"Promedio {tipo_caro}: ${precio_promedio_tipo:,.2f} MXN"})

        if 'pais_origen' in df.columns and 'nombre' in df.columns and 'precio_actual' in df.columns:
            paises_precio = df.groupby('pais_origen', observed=True).agg(
                cantidad=('nombre', 'count'),
                precio_promedio=('precio_actual', 'mean')
            )
            precio_mediano_global = df['precio_actual'].median()
            paises_nicho = paises_precio[(paises_precio['cantidad'] < 10) & (paises_precio['precio_promedio'] > precio_mediano_global)]
            if len(paises_nicho) > 0:
                for pais in paises_nicho.index[:3]:
                    alertas.append({'tipo': 'oportunidad', 'nivel': 'success', 'icono': '🌟', 'mensaje': f"Nicho premium: {pais}", 'detalle': f"Bajo volumen ({paises_nicho.loc[pais, 'cantidad']} prod.) pero alto precio (${paises_nicho.loc[pais, 'precio_promedio']:,.0f})"})

        if 'uva_varietal' in df.columns:
            uvas_validas = df[~df['uva_varietal'].isin(['No especificado', 'Tinto', 'Blanco', '', None])]
            if len(uvas_validas) > 0:
                uvas_count = uvas_validas['uva_varietal'].value_counts()
                uvas_raras = uvas_count[uvas_count < 5]
                if len(uvas_raras) > 0:
                    alertas.append({'tipo': 'catalogo', 'nivel': 'info', 'icono': '🍇', 'mensaje': f"{len(uvas_raras)} uvas con <5 productos", 'detalle': "Oportunidad de diferenciación con uvas poco comunes"})
    except Exception as e:
        st.sidebar.error(f"⚠️ Error generando alertas: {e}")
        alertas = []

    st.subheader(f"📋 Panel de Alertas ({len(alertas)})")
    if alertas:
        criticas = [a for a in alertas if a['nivel'] == 'error']
        advertencias = [a for a in alertas if a['nivel'] == 'warning']
        oportunidades = [a for a in alertas if a['nivel'] == 'success' or a['tipo'] == 'oportunidad']
        info = [a for a in alertas if a['nivel'] == 'info' and a['tipo'] != 'oportunidad']

        if criticas:
            with st.expander(f"**🔴 Críticas ({len(criticas)})**", expanded=False):
                for i, alerta in enumerate(criticas):
                    st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**")
                    st.caption(f"_{alerta['detalle']}_")
                    if gemini and st.button(f"💡 Explicar Alerta {i+1}", key=f"explain_crit_{i}", help="Obtener explicación detallada de la IA"):
                        with st.spinner("Generando explicación..."):
                            try:
                                explicacion = gemini.explicar_alerta(alerta, df)
                                st.info(explicacion)
                            except Exception as e:
                                st.error(f"Error al explicar: {e}")
                    if i < len(criticas) - 1:
                        st.markdown("---")

        if advertencias:
            with st.expander(f"**🟡 Advertencias ({len(advertencias)})**", expanded=False):
                for i, alerta in enumerate(advertencias):
                    st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**")
                    st.caption(f"_{alerta['detalle']}_")
                    if gemini and st.button(f"💡 Explicar Alerta {i+1}", key=f"explain_warn_{i}", help="Obtener explicación detallada de la IA"):
                        with st.spinner("Generando explicación..."):
                            try:
                                explicacion = gemini.explicar_alerta(alerta, df)
                                st.info(explicacion)
                            except Exception as e:
                                st.error(f"Error al explicar: {e}")
                    if i < len(advertencias) - 1:
                        st.markdown("---")

        if oportunidades:
            with st.expander(f"**🟢 Oportunidades ({len(oportunidades)})**", expanded=False):
                for i, alerta in enumerate(oportunidades):
                    st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**")
                    st.caption(f"_{alerta['detalle']}_")
                    if gemini and st.button(f"💡 Explicar Alerta {i+1}", key=f"explain_opp_{i}", help="Obtener explicación detallada de la IA"):
                        with st.spinner("Generando explicación..."):
                            try:
                                explicacion = gemini.explicar_alerta(alerta, df)
                                st.info(explicacion)
                            except Exception as e:
                                st.error(f"Error al explicar: {e}")
                    if i < len(oportunidades) - 1:
                        st.markdown("---")

        if info:
            with st.expander(f"**🔵 Información ({len(info)})**", expanded=False):
                for i, alerta in enumerate(info):
                    st.markdown(f"{alerta['icono']} **{alerta['mensaje']}**")
                    st.caption(f"_{alerta['detalle']}_")
                    if gemini and st.button(f"💡 Explicar Alerta {i+1}", key=f"explain_info_{i}", help="Obtener explicación detallada de la IA"):
                        with st.spinner("Generando explicación..."):
                            try:
                                explicacion = gemini.explicar_alerta(alerta, df)
                                st.info(explicacion)
                            except Exception as e:
                                st.error(f"Error al explicar: {e}")
                    if i < len(info) - 1:
                        st.markdown("---")
    else:
        st.success("✅ Sin alertas pendientes")

    st.markdown("---")

    st.header("🔍 Filtros de Mercado")
    tiendas_options = sorted(df['tienda'].unique())
    tipos_options = sorted(df['tipo_vino'].unique())
    paises_options = sorted(df['pais_origen'].dropna().unique())
    segmentos_options = sorted(df['segmento_precio'].dropna().unique())

    with st.expander("🏪 Tiendas", expanded=False):
        tiendas_seleccionadas = st.multiselect(
            "Selecciona Tiendas",
            options=tiendas_options,
            default=tiendas_options,
            label_visibility="collapsed"
        )

    with st.expander("🍷 Tipo de Vino", expanded=False):
        tipos_seleccionados = st.multiselect(
            "Selecciona Tipos de Vino",
            options=tipos_options,
            default=tipos_options,
            label_visibility="collapsed"
        )

    with st.expander("🌍 País de Origen", expanded=False):
        paises_seleccionados = st.multiselect(
            "Selecciona Países de Origen",
            options=paises_options,
            default=paises_options,
            label_visibility="collapsed"
        )

    with st.expander("💰 Segmento de Precio", expanded=False):
        segmentos_seleccionados = st.multiselect(
            "Selecciona Segmentos de Precio",
            options=segmentos_options,
            default=segmentos_options,
            label_visibility="collapsed"
        )

    with st.expander("💲 Rango de Precio (MXN)", expanded=False):
        precio_min_val = float(df['precio_actual'].min())
        precio_max_val = float(df['precio_actual'].max())
        if precio_min_val >= precio_max_val:
            precio_max_val = precio_min_val + 100
        default_range = (precio_min_val, precio_max_val)

        rango_precio = st.slider(
            "Selecciona un Rango de Precio",
            min_value=precio_min_val,
            max_value=precio_max_val,
            value=default_range,
            label_visibility="collapsed"
        )

    st.markdown("---")
    st.header("🔍 Análisis Rápido IA")
    if gemini:
        if st.button("⚡ Análisis Instantáneo", type="primary", use_container_width=True, key="analisis_rapido_sidebar"):
            with st.spinner("Analizando datos globales..."):
                prompt_rapido = f"""
Análisis EXPRESS de estos datos GLOBALES de vinos:

DATOS CLAVE:
- {len(df)} productos
- Precio promedio: ${df['precio_actual'].mean():.2f}
- {df['tienda'].nunique()} tiendas
- {(df['tiene_descuento'].sum() / len(df)) * 100:.1f}% con descuento

Da UN insight clave y UNA acción específica.
Máximo 80 palabras.
"""
                try:
                    response = gemini.model.generate_content(prompt_rapido)
                    st.success("💡 Insight Rápido (Global):")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")

query_parts = []
if tiendas_seleccionadas: query_parts.append("tienda in @tiendas_seleccionadas")
if tipos_seleccionados: query_parts.append("tipo_vino in @tipos_seleccionados")
if paises_seleccionados: query_parts.append("pais_origen in @paises_seleccionados")
if segmentos_seleccionados: query_parts.append("segmento_precio in @segmentos_seleccionados")
if rango_precio and len(rango_precio) == 2 and rango_precio[0] <= rango_precio[1]:
    query_parts.append("precio_actual >= @rango_precio[0] and precio_actual <= @rango_precio[1]")

df_filtrado = df.copy()
if query_parts:
    query = " and ".join(query_parts)
    try:
        df_filtrado = df.query(query, local_dict={
            'tiendas_seleccionadas': tiendas_seleccionadas,
            'tipos_seleccionados': tipos_seleccionados,
            'paises_seleccionados': paises_seleccionados,
            'segmentos_seleccionados': segmentos_seleccionados,
            'rango_precio': rango_precio
        })
    except Exception as e:
        st.error(f"⚠️ Error al aplicar filtros: {e}. Mostrando datos sin filtrar.")
        df_filtrado = df.copy()

if df_filtrado.empty and not df.empty:
    st.error("❌ No hay datos que coincidan con los filtros seleccionados. Por favor, amplía tu selección.")
elif df.empty:
    st.error("❌ El DataFrame original está vacío. No hay datos para mostrar o filtrar.")
    st.stop()

df_precios = quality.get_dataset_for_analysis(df_filtrado, 'precio')
df_catalogo = quality.get_dataset_for_analysis(df_filtrado, 'catalogo')


tab_historico, tab_resumen, tab_precios, tab_catalogo, tab_ia, tab_herramientas, tab_comparativa = st.tabs([
    "📈 Histórico",
    "📊 Resumen",
    "💰 Precios",
    "🍇 Uvas/Nichos",
    "🤖 IA",
    "⚖️ Herramientas",
    "🔬 Comparativo"
])

with tab_historico:
    st.header("📈 Análisis Histórico y Tendencias")
    if historical_analyzer and historical_available:
        st.subheader("📊 Resumen del Historial")
        try:
            stats = historical_analyzer.get_summary_statistics()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Snapshots", stats.get('num_snapshots', 'N/A'))
            col2.metric("Días Cubiertos", stats.get('dias_entre_snapshots', 'N/A'))
            col3.metric("Productos Promedio", f"{stats.get('productos_promedio', 0):,}")
            col4.metric("Precio Promedio Hist.", f"${stats.get('precio_promedio_historico', 0):,.2f}")
        except Exception as e:
            st.error(f"Error al obtener estadísticas: {e}")

        st.markdown("---")
        st.subheader("💰 Evolución de Precios en el Tiempo")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            tipos_hist_options = ['Todos'] + sorted(df['tipo_vino'].dropna().unique())
            filter_tipo = st.selectbox("Filtrar por tipo:", tipos_hist_options, key='hist_tipo')
        with col_f2:
            paises_hist_options = ['Todos'] + sorted(df['pais_origen'].dropna().unique())
            filter_pais = st.selectbox("Filtrar por país:", paises_hist_options, key='hist_pais')

        product_filter = {}
        if filter_tipo != 'Todos': product_filter['tipo_vino'] = filter_tipo
        if filter_pais != 'Todos': product_filter['pais_origen'] = filter_pais

        try:
            evolution = historical_analyzer.get_price_evolution(product_filter=product_filter if product_filter else None)
            if not evolution.empty and 'fecha' in evolution.columns:
                if 'fecha_dt' not in evolution.columns:
                    evolution['fecha_dt'] = pd.to_datetime(evolution['fecha'], format='%Y%m%d')
                evolution = evolution.sort_values('fecha_dt')

                fig_evol = go.Figure()
                fig_evol.add_trace(go.Scatter(
                    x=evolution['fecha_dt'],
                    y=evolution['precio_promedio'],
                    mode='lines+markers',
                    name='Precio Promedio',
                    line=dict(color='#8B0000', width=3)
                ))
                fig_evol.add_trace(go.Scatter(
                    x=evolution['fecha_dt'],
                    y=evolution['precio_mediano'],
                    mode='lines+markers',
                    name='Precio Mediano',
                    line=dict(color='#FF6B6B', width=2, dash='dash')
                ))
                fig_evol.update_layout(
                    title='Evolución Precio Promedio y Mediano',
                    xaxis_title='Fecha',
                    yaxis_title='Precio (MXN)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_evol, use_container_width=True)

                if 'variacion_precio' in evolution.columns:
                    with st.expander("Ver tabla de evolución detallada"):
                        st.dataframe(
                            evolution[['fecha', 'precio_promedio', 'precio_mediano', 'variacion_precio', 'num_productos']].style.format({
                                'precio_promedio': '${:,.2f}',
                                'precio_mediano': '${:,.2f}',
                                'variacion_precio': '{:+.2f}%',
                                'num_productos': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
            else:
                st.info("No hay suficientes datos históricos para mostrar evolución con estos filtros.")
        except Exception as e:
            st.error(f"Error al generar gráfico de evolución: {e}")

        st.markdown("---")
        st.subheader("🔄 Rotación de Inventario (Último Periodo)")
        try:
            rotation = historical_analyzer.calculate_rotation_rate()
            if rotation and 'error' not in rotation:
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Descontinuados",
                    rotation.get('productos_descontinuados', 0),
                    delta=f"{rotation.get('tasa_rotacion_pct', 0)}% rotación"
                )
                col2.metric(
                    "Nuevos",
                    rotation.get('productos_nuevos', 0),
                    delta=f"{rotation.get('tasa_incorporacion_pct', 0)}% incorp."
                )
                col3.metric(
                    "Cambio Neto",
                    f"{rotation.get('cambio_neto', 0):+d}",
                    delta_color="off"
                )
                st.caption(f"Comparando {rotation.get('fecha_anterior', 'N/A')} vs {rotation.get('fecha_actual', 'N/A')}")

                rotation_by_store = historical_analyzer.get_rotation_by_store()
                if not rotation_by_store.empty:
                    fig_rot = px.bar(
                        rotation_by_store,
                        x='tienda',
                        y='tasa_rotacion',
                        title='Tasa de Rotación (%) por Tienda',
                        labels={'tasa_rotacion': 'Tasa Rotación (%)', 'tienda': 'Tienda'},
                        color='tasa_rotacion',
                        color_continuous_scale='Reds',
                        text='tasa_rotacion'
                    )
                    fig_rot.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig_rot, use_container_width=True)
                    st.markdown("""**💡 Interpretación:** < 10%: Estable | 10-20%: Saludable | > 20%: Alta rotación""")
            elif rotation and 'error' in rotation:
                st.info(rotation['error'])
            else:
                st.info("No se pudo calcular la rotación (datos insuficientes o error).")
        except Exception as e:
            st.error(f"Error al calcular rotación: {e}")

        st.markdown("---")
        if len(available_hist_dates) >= 3:
            st.subheader("📊 Tendencias de Precio Detectadas (General)")
            try:
                trends = historical_analyzer.detect_price_trends(min_snapshots=3)
                if trends and 'error' not in trends:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### Por Tipo de Vino")
                        if trends.get('por_tipo'):
                            for tipo, data in trends['por_tipo'].items():
                                st.metric(tipo, data['tendencia'], delta=f"{data['variacion_pct']:+.2f}%")
                        else:
                            st.info("No hay datos suficientes por tipo.")
                    with col2:
                        st.markdown("##### Por País (Top 5)")
                        if trends.get('por_pais'):
                            for pais, data in trends['por_pais'].items():
                                st.metric(pais, data['tendencia'], delta=f"{data['variacion_pct']:+.2f}%")
                        else:
                            st.info("No hay datos suficientes por país.")
                elif trends and 'error' in trends:
                    st.info(trends['error'])
                else:
                    st.info("No se pudieron detectar tendencias (datos insuficientes o error).")
            except Exception as e:
                st.error(f"Error al detectar tendencias: {e}")
        else:
            st.info("ℹ️ Se necesitan al menos 3 snapshots para detectar tendencias de precio.")

        if SimplePredictor and historical_analyzer and len(available_hist_dates) >= 3:
            st.markdown("---")
            st.header("🔮 Predicciones y Proyecciones")

            try:
                predictor = SimplePredictor(historical_analyzer)
                predictor_available = True
            except Exception as e:
                st.error(f"Error al inicializar Predictor: {e}")
                predictor_available = False

            if predictor_available:
                pred_tab1, pred_tab2, pred_tab3 = st.tabs([
                    "💰 Predicción de Precios",
                    "🔄 Tendencia de Rotación",
                    "📅 Patrones Estacionales"
                ])

                with pred_tab1:
                    st.subheader("Predicción de Precios del Próximo Período")
                    st.markdown('Proyección basada en tendencias históricas usando tus datos reales.')

                    segmentos_pred_options = ['Global'] + sorted(df['segmento_precio'].dropna().unique())
                    segmento_pred = st.selectbox(
                        "Segmento a predecir:",
                        segmentos_pred_options,
                        key='pred_segmento'
                    )
                    seg_param = None if segmento_pred == 'Global' else segmento_pred

                    with st.spinner(f"Calculando predicción para {segmento_pred}..."):
                        prediccion = predictor.predict_next_period_prices(seg_param)

                    if 'error' not in prediccion:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Precio Actual", f"${prediccion['precio_actual']:.2f}")
                        with col2:
                            st.metric(
                                "Precio Predicho",
                                f"${prediccion['precio_predicho']:.2f}",
                                delta=f"{prediccion['variacion_pct']:+.2f}%"
                            )
                        with col3:
                            tendencia_txt = prediccion['tendencia'].upper()
                            if prediccion['tendencia'] == 'alcista':
                                st.success(f"📈 {tendencia_txt}")
                            elif prediccion['tendencia'] == 'bajista':
                                st.error(f"📉 {tendencia_txt}")
                            else:
                                st.info(f"➡️ {tendencia_txt}")

                        st.markdown("---")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("##### 📊 Detalles")
                            st.metric("Confianza", prediccion['confianza'].title(), help="Basado en consistencia histórica y R²")
                            st.metric("Basado en Snapshots", prediccion['basado_en_snapshots'])
                            st.metric("Fecha Estimada", prediccion['fecha_prediccion'])
                        with col_b:
                            st.markdown("##### 📈 Rango (95%)")
                            rango_min, rango_max = prediccion['rango_prediccion']
                            st.markdown(f"- **Mínimo:** ${rango_min:.2f}")
                            st.markdown(f"- **Esperado:** ${prediccion['precio_predicho']:.2f}")
                            st.markdown(f"- **Máximo:** ${rango_max:.2f}")
                            st.caption(f"Margen de error: ±${prediccion['margen_error']:.2f}")

                        st.markdown("---")
                        st.markdown("##### 📈 Visualización")
                        evolution_graph = historical_analyzer.get_price_evolution(
                            product_filter={'segmento_precio': seg_param} if seg_param else None
                        )
                        if not evolution_graph.empty:
                            if 'fecha_dt' not in evolution_graph.columns:
                                evolution_graph['fecha_dt'] = pd.to_datetime(evolution_graph['fecha'], format='%Y%m%d')
                            evolution_graph = evolution_graph.sort_values('fecha_dt')

                            fig_pred = go.Figure()
                            fig_pred.add_trace(go.Scatter(
                                x=evolution_graph['fecha_dt'],
                                y=evolution_graph['precio_promedio'],
                                mode='lines+markers',
                                name='Histórico',
                                line=dict(color='#8B0000', width=2),
                                marker=dict(size=8)
                            ))

                            try:
                                fecha_pred_dt_graph = datetime.strptime(prediccion['fecha_prediccion'], '%Y-%m-%d')
                                fig_pred.add_trace(go.Scatter(
                                    x=[fecha_pred_dt_graph],
                                    y=[prediccion['precio_predicho']],
                                    mode='markers',
                                    name='Predicción',
                                    marker=dict(size=15, color='orange', symbol='star', line=dict(color='red', width=2))
                                ))
                                fig_pred.add_trace(go.Scatter(
                                    x=[fecha_pred_dt_graph, fecha_pred_dt_graph],
                                    y=[rango_min, rango_max],
                                    mode='lines',
                                    name='Rango 95%',
                                    line=dict(color='rgba(255,165,0,0.3)', width=10)
                                ))
                                ultimo_historico_dt = evolution_graph['fecha_dt'].iloc[-1]
                                fig_pred.add_trace(go.Scatter(
                                    x=[ultimo_historico_dt, fecha_pred_dt_graph],
                                    y=[prediccion['precio_actual'], prediccion['precio_predicho']],
                                    mode='lines',
                                    name='Proyección',
                                    line=dict(color='gray', width=2, dash='dash')
                                ))

                                fig_pred.update_layout(
                                    title=f'Proyección de Precio - {segmento_pred}',
                                    xaxis_title='Fecha',
                                    yaxis_title='Precio (MXN)',
                                    hovermode='x unified',
                                    template='plotly_white',
                                    height=400
                                )
                                st.plotly_chart(fig_pred, use_container_width=True)
                            except ValueError:
                                st.warning("No se pudo graficar la predicción por formato de fecha inválido.")

                        if gemini:
                            st.markdown("---")
                            if st.button("🧠 Interpretar Predicción con IA", key="interpretar_pred_precio"):
                                with st.spinner("Analizando predicción..."):
                                    prompt_interpret_pred = f"""
Interpreta esta predicción de precios:

**Segmento:** {segmento_pred}
**Precio actual:** ${prediccion['precio_actual']:.2f}
**Precio predicho:** ${prediccion['precio_predicho']:.2f}
**Variación:** {prediccion['variacion_pct']:+.2f}%
**Tendencia:** {prediccion['tendencia']}
**Confianza:** {prediccion['confianza']}
**Basado en:** {prediccion['basado_en_snapshots']} snapshots

Proporciona:
1. ¿Qué significa esta tendencia para el negocio?
2. ¿Qué acciones tomar antes de que llegue esta fecha ({prediccion['fecha_prediccion']})?
3. ¿Es momento de ajustar precios ahora o esperar?

Máximo 150 palabras.
"""
                                    try:
                                        response = gemini.model.generate_content(prompt_interpret_pred)
                                        st.markdown("#### Interpretación IA:")
                                        st.markdown(response.text)
                                    except Exception as e:
                                        st.error(f"Error al interpretar predicción: {e}")
                    else:
                        st.warning(prediccion['error'])

                with pred_tab2:
                    st.subheader("Tendencia de Rotación de Inventario")
                    st.markdown('Proyección de la tasa de descontinuación de productos.')

                    with st.spinner("Calculando tendencia de rotación..."):
                        pred_rot = predictor.predict_rotation_trend()

                    if 'error' not in pred_rot:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tasa Promedio Hist.", f"{pred_rot['tasa_rotacion_promedio']:.2f}%")
                        with col2:
                            st.metric(
                                "Tasa Reciente",
                                f"{pred_rot['tasa_rotacion_reciente']:.2f}%",
                                delta=f"{pred_rot['tasa_rotacion_reciente'] - pred_rot['tasa_rotacion_promedio']:+.2f}pp"
                            )
                        with col3:
                            tendencia_rot = pred_rot['tendencia'].upper()
                            if pred_rot['tendencia'] == 'acelerándose':
                                st.warning(f"⚠️ {tendencia_rot}")
                            elif pred_rot['tendencia'] == 'desacelerándose':
                                st.success(f"✅ {tendencia_rot}")
                            else:
                                st.info(f"➡️ {tendencia_rot}")

                        st.markdown("---")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("##### 📊 Análisis")
                            st.markdown(f"**Interpretación:** {pred_rot['interpretacion']}")
                            st.markdown(f"**Períodos analizados:** {pred_rot['periodos_analizados']}")
                            st.metric("Predicción Próximo Período", f"~{pred_rot['prediccion_proximo_periodo']:.2f}%")
                        with col_b:
                            st.markdown("##### 💡 Recomendaciones")
                            tasa_r = pred_rot['tasa_rotacion_reciente']
                            if tasa_r < 10:
                                st.success("✅ **Rotación Saludable**: Catálogo estable. Considerar renovación gradual.")
                            elif tasa_r < 20:
                                st.info("ℹ️ **Rotación Normal**: Renovación activa. Monitorear salidas.")
                            else:
                                st.warning("⚠️ **Rotación Alta**: Posible volatilidad. Revisar causas y estabilizar core.")
                    else:
                        st.warning(pred_rot['error'])

                with pred_tab3:
                    st.subheader("Patrones Estacionales del Mercado")
                    st.markdown('Identifica comportamientos promedio por mes basado en tu historial.')

                    if len(available_hist_dates) >= 4:
                        with st.spinner("Analizando estacionalidad..."):
                            pred_season = predictor.forecast_seasonal_patterns()

                        if 'error' not in pred_season:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mes Precios Altos", pred_season.get('mes_precios_altos', 'N/A'))
                            with col2:
                                st.metric("Mes Precios Bajos", pred_season.get('mes_precios_bajos', 'N/A'))
                            with col3:
                                st.metric("Mes Más Descuentos", pred_season.get('mes_mas_descuentos', 'N/A'))

                            st.markdown("---")
                            st.metric(
                                "Variación Estacional Precio",
                                f"{pred_season.get('variacion_estacional', 0.0):.1f}%",
                                help="Diferencia % entre mes más caro y más barato"
                            )

                            if 'datos_mensuales' in pred_season and pred_season['datos_mensuales']:
                                st.markdown("---")
                                st.markdown("##### 📊 Patrones Mensuales Promedio")
                                meses_data = pred_season['datos_mensuales']
                                meses_nombres = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
                                               7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}

                                df_meses_list = []
                                for mes_num in range(1, 13):
                                    data = meses_data.get(mes_num, {})
                                    df_meses_list.append({
                                        'mes': meses_nombres.get(mes_num, f'M{mes_num}'),
                                        'mes_num': mes_num,
                                        'precio': data.get('precio_promedio', np.nan),
                                        'descuento': data.get('pct_descuento', np.nan)
                                    })
                                df_meses = pd.DataFrame(df_meses_list).sort_values('mes_num')

                                fig_season = go.Figure()
                                fig_season.add_trace(go.Scatter(
                                    x=df_meses['mes'],
                                    y=df_meses['precio'],
                                    mode='lines+markers',
                                    name='Precio Promedio',
                                    yaxis='y',
                                    line=dict(color='#8B0000', width=2),
                                    marker=dict(size=10),
                                    connectgaps=True
                                ))
                                fig_season.add_trace(go.Scatter(
                                    x=df_meses['mes'],
                                    y=df_meses['descuento'],
                                    mode='lines+markers',
                                    name='% Descuento',
                                    yaxis='y2',
                                    line=dict(color='orange', width=2, dash='dash'),
                                    marker=dict(size=8),
                                    connectgaps=True
                                ))

                                fig_season.update_layout(
                                    title='Comportamiento Mensual Promedio del Mercado',
                                    xaxis=dict(title='Mes'),
                                    yaxis=dict(title='Precio Promedio (MXN)', side='left', zeroline=False),
                                    yaxis2=dict(title='% Descuento', overlaying='y', side='right', zeroline=False),
                                    hovermode='x unified',
                                    template='plotly_white',
                                    height=400
                                )
                                st.plotly_chart(fig_season, use_container_width=True)

                                st.markdown("**💡 Interpretación:**")
                                st.markdown(f"- **Mejor momento para comprar (precios bajos):** {pred_season.get('mes_precios_bajos', 'N/A')}")
                                st.markdown(f"- **Mayor actividad promocional:** {pred_season.get('mes_mas_descuentos', 'N/A')}")
                                st.markdown(f"- **Temporada alta (precios altos):** {pred_season.get('mes_precios_altos', 'N/A')}")

                            st.markdown("---")
                            st.info(f"📊 Análisis basado en {pred_season.get('meses_analizados', 0)} mes(es) con datos. Más snapshots (idealmente >1 año) mejorarán la precisión.")
                        else:
                            st.warning(pred_season['error'])
                    else:
                        st.info("ℹ️ Se necesitan al menos 4 snapshots (cubriendo varios meses) para analizar patrones estacionales.")

                st.markdown("---")
                st.subheader("📄 Resumen Ejecutivo de Predicciones")

                if st.button("📝 Generar Resumen Completo", key="generar_resumen_pred", type="primary"):
                    with st.spinner("Generando resumen..."):
                        resumen = predictor.generate_forecast_summary()
                        st.markdown(resumen)
                        st.download_button(
                            label="💾 Descargar Resumen de Predicciones",
                            data=resumen,
                            file_name=f"predicciones_{datetime.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )
            else:
                st.error("No se pudo inicializar el módulo de predicciones.")

        elif SimplePredictor and historical_analyzer and len(available_hist_dates) < 3:
            st.info(f"""
**🔮 Predicciones no disponibles aún**

Se necesitan al menos 3 snapshots de datos para generar predicciones de precio y rotación.
Se necesitan al menos 4 snapshots para análisis estacional.

**Datos actuales:** {len(available_hist_dates)} snapshot(s)
""")
        elif not SimplePredictor:
            st.warning("Módulo SimplePredictor no disponible.")
        elif not historical_analyzer:
            st.warning("Módulo HistoricalAnalyzer no disponible, no se pueden generar predicciones.")

    elif not historical_available:
        st.warning("⚠️ Se necesitan al menos 2 capturas de datos para análisis histórico.")
        st.info("Ejecuta el scraper periódicamente para acumular historial.")
    else:
        st.error("⚠️ Error inesperado al cargar el historial.")

with tab_resumen:
    st.header("Resumen General del Mercado")
    if 'df_filtrado' in locals() and not df_filtrado.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Productos Filtrados", f"{len(df_filtrado):,}")
        col2.metric("Precio Promedio", f"${df_precios['precio_actual'].mean():,.2f}")
        col3.metric("Con Descuento", f"{df_filtrado['tiene_descuento'].mean() * 100:.1f}%")
        col4.metric("Tiendas Activas", df_filtrado['tienda'].nunique())

        st.markdown("---")
        st.header("📚 Análisis Básico de Catálogo")
        col1_cat, col2_cat = st.columns(2)

        with col1_cat:
            st.subheader("Distribución por País de Origen")
            pais_count = df_catalogo['pais_origen'].value_counts().head(10)
            if not pais_count.empty:
                pais_df = pais_count.reset_index()
                pais_df.columns = ['País', 'Cantidad']
                fig_pais = px.bar(
                    pais_df,
                    x='Cantidad',
                    y='País',
                    orientation='h',
                    title="Top 10 Países",
                    labels={'Cantidad': 'Productos', 'País': 'País'},
                    color='Cantidad',
                    color_continuous_scale='Reds',
                    text='Cantidad'
                )
                fig_pais.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_pais, use_container_width=True)
            else:
                st.info("No hay datos de país para mostrar con los filtros actuales.")

        with col2_cat:
            st.subheader("Distribución por Tipo de Vino")
            tipo_count = df_catalogo['tipo_vino'].value_counts()
            if not tipo_count.empty:
                fig_tipo = px.pie(
                    values=tipo_count.values,
                    names=tipo_count.index,
                    title="Composición por Tipo",
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.3
                )
                st.plotly_chart(fig_tipo, use_container_width=True)
            else:
                st.info("No hay datos de tipo de vino para mostrar con los filtros actuales.")

        st.markdown("---")
        if gemini:
            st.subheader("🧠 Análisis IA de Esta Sección")
            if st.button("💡 Analizar Resumen y Catálogo", key="analizar_tab_resumen", type="primary"):
                with st.spinner("Analizando catálogo..."):
                    current_tab_name = 'catalogo'
                    alertas_relacionadas = [a for a in alertas if a['tipo'] == 'catalogo' or a['tipo'] == 'oportunidad']
                    analisis = gemini.analizar_pestana_actual(
                        df_filtrado,
                        pestana=current_tab_name,
                        alertas=alertas_relacionadas
                    )
                    st.markdown(analisis)
    else:
        st.warning("No hay datos disponibles con los filtros seleccionados.")

with tab_precios:
    st.header("Análisis Detallado de Precios y Competencia")
    if 'df_filtrado' in locals() and not df_filtrado.empty:
        st.subheader("💰 Análisis de Precios")
        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "Distribución General",
            "Comparativa por Tienda",
            "Precio por Tipo"
        ])

        with sub_tab1:
            fig_hist = px.histogram(
                df_precios,
                x='precio_actual',
                nbins=50,
                title="Distribución de Precios",
                labels={'precio_actual': 'Precio (MXN)', 'count': 'Cantidad'},
                color_discrete_sequence=['#8B0000']
            )
            median_price = df_precios['precio_actual'].median()
            if pd.notna(median_price):
                fig_hist.add_vline(
                    x=median_price,
                    line_dash="dash",
                    annotation_text=f"Mediana: ${median_price:,.2f}"
                )
            st.plotly_chart(fig_hist, use_container_width=True)

        with sub_tab2:
            fig_box = px.box(
                df_precios,
                x='tienda',
                y='precio_actual',
                title="Rangos de Precio por Tienda",
                labels={'precio_actual': 'Precio (MXN)', 'tienda': 'Tienda'},
                color='tienda'
            )
            st.plotly_chart(fig_box, use_container_width=True)

            st.subheader("🎻 Densidad de Precios por Competidor")
            fig_violin = px.violin(
                df_filtrado,
                x='tienda',
                y='precio_actual',
                color='tienda',
                box=True,
                title='Distribución y Densidad de Precios',
                labels={'precio_actual': 'Precio (MXN)', 'tienda': 'Tienda'}
            )
            st.plotly_chart(fig_violin, use_container_width=True)

        with sub_tab3:
            if 'tipo_vino' in df_precios.columns:
                precio_tipo = df_precios.groupby('tipo_vino', observed=True)['precio_actual'].agg(['mean', 'count']).reset_index()
                precio_tipo = precio_tipo[precio_tipo['count'] > 5].sort_values('mean', ascending=False)
                if not precio_tipo.empty:
                    fig_tipo_mean = px.bar(
                        precio_tipo,
                        x='tipo_vino',
                        y='mean',
                        title="Precio Promedio por Tipo ( >5 prod.)",
                        labels={'mean': 'Precio Promedio (MXN)', 'tipo_vino': 'Tipo'},
                        color='mean',
                        color_continuous_scale='Reds',
                        text='mean'
                    )
                    fig_tipo_mean.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                    st.plotly_chart(fig_tipo_mean, use_container_width=True)
                else:
                    st.info("No hay suficientes tipos de vino (>5 productos) para mostrar precio promedio.")
            else:
                st.warning("Columna 'tipo_vino' no encontrada.")

        st.markdown("---")
        st.subheader("🏪 Mapa de Competitividad")
        if 'nombre' in df_filtrado.columns and 'precio_actual' in df_filtrado.columns:
            df_competencia = df_filtrado.groupby('tienda', observed=True).agg(
                num_vinos=('nombre', 'count'),
                precio_promedio=('precio_actual', 'mean')
            ).reset_index()
            if not df_competencia.empty:
                fig_competidores = px.scatter(
                    df_competencia,
                    x='num_vinos',
                    y='precio_promedio',
                    size='num_vinos',
                    color='precio_promedio',
                    text='tienda',
                    title='Posicionamiento: Catálogo vs. Precio',
                    labels={'num_vinos': 'No. Vinos', 'precio_promedio': 'Precio Promedio (MXN)'},
                    color_continuous_scale='RdYlGn_r',
                    size_max=60
                )
                fig_competidores.update_traces(textposition='top center')
                st.plotly_chart(fig_competidores, use_container_width=True)
            else:
                st.info("No hay datos suficientes para generar mapa de competitividad.")
        else:
            st.warning("Columnas 'nombre' o 'precio_actual' no encontradas para mapa de competitividad.")

        st.markdown("---")
        st.subheader("📊 Elasticidad por Segmento")
        if 'segmento_precio' in df_precios.columns and 'precio_actual' in df_precios.columns:
            elasticidad = df_precios.groupby('segmento_precio', observed=True).agg(
                precio_min=('precio_actual', 'min'),
                precio_max=('precio_actual', 'max'),
                precio_mean=('precio_actual', 'mean'),
                precio_std=('precio_actual', 'std'),
                count=('precio_actual', 'count')
            ).reset_index()
            elasticidad.columns = ['segmento', 'min', 'max', 'mean', 'std', 'n_count']
            elasticidad['coef_variacion'] = np.where(
                elasticidad['mean'] > 0,
                (elasticidad['std'] / elasticidad['mean']) * 100,
                0
            )

            if not elasticidad.empty:
                fig_elasticidad = px.scatter(
                    elasticidad,
                    x='mean',
                    y='n_count',
                    size='coef_variacion',
                    color='coef_variacion',
                    text='segmento',
                    title='Volumen vs. Precio Promedio (Tamaño = Variabilidad)',
                    labels={'mean': 'Precio Promedio', 'n_count': 'Cantidad', 'coef_variacion': 'Variabilidad (%)'},
                    color_continuous_scale='RdYlGn_r'
                )
                fig_elasticidad.update_traces(textposition='top center')
                st.plotly_chart(fig_elasticidad, use_container_width=True)
                st.markdown("""**💡 Decisión Estratégica:** Burbujas grandes = Alta variabilidad = Clientes menos sensibles al precio. Burbujas pequeñas = Precios consistentes = Guerra de precios.""")
            else:
                st.info("No hay datos suficientes por segmento para análisis de elasticidad.")
        else:
            st.warning("Columnas 'segmento_precio' o 'precio_actual' no encontradas para análisis de elasticidad.")

        st.markdown("---")
        st.subheader("🔄 Indicadores de Rotación (Descuentos)")
        if 'tienda' in df_filtrado.columns and 'tiene_descuento' in df_filtrado.columns and 'nombre' in df_filtrado.columns:
            df_rotacion = df_filtrado.groupby('tienda', observed=True).agg(
                con_descuento=('tiene_descuento', 'sum'),
                total=('nombre', 'count')
            ).reset_index()
            df_rotacion = df_rotacion[df_rotacion['total'] > 0]
            df_rotacion['%_descuento'] = (df_rotacion['con_descuento'] / df_rotacion['total']) * 100
            if not df_rotacion.empty:
                fig_rotacion = px.bar(
                    df_rotacion.sort_values('%_descuento', ascending=False),
                    x='tienda',
                    y='%_descuento',
                    title='% Productos con Descuento por Tienda',
                    labels={'%_descuento': '% Descuento', 'tienda': 'Tienda'},
                    color='%_descuento',
                    color_continuous_scale='Reds',
                    text='%_descuento'
                )
                fig_rotacion.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_rotacion, use_container_width=True)
                st.markdown("""**💡 Insight de Inventario:** > 30% descuento: Posible baja rotación/agresividad. < 15%: Buena rotación/marca fuerte.""")
            else:
                st.info("No hay datos suficientes por tienda para análisis de descuentos.")
        else:
            st.warning("Columnas 'tienda', 'tiene_descuento' o 'nombre' no encontradas para análisis de descuentos.")

        st.markdown("---")
        if gemini:
            st.subheader("🧠 Análisis IA de Esta Sección")
            if st.button("💡 Analizar Pestaña de Precios", key="analizar_tab_precios", type="primary"):
                with st.spinner("Analizando datos de precios..."):
                    analisis = gemini.analizar_pestana_actual(
                        df_filtrado,
                        pestana='precios',
                        alertas=[a for a in alertas if a['tipo'] == 'precio']
                    )
                    st.markdown(analisis)
    else:
        st.warning("No hay datos disponibles con los filtros seleccionados.")

with tab_catalogo:
    st.header("Análisis de Variedad y Oportunidades")
    if 'df_filtrado' in locals() and not df_filtrado.empty:
        st.subheader("🍇 Cuota de Mercado por Tipo de Uva")
        if 'uva_varietal' in df_filtrado.columns:
            df_uvas_treemap = df_filtrado[
                ~df_filtrado['uva_varietal'].isin(['No especificado', 'Tinto', 'Blanco', '', None]) &
                df_filtrado['uva_varietal'].notna()
            ].copy()
            if not df_uvas_treemap.empty:
                df_uvas_treemap_counts = df_uvas_treemap['uva_varietal'].value_counts().nlargest(20).reset_index()
                df_uvas_treemap_counts.columns = ['uva_varietal', 'cantidad']
                fig_treemap = px.treemap(
                    df_uvas_treemap_counts,
                    path=['uva_varietal'],
                    values='cantidad',
                    title='Distribución por Uva (Top 20)',
                    color='cantidad',
                    color_continuous_scale='Reds',
                    labels={'cantidad': 'Cantidad'}
                )
                st.plotly_chart(fig_treemap, use_container_width=True)
                st.markdown("**Insight:** Rectángulos grandes = uvas dominantes. Balancea inventario.")
            else:
                st.info("No hay datos válidos de uvas para mostrar con los filtros actuales.")
        else:
            st.info("Columna 'uva_varietal' no encontrada.")

        st.markdown("---")
        st.subheader("💎 Oportunidades de Nicho")
        if 'tipo_vino' in df_catalogo.columns and 'pais_origen' in df_catalogo.columns:
            combinaciones_nicho = df_catalogo.groupby(['tipo_vino', 'pais_origen'], observed=True).size().reset_index(name='cantidad')
            if not combinaciones_nicho.empty:
                try:
                    pivot = combinaciones_nicho.pivot_table(
                        index='tipo_vino',
                        columns='pais_origen',
                        values='cantidad',
                        aggfunc='sum'
                    ).fillna(0)
                    if not pivot.empty:
                        fig_heatmap = px.imshow(
                            pivot,
                            title="Mapa de Calor: Tipo vs. País",
                            labels={'x': 'País', 'y': 'Tipo', 'color': 'Cantidad'},
                            color_continuous_scale='YlOrRd',
                            aspect='auto'
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.info("No se pudo generar el pivot para el mapa de calor.")
                except Exception as e:
                    st.error(f"Error generando mapa de calor: {e}")
            else:
                st.info("No hay datos para calcular combinaciones de nicho.")
        else:
            st.warning("Columnas 'tipo_vino' o 'pais_origen' no encontradas para mapa de calor.")

        st.markdown("---")
        st.subheader("🎯 Índice de Saturación del Mercado")
        if 'tipo_vino' in df_catalogo.columns and 'pais_origen' in df_catalogo.columns and 'tienda' in df_catalogo.columns and 'nombre' in df_catalogo.columns:
            saturacion = df_catalogo.groupby(['tipo_vino', 'pais_origen'], observed=True).agg(
                tiendas=('tienda', 'nunique'),
                productos=('nombre', 'count')
            ).reset_index()
            saturacion['indice_saturacion'] = np.where(
                saturacion['tiendas'] > 0,
                saturacion['productos'] / saturacion['tiendas'],
                0
            )
            saturacion = saturacion[saturacion['productos'] > 0].sort_values('indice_saturacion').head(10)
            if not saturacion.empty:
                fig_saturacion = px.bar(
                    saturacion,
                    x='indice_saturacion',
                    y=saturacion['tipo_vino'] + ' - ' + saturacion['pais_origen'].fillna('N/A'),
                    orientation='h',
                    title='Top 10 Nichos Menos Saturados',
                    labels={'indice_saturacion': 'Índice (Prod/Tienda)', 'y': 'Nicho'},
                    color='indice_saturacion',
                    color_continuous_scale='Greens_r'
                )
                fig_saturacion.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_saturacion, use_container_width=True)
                st.markdown("""**💡 Interpretación:** Índice bajo (< 5): Poca competencia. Índice alto (> 15): Mercado saturado.""")
            else:
                st.info("No hay datos suficientes para calcular índice de saturación.")
        else:
            st.warning("Faltan columnas ('tipo_vino', 'pais_origen', 'tienda', 'nombre') para índice de saturación.")

        st.markdown("---")
        st.subheader("🎁 Catálogo Inicial Recomendado")
        st.markdown("Basado en popularidad, disponibilidad y precio:")
        if 'uva_varietal' in df_catalogo.columns and 'precio_actual' in df_catalogo.columns and 'nombre' in df_catalogo.columns and 'tienda' in df_catalogo.columns:
            recomendaciones = df_catalogo.dropna(subset=['tipo_vino', 'pais_origen', 'uva_varietal', 'precio_actual']).groupby(
                ['tipo_vino', 'pais_origen', 'uva_varietal'],
                observed=True
            ).agg(
                precio_promedio=('precio_actual', 'mean'),
                frecuencia=('nombre', 'count'),
                tiendas=('tienda', 'nunique')
            ).reset_index()
            recomendaciones['score'] = np.where(
                recomendaciones['precio_promedio'] > 0,
                (recomendaciones['frecuencia'] * 0.4) + (recomendaciones['tiendas'] * 0.3) + ((1 / (recomendaciones['precio_promedio'] / 500 + 1e-6)) * 0.3),
                0
            )
            top_20 = recomendaciones.sort_values('score', ascending=False).head(20)
            if not top_20.empty:
                st.dataframe(
                    top_20[['tipo_vino', 'pais_origen', 'uva_varietal', 'precio_promedio', 'frecuencia']].rename(
                        columns={'uva_varietal': 'uva'}
                    ).style.format({'precio_promedio': '${:,.2f}'}),
                    use_container_width=True
                )
                csv_data = top_20.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Recomendación (CSV)",
                    data=csv_data,
                    file_name='catalogo_recomendado.csv',
                    mime='text/csv'
                )
            else:
                st.info("No se pudieron generar recomendaciones con los datos actuales.")
        else:
            st.warning("Faltan columnas ('uva_varietal', 'precio_actual', 'nombre', 'tienda') para generar recomendación.")

        st.markdown("---")
        if gemini:
            st.subheader("🧠 Análisis IA de Esta Sección")
            if st.button("💡 Analizar Nichos y Uvas", key="analizar_tab_nichos", type="primary"):
                with st.spinner("Analizando nichos y oportunidades..."):
                    analisis = gemini.analizar_pestana_actual(
                        df_filtrado,
                        pestana='nichos',
                        alertas=[a for a in alertas if a['tipo'] == 'oportunidad']
                    )
                    st.markdown(analisis)
    else:
        st.warning("No hay datos disponibles con los filtros seleccionados.")

with tab_ia:
    st.header("🤖 Análisis y Herramientas con Inteligencia Artificial")
    if gemini:
        if 'df_filtrado' not in locals() or df_filtrado.empty:
            st.warning("Aplica filtros válidos en la barra lateral para usar las herramientas de IA.")
        else:
            sub_tab_ia1, sub_tab_ia2, sub_tab_ia3, sub_tab_ia4 = st.tabs([
                "🎯 Análisis de Producto",
                "📦 Catálogo Inicial",
                "💡 Insights del Segmento",
                "❓ Pregúntale al Experto IA"
            ])

            with sub_tab_ia1:
                st.markdown("##### Analizar Posicionamiento de un Producto")
                col1, col2 = st.columns([3,1])
                with col1:
                    producto_buscar = st.text_input(
                        "Nombre (o parte):",
                        placeholder="Ej: Casa Madero Cabernet",
                        key="ia_prod_buscar"
                    )
                with col2:
                    analizar_prod_btn = st.button(
                        "🔍 Analizar",
                        key="analizar_producto_main",
                        type="primary",
                        use_container_width=True
                    )

                if analizar_prod_btn:
                    if producto_buscar:
                        with st.spinner("Analizando producto..."):
                            try:
                                analisis = gemini.analizar_producto_especifico(producto_buscar, df_filtrado)
                                st.markdown(analisis)
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        st.warning("Ingresa un nombre.")

            with sub_tab_ia2:
                st.markdown("##### Recomendación de Catálogo Inicial")
                col1, col2 = st.columns([3,1])
                with col1:
                    presupuesto = st.number_input(
                        "Presupuesto (MXN):",
                        10000,
                        1000000,
                        50000,
                        5000,
                        key="ia_presupuesto"
                    )
                with col2:
                    generar_cat_btn = st.button(
                        "📦 Generar",
                        key="generar_catalogo_main",
                        type="primary",
                        use_container_width=True
                    )

                if generar_cat_btn:
                    with st.spinner("Generando recomendación..."):
                        try:
                            recomendacion = gemini.generar_recomendacion_catalogo(df_filtrado, presupuesto)
                            st.markdown(recomendacion)
                        except Exception as e:
                            st.error(f"Error: {e}")

            with sub_tab_ia3:
                st.markdown("##### Obtener Insights del Mercado Filtrado")
                st.caption("Análisis basado en alertas globales y datos filtrados.")

                if st.button("🧠 Generar Análisis del Segmento", key="analisis_segmento_main", type="primary", use_container_width=True):
                    with st.spinner("Analizando segmento..."):
                        try:
                            analisis_segmento = gemini.analizar_alertas_inteligentes(df_filtrado, alertas)
                            st.markdown("---")
                            st.markdown(analisis_segmento)
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"Error: {e}")

                st.markdown("---")
                st.markdown("##### Análisis de Alertas Globales (Todo el Mercado)")
                st.caption("Pulsa para obtener un resumen estratégico basado en **todas** las alertas del mercado completo (ignora filtros).")

                if st.button("🧠 Analizar Alertas Globales con IA", key="analisis_alertas_main"):
                    with st.spinner("Analizando alertas globales..."):
                        try:
                            analisis_global_ia = gemini.analizar_alertas_inteligentes(df, alertas)
                            st.markdown("---")
                            st.markdown(analisis_global_ia)
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"Error al generar análisis IA: {e}")

            with sub_tab_ia4:
                st.subheader("❓ Consultor IA de Negocios de Vino")
                st.markdown("Selecciona una pregunta común o escribe la tuya. El análisis se basa en tus datos actuales filtrados.")

                categorias = {
                    "📊 Estrategia de Precios": [
                        "¿Mis precios son competitivos comparados con el mercado?",
                        "¿En qué segmento de precio debería enfocarme?",
                        "¿Dónde puedo aumentar precios sin perder competitividad?",
                        "¿Qué productos están mal preciados?"
                    ],
                    "📚 Gestión de Catálogo": [
                        "¿Qué vino debería agregar a mi catálogo?",
                        "¿Qué tipo de vino me falta para completar mi oferta?",
                        "¿Tengo demasiada variedad o muy poca?",
                        "¿Qué productos debería descontinuar?"
                    ],
                    "🎯 Competencia": [
                        "¿Qué competidor debo vigilar más de cerca?",
                        "¿En qué me diferencia La Europea/Costco/HEB?",
                        "¿Dónde está la oportunidad vs mis competidores?",
                        "¿Cómo me comparo con el líder del mercado?"
                    ],
                    "💰 Rentabilidad": [
                        "¿Dónde estoy perdiendo dinero?",
                        "¿Qué segmento es más rentable?",
                        "¿Cuánto margen puedo esperar en cada categoría?",
                        "¿Vale la pena entrar al segmento premium?"
                    ],
                    "🚀 Crecimiento": [
                        "¿Cuál es la mejor oportunidad de crecimiento inmediato?",
                        "¿Qué nicho tiene menos competencia?",
                        "¿Debo especializarme o diversificar?",
                        "¿Qué estrategia me recomiendas para los próximos 6 meses?"
                    ],
                    "🛒 Cliente y Demanda": [
                        "¿Qué buscan los clientes según estos datos?",
                        "¿Cuál es el perfil de producto ideal?",
                        "¿Qué promociones funcionarían mejor?",
                        "¿Cómo afecta la estacionalidad a mi negocio?"
                    ]
                }

                categoria_seleccionada = st.selectbox(
                    "Selecciona una categoría:",
                    list(categorias.keys()),
                    key="categoria_pregunta"
                )

                if 'last_category' not in st.session_state or st.session_state.last_category != categoria_seleccionada:
                    st.session_state.last_category = categoria_seleccionada
                    st.session_state.pregunta_key = f"pregunta_predefinida_{categoria_seleccionada}"

                pregunta_opts = [""] + categorias[categoria_seleccionada]
                pregunta_seleccionada = st.selectbox(
                    "Elige tu pregunta:",
                    pregunta_opts,
                    key=st.session_state.get("pregunta_key", "pregunta_predefinida_default")
                )

                st.markdown("##### O escribe tu propia pregunta:")
                pregunta_custom = st.text_area(
                    "Pregunta personalizada:",
                    placeholder="Ej: ¿Cuántos vinos de cada tipo debería tener en inventario?",
                    key="pregunta_custom",
                    height=80
                )

                pregunta_final = pregunta_custom if pregunta_custom else pregunta_seleccionada

                col1_q, col2_q, col3_q = st.columns([1, 2, 1])
                with col2_q:
                    analizar_pregunta = st.button(
                        "🧠 Obtener Respuesta del Experto IA",
                        key="analizar_pregunta_ia",
                        type="primary",
                        use_container_width=True,
                        disabled=not pregunta_final
                    )

                if analizar_pregunta and pregunta_final:
                    st.markdown("---")
                    st.markdown(f"**Pregunta:** _{pregunta_final}_")
                    st.markdown("---")
                    with st.spinner("El experto IA está analizando tus datos..."):
                        try:
                            respuesta = gemini.responder_pregunta_predefinida(pregunta_final, df_filtrado)
                            st.markdown("### 💡 Respuesta del Experto:")
                            st.markdown(respuesta)
                            st.markdown("---")
                            st.download_button(
                                label="📥 Descargar Respuesta",
                                data=f"Pregunta: {pregunta_final}\n\nRespuesta:\n{respuesta}",
                                file_name=f"consulta_ia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Error al procesar pregunta: {e}")
                elif not pregunta_final and analizar_pregunta:
                    st.warning("Por favor, selecciona o escribe una pregunta.")
                elif not pregunta_final:
                    st.info("👆 Selecciona o escribe una pregunta para comenzar")

                st.markdown("---")
                with st.expander("💡 Tips para Mejores Resultados"):
                    st.markdown("""**Cómo hacer buenas preguntas:**
✅ **Específicas:** "¿Qué Cabernet debería agregar?" vs "¿Qué vino agregar?"
✅ **Con contexto:** "Tengo $50k, ¿qué comprar?" vs "¿Qué comprar?"
✅ **Accionables:** "¿Cómo aumentar margen?" vs "¿Qué opinas del mercado?"
✅ **Basadas en datos:** Las respuestas se generan analizando TUS datos filtrados

**Limitaciones:**
- Las respuestas son sugerencias, no garantías
- Se basan en datos históricos y patrones de mercado
- Valida las recomendaciones con tu experiencia local""")
    else:
        st.warning("⚠️ Las herramientas IA no están disponibles.")
        st.info("Verifica tu API Key de Gemini en el archivo .env o secrets.")

with tab_herramientas:
    st.header("Calculadoras y Simuladores")

    if PriceSimulator and 'df_filtrado' in locals() and not df_filtrado.empty:
        st.subheader("🔮 Simulador de Escenarios de Precio")
        try:
            segmentos_disponibles_sim = sorted(df_filtrado['segmento_precio'].dropna().unique())
            simulator = PriceSimulator(df_filtrado)

            sim_tab1, sim_tab2, sim_tab3, sim_tab4 = st.tabs([
                "💰 Ajuste por Segmento",
                "🏷️ Estrategia de Descuentos",
                "📊 Cambio de Mix",
                "📈 Elasticidad de Demanda"
            ])

            with sim_tab1:
                st.markdown("Ajusta los precios por segmento y observa el impacto en margen.")
                col1, col2 = st.columns(2)
                changes_by_segment = {}

                with col1:
                    st.markdown("##### Ajustes de Precio (%)")
                    for i, segmento in enumerate(segmentos_disponibles_sim):
                        cambio = st.slider(f"{segmento}", -30, 30, 0, key=f"cambio_{segmento}")
                        if cambio != 0:
                            changes_by_segment[segmento] = cambio
                    costo_operativo = st.slider("Costo operativo estimado (% del precio)", 40, 80, 60, key="sim1_costo")

                with col2:
                    st.markdown("##### Impacto Proyectado")
                    if changes_by_segment:
                        df_sim, impacto = simulator.simulate_price_changes(changes_by_segment, costo_operativo_pct=costo_operativo)
                        st.metric("Precio Promedio Actual", f"${impacto['precio_actual_promedio']:,.2f}")
                        st.metric(
                            "Precio Promedio Simulado",
                            f"${impacto['precio_simulado_promedio']:,.2f}",
                            delta=f"{impacto['variacion_precio_pct']:+.2f}%"
                        )
                        st.metric("Margen Actual", f"{impacto['margen_actual_pct']:.1f}%")
                        st.metric(
                            "Margen Simulado",
                            f"{impacto['margen_simulado_pct']:.1f}%",
                            delta=f"{impacto['variacion_margen_pp']:+.1f} pp"
                        )
                        st.metric(
                            "Productos Afectados",
                            f"{impacto['productos_afectados']:,} de {impacto['total_productos']:,}"
                        )
                    else:
                        st.info("👆 Ajusta los sliders para ver el impacto")

                if gemini and changes_by_segment:
                    st.markdown("---")
                    if st.button("🧠 Analizar Este Escenario con IA", key="analizar_escenario_precios"):
                        with st.spinner("Analizando escenario..."):
                            prompt = f"""Analiza este escenario de cambio de precios:
Cambios: {changes_by_segment}
Impacto: {impacto}

Proporciona:
1. Riesgo (bajo/medio/alto)
2. Impacto en competitividad
3. Recomendación (implementar, ajustar, rechazar)

Máximo 150 palabras."""
                            try:
                                response = gemini.model.generate_content(prompt)
                                st.markdown(response.text)
                            except Exception as e:
                                st.error(f"Error: {e}")

            with sim_tab2:
                st.markdown("Define qué porcentaje de tu catálogo debería tener descuento.")
                col1, col2 = st.columns(2)

                with col1:
                    descuento_actual = (df_filtrado['tiene_descuento'].sum() / len(df_filtrado)) * 100
                    st.metric("% Actual con Descuento", f"{descuento_actual:.1f}%")
                    target_discount = st.slider("% Objetivo con Descuento", 0, 50, int(descuento_actual))
                    segmentos_descuento = st.multiselect(
                        "Aplicar solo a estos segmentos:",
                        segmentos_disponibles_sim,
                        default=segmentos_disponibles_sim,
                        key="sim2_segmentos"
                    )

                with col2:
                    if target_discount != int(descuento_actual):
                        df_sim_disc, impacto_disc = simulator.simulate_discount_strategy(
                            target_discount,
                            segmentos_descuento if segmentos_descuento else None
                        )
                        st.metric("Productos con Descuento Actual", f"{impacto_disc['productos_con_descuento_actual']}")
                        st.metric(
                            "Productos con Descuento Simulado",
                            f"{impacto_disc['productos_con_descuento_simulado']}",
                            delta=f"{impacto_disc['productos_con_descuento_simulado'] - impacto_disc['productos_con_descuento_actual']:+d}"
                        )
                        st.metric(
                            "Impacto en Precio Promedio",
                            f"${abs(impacto_disc['impacto_precio_promedio']):,.2f}",
                            delta=f"{impacto_disc['impacto_precio_promedio']:,.2f}"
                        )
                        if impacto_disc.get('productos_modificados', 0) > 0:
                            st.info(f"📝 Se modificarían {impacto_disc['productos_modificados']} productos")
                        elif target_discount > descuento_actual and len(df_filtrado[~df_filtrado['tiene_descuento']]) == 0:
                            st.warning("⚠️ No hay productos sin descuento para agregar")
                        elif target_discount < descuento_actual and len(df_filtrado[df_filtrado['tiene_descuento']]) == 0:
                            st.warning("⚠️ No hay productos con descuento para quitar")
                        else:
                            st.warning("⚠️ No fue posible aplicar el cambio (revisa filtros)")
                    else:
                        st.info("👆 Ajusta el % objetivo para simular")

                if gemini and target_discount != int(descuento_actual):
                    st.markdown("---")
                    if st.button("🧠 Analizar Estrategia de Descuentos", key="analizar_descuentos"):
                        with st.spinner("Analizando estrategia..."):
                            prompt = f"""Analiza esta estrategia de descuentos:
- Descuento actual: {descuento_actual:.1f}%
- Descuento objetivo: {target_discount}%
- Impacto en precio: ${impacto_disc['impacto_precio_promedio']:,.2f}

Evalúa:
1. ¿Es buena estrategia?
2. ¿Riesgos?
3. ¿Cuándo aplicarla?

Máximo 120 palabras."""
                            try:
                                response = gemini.model.generate_content(prompt)
                                st.markdown(response.text)
                            except Exception as e:
                                st.error(f"Error: {e}")

            with sim_tab3:
                st.markdown("Experimenta con diferentes distribuciones de inventario por segmento.")
                col1, col2 = st.columns(2)
                current_mix = (df_filtrado['segmento_precio'].value_counts(normalize=True) * 100).to_dict()

                with col1:
                    st.markdown("##### Mix Actual")
                    for seg, pct in sorted(current_mix.items()):
                        st.metric(seg, f"{pct:.1f}%")

                with col2:
                    st.markdown("##### Mix Objetivo")
                    target_mix = {}
                    remaining = 100
                    segmentos_sim_sorted = sorted(segmentos_disponibles_sim)

                    if segmentos_sim_sorted:
                        for i, seg in enumerate(segmentos_sim_sorted):
                            default_val = int(current_mix.get(seg, 0))
                            upper_limit = remaining if i < len(segmentos_sim_sorted) - 1 else remaining
                            val = st.number_input(f"{seg} (%)", 0, upper_limit, default_val, key=f"mix_{seg}")
                            target_mix[seg] = val
                            if i < len(segmentos_sim_sorted) - 1:
                                remaining -= val

                        last_seg = segmentos_sim_sorted[-1]
                        target_mix[last_seg] = max(0, remaining)
                        st.metric(f"{last_seg} (%)", f"{max(0, remaining):.0f}")
                    else:
                        st.warning("No hay segmentos definidos en los datos filtrados.")

                    total_pct = sum(target_mix.values())
                    if abs(total_pct - 100) > 1:
                        st.warning(f"⚠️ El total debe sumar 100% (actual: {total_pct:.0f}%)")

                if abs(total_pct - 100) <= 1 and target_mix:
                    df_sim_mix, impacto_mix = simulator.simulate_mix_change(target_mix)
                    st.markdown("---")
                    st.markdown("##### Impacto del Cambio de Mix")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Precio Actual Ponderado", f"${impacto_mix['precio_actual_ponderado']:,.2f}")
                    col_b.metric(
                        "Precio Simulado Ponderado",
                        f"${impacto_mix['precio_simulado_ponderado']:,.2f}",
                        delta=f"{impacto_mix['variacion_precio_pct']:+.2f}%"
                    )
                    col_c.metric("Total Inventario", f"{impacto_mix['total_inventario']:,}")

                    st.markdown("**Cambios Necesarios por Segmento:**")
                    if 'cambios_por_segmento' in impacto_mix:
                        for seg, cambio in impacto_mix['cambios_por_segmento'].items():
                            if cambio != 0:
                                emoji = "➕" if cambio > 0 else "➖"
                                st.caption(f"{emoji} {seg}: {abs(cambio)} productos")

                    if gemini:
                        if st.button("🧠 Analizar Cambio de Mix", key="analizar_mix"):
                            with st.spinner("Analizando mix..."):
                                prompt = f"""Analiza este cambio de mix de productos:
**Mix Actual:**
{chr(10).join([f'- {seg}: {pct:.1f}%' for seg, pct in current_mix.items()])}

**Mix Propuesto:**
{chr(10).join([f'- {seg}: {pct:.1f}%' for seg, pct in target_mix.items()])}

**Impacto:** Precio ponderado cambia {impacto_mix['variacion_precio_pct']:+.2f}%

Evalúa:
1. ¿Sólido?
2. ¿Riesgos?
3. ¿Posicionamiento?

Máximo 150 palabras."""
                                try:
                                    response = gemini.model.generate_content(prompt)
                                    st.markdown(response.text)
                                except Exception as e:
                                    st.error(f"Error: {e}")

            with sim_tab4:
                st.markdown("Estima cómo reaccionará la demanda ante cambios de precio.")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### Configuración")
                    segmento_elasticidad = st.selectbox(
                        "Segmento a analizar:",
                        ['Global'] + list(segmentos_disponibles_sim),
                        key="sim4_segmento"
                    )
                    cambio_precio_test = st.slider("Cambio de precio a simular (%)", -30, 30, 10, key="sim4_cambio")
                    seg_param = None if segmento_elasticidad == 'Global' else segmento_elasticidad
                    elasticity_result = simulator.estimate_demand_elasticity(cambio_precio_test, segment=seg_param)

                with col2:
                    st.markdown("##### Resultados")
                    st.metric(
                        "Elasticidad Estimada",
                        f"{elasticity_result['elasticidad']:.2f}",
                        help="Negativo indica reducción de demanda ante aumento de precio"
                    )
                    st.metric("Cambio Esperado en Demanda", f"{elasticity_result['cambio_demanda_esperado_pct']:+.1f}%")
                    st.metric("Unidades Actuales", f"{elasticity_result['unidades_actuales']:,}")
                    st.metric(
                        "Unidades Proyectadas",
                        f"{elasticity_result['unidades_proyectadas']:,}",
                        delta=f"{elasticity_result['unidades_proyectadas'] - elasticity_result['unidades_actuales']:+,}"
                    )
                    st.info(f"**Interpretación:** {elasticity_result['interpretacion']}")

                st.markdown("---")
                st.markdown("##### Curva de Demanda Estimada")
                df_elasticidad_segmento = df_filtrado[df_filtrado['segmento_precio'] == seg_param] if seg_param else df_filtrado

                if not df_elasticidad_segmento.empty:
                    precio_base = df_elasticidad_segmento['precio_actual'].mean()
                    demanda_base = elasticity_result['unidades_actuales']

                    if demanda_base > 0 and precio_base > 0:
                        cambios_precio = np.arange(-30, 31, 5)
                        precios = [precio_base * (1 + cp / 100) for cp in cambios_precio]
                        demandas = [demanda_base * (1 + (elasticity_result['elasticidad'] * cp / 100)) for cp in cambios_precio]

                        fig_elasticity = go.Figure()
                        fig_elasticity.add_trace(go.Scatter(
                            x=precios,
                            y=demandas,
                            mode='lines+markers',
                            name='Demanda Estimada',
                            line=dict(color='#8B0000', width=2)
                        ))
                        fig_elasticity.add_trace(go.Scatter(
                            x=[precio_base],
                            y=[demanda_base],
                            mode='markers',
                            name='Punto Actual',
                            marker=dict(size=15, color='red', symbol='star')
                        ))

                        precio_simulado = precio_base * (1 + cambio_precio_test / 100)
                        demanda_simulada = elasticity_result['unidades_proyectadas']
                        fig_elasticity.add_trace(go.Scatter(
                            x=[precio_simulado],
                            y=[demanda_simulada],
                            mode='markers',
                            name='Escenario Simulado',
                            marker=dict(size=12, color='orange', symbol='diamond')
                        ))

                        fig_elasticity.update_layout(
                            title=f'Curva de Demanda - {segmento_elasticidad}',
                            xaxis_title='Precio (MXN)',
                            yaxis_title='Demanda Estimada (unidades)',
                            hovermode='closest'
                        )
                        st.plotly_chart(fig_elasticity, use_container_width=True)

                        if gemini:
                            if st.button("🧠 Analizar Elasticidad", key="analizar_elasticidad"):
                                with st.spinner("Analizando elasticidad..."):
                                    prompt = f"""Analiza esta estimación de elasticidad:
**Segmento:** {elasticity_result['segmento']}
**Elasticidad:** {elasticity_result['elasticidad']:.2f} ({elasticity_result['interpretacion']})
**Escenario:** Cambio de precio de {cambio_precio_test:+.1f}% resulta en cambio de demanda de {elasticity_result['cambio_demanda_esperado_pct']:+.1f}%

Proporciona:
1. Estrategia de precio óptima
2. Riesgo del cambio
3. Recomendación

Máximo 120 palabras."""
                                    try:
                                        response = gemini.model.generate_content(prompt)
                                        st.markdown(response.text)
                                    except Exception as e:
                                        st.error(f"Error: {e}")

                        st.caption("Nota: Las elasticidades son estimaciones generales y pueden variar.")
                    else:
                        st.info("No hay suficientes datos (demanda o precio base = 0) en este segmento para graficar la elasticidad.")
                else:
                    st.info(f"No hay productos en el segmento '{seg_param}' con los filtros actuales.")

        except Exception as e:
            st.error(f"Error al inicializar el simulador de precios: {e}")

    elif not PriceSimulator:
        st.warning("⚠️ Módulo PriceSimulator no disponible. Funcionalidad deshabilitada.")
    else:
        st.warning("⚠️ No hay datos filtrados para usar el simulador.")

    st.markdown("---")

    if 'df_filtrado' in locals() and not df_filtrado.empty:
        st.subheader("💰 Simulador de Rentabilidad")
        with st.expander("Expandir Simulador", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Configuración de Costos")
                margen_objetivo = st.slider("Margen objetivo (%)", 10, 100, 40, key="sim_margen")
                costo_operativo = st.number_input(
                    "Costo operativo/botella (MXN)",
                    10.0,
                    200.0,
                    50.0,
                    5.0,
                    key="sim_costo_op"
                )
            with col2:
                st.caption("Precios Sugeridos (Top 5 Tipos por Mediana)")
                if not df_precios.empty and 'tipo_vino' in df_precios.columns:
                    precio_competencia = df_precios.groupby('tipo_vino', observed=True)['precio_actual'].median()
                    num_tipos_mostrar = min(5, len(precio_competencia))
                    if num_tipos_mostrar > 0:
                        for i, tipo in enumerate(precio_competencia.nlargest(num_tipos_mostrar).index):
                            precio_mercado = precio_competencia[tipo]
                            precio_sugerido = precio_mercado * 0.95
                            ganancia = precio_sugerido - costo_operativo
                            margen_real = (ganancia / precio_sugerido) * 100 if precio_sugerido > 0 else 0
                            st.metric(
                                label=f"{tipo}",
                                value=f"${precio_sugerido:,.2f}",
                                delta=f"{margen_real:.1f}% margen"
                            )
                    else:
                        st.info("No hay suficientes tipos de vino (mediana) para mostrar precios sugeridos.")
                else:
                    st.info("No hay datos de precios o tipos de vino para calcular sugerencias.")

        st.markdown("---")
        st.subheader("⚖️ Calculadora de Punto de Equilibrio")
        with st.expander("Expandir Calculadora"):
            col1_pe, col2_pe, col3_pe = st.columns(3)
            precio_prom_default = float(df_precios['precio_actual'].mean()) if not df_precios.empty else 150.0

            with col1_pe:
                costos_fijos_mes = st.number_input("Costos fijos/mes", 10000, 200000, 50000, 1000, key="pe_costos_fijos")
            with col2_pe:
                precio_venta_promedio = st.number_input("Precio venta promedio", 100.0, 5000.0, precio_prom_default, key="pe_precio_venta")
            with col3_pe:
                costo_variable_botella = st.number_input(
                    "Costo variable/botella",
                    50.0,
                    max(50.0, precio_venta_promedio - 1),
                    float(precio_venta_promedio * 0.6),
                    key="pe_costo_variable"
                )

            margen_contribucion = precio_venta_promedio - costo_variable_botella
            if margen_contribucion > 0:
                botellas_equilibrio = costos_fijos_mes / margen_contribucion
                col1_res, col2_res, col3_res = st.columns(3)
                col1_res.metric("Botellas/Mes (Equilibrio)", f"{botellas_equilibrio:,.0f}")
                col2_res.metric("Botellas/Día", f"{(botellas_equilibrio / 30):,.1f}")
                col3_res.metric("Margen/Botella", f"${margen_contribucion:,.2f}")

                max_vol = int(botellas_equilibrio * 2.5) if botellas_equilibrio > 0 else 100
                step = max(1, int(max_vol / 15))
                volumenes = list(range(0, max_vol + step, step))
                ganancias = [(v * margen_contribucion) - costos_fijos_mes for v in volumenes]

                fig_pe = px.line(
                    x=volumenes,
                    y=ganancias,
                    title='Ganancia/Pérdida vs Volumen',
                    labels={'x': 'Botellas/Mes', 'y': 'Ganancia (MXN)'}
                )
                fig_pe.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Punto Equilibrio")
                fig_pe.add_vline(x=botellas_equilibrio, line_dash="dash", line_color="red")
                st.plotly_chart(fig_pe, use_container_width=True)
            else:
                st.error("Margen de contribución negativo o cero. Ajusta precios/costos.")

        st.markdown("---")
        st.subheader("🛒 Concentración del Mercado (HHI)")
        if 'tienda' in df_filtrado.columns:
            market_share = df_filtrado['tienda'].value_counts(normalize=True) * 100
            if not market_share.empty:
                hhi = (market_share ** 2).sum()
                col1_hhi, col2_hhi = st.columns([1, 2])
                with col1_hhi:
                    st.metric("Índice HHI", f"{hhi:,.0f}")
                    if hhi < 1500:
                        st.success("✅ COMPETITIVO")
                    elif hhi < 2500:
                        st.warning("🟡 MODERADO")
                    else:
                        st.error("🔴 CONCENTRADO")
                    st.caption("(<1500 Comp., 1500-2500 Mod., >2500 Conc.)")
                with col2_hhi:
                    fig_hhi = px.pie(
                        values=market_share.values,
                        names=market_share.index,
                        title='Cuota de Mercado por Tienda',
                        hole=0.4
                    )
                    st.plotly_chart(fig_hhi, use_container_width=True)
            else:
                st.info("No hay datos de tiendas para calcular HHI.")
        else:
            st.warning("Columna 'tienda' no encontrada para calcular HHI.")
    else:
        st.warning("⚠️ Aplica filtros válidos para usar las herramientas.")

with tab_comparativa:
    st.header("🔬 Análisis Comparativo Temporal")

    if len(available_hist_dates) >= 2:
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            date1_comp = st.selectbox("Fecha base (anterior):", available_hist_dates, index=1, key="comp_date1")
        with col2:
            date2_comp = st.selectbox("Fecha a comparar (nueva):", available_hist_dates, index=0, key="comp_date2")

        run_comparison_main = False
        with col3:
            st.write("")
            st.write("")
            if date1_comp != date2_comp:
                run_comparison_main = st.button("📊 Comparar", key="compare_main", type="primary", use_container_width=True)
            else:
                st.warning("Selecciona fechas diferentes.")

        if run_comparison_main:
            @st.cache_data
            def load_specific_data(date_str):
                filepath = Path(f'data/consolidated/{date_str}/datos_completos_listos.csv')
                if filepath.exists():
                    try:
                        return pd.read_csv(filepath)
                    except Exception as e:
                        st.error(f"Error cargando {filepath.name}: {e}")
                        return pd.DataFrame()
                else:
                    st.error(f"Archivo no encontrado: {filepath}")
                    return pd.DataFrame()

            df_base = load_specific_data(date1_comp)
            df_compare = load_specific_data(date2_comp)

            if df_base.empty or df_compare.empty:
                st.error("No se pudieron cargar datos para una o ambas fechas seleccionadas.")
            elif not all(col in df_base.columns and col in df_compare.columns for col in ['nombre', 'tienda', 'precio_actual']):
                st.error("Faltan columnas esenciales ('nombre', 'tienda', 'precio_actual') en uno o ambos archivos para comparar.")
            else:
                st.subheader("💰 Evolución de Precios")
                try:
                    df_base['product_id'] = df_base['nombre'].astype(str) + " | " + df_base['tienda'].astype(str)
                    df_compare['product_id'] = df_compare['nombre'].astype(str) + " | " + df_compare['tienda'].astype(str)

                    df_merged = pd.merge(
                        df_base[['product_id', 'nombre', 'precio_actual']],
                        df_compare[['product_id', 'precio_actual']],
                        on='product_id',
                        suffixes=('_anterior', '_nuevo'),
                        how='inner'
                    )

                    if not df_merged.empty:
                        df_merged['cambio_precio'] = df_merged['precio_actual_nuevo'] - df_merged['precio_actual_anterior']
                        df_merged['cambio_pct'] = np.where(
                            df_merged['precio_actual_anterior'].fillna(0) > 0,
                            ((df_merged['cambio_precio'] / df_merged['precio_actual_anterior']) * 100),
                            np.nan
                        )
                        df_merged.replace([np.inf, -np.inf], np.nan, inplace=True)

                        df_con_cambios = df_merged.dropna(subset=['cambio_pct'])
                        df_con_cambios = df_con_cambios[df_con_cambios['cambio_precio'] != 0].sort_values('cambio_pct', ascending=False)

                        st.metric("Productos Comunes Analizados", f"{len(df_merged)} vinos")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("##### 📈 Mayor Aumento")
                            st.dataframe(
                                df_con_cambios[['nombre', 'precio_actual_anterior', 'precio_actual_nuevo', 'cambio_pct']].head(10).style.format({
                                    'precio_actual_anterior': '${:,.2f}',
                                    'precio_actual_nuevo': '${:,.2f}',
                                    'cambio_pct': '{:+.1f}%'
                                })
                            )
                        with col_b:
                            st.markdown("##### 📉 Mayor Reducción")
                            st.dataframe(
                                df_con_cambios[['nombre', 'precio_actual_anterior', 'precio_actual_nuevo', 'cambio_pct']].tail(10).sort_values('cambio_pct').style.format({
                                    'precio_actual_anterior': '${:,.2f}',
                                    'precio_actual_nuevo': '${:,.2f}',
                                    'cambio_pct': '{:+.1f}%'
                                })
                            )
                    else:
                        st.info("No se encontraron productos comunes entre las fechas seleccionadas para comparar precios.")

                    st.subheader("📦 Evolución del Catálogo")
                    set_base = set(df_base['product_id'])
                    set_compare = set(df_compare['product_id'])
                    vinos_nuevos = set_compare - set_base
                    vinos_descontinuados = set_base - set_compare

                    col_c, col_d, col_e = st.columns(3)
                    col_c.metric("Vinos Nuevos", f"{len(vinos_nuevos)}")
                    col_d.metric("Vinos Descontinuados", f"{len(vinos_descontinuados)}")
                    col_e.metric("Cambio Neto", f"{len(set_compare) - len(set_base):+d}", delta_color="off")

                    st.subheader("🏪 Cambio en Catálogo por Tienda")
                    cat_base = df_base['tienda'].value_counts().reset_index().rename(columns={'count': 'vinos_anterior', 'tienda': 'tienda'})
                    cat_compare = df_compare['tienda'].value_counts().reset_index().rename(columns={'count': 'vinos_nuevo', 'tienda': 'tienda'})
                    df_tiendas = pd.merge(cat_base, cat_compare, on='tienda', how='outer').fillna(0)
                    df_tiendas['cambio'] = df_tiendas['vinos_nuevo'] - df_tiendas['vinos_anterior']

                    if not df_tiendas.empty:
                        fig_tiendas_comp = px.bar(
                            df_tiendas,
                            x='tienda',
                            y='cambio',
                            title='Cambio Neto Vinos por Tienda',
                            labels={'tienda': 'Tienda', 'cambio': 'Cambio Neto'},
                            color='cambio',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_tiendas_comp, use_container_width=True)
                    else:
                        st.info("No hay datos de tiendas para comparar catálogos.")
                except KeyError as ke:
                    st.error(f"Error de columna faltante durante la comparación: {ke}. Verifica los archivos CSV.")
                except Exception as ex:
                    st.error(f"Error inesperado durante la comparación: {ex}")
    else:
        st.info("ℹ️ Se necesitan al menos dos carpetas de datos ('./data/consolidated/YYYYMMDD/') para usar la función de comparación.")

st.markdown("---")
st.info("Dashboard creado para el análisis estratégico del mercado de vinos en México.")