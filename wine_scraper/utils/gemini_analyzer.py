# wine_scraper/utils/gemini_analyzer.py
import os
import google.generativeai as genai
from typing import Dict, List, Any
import pandas as pd
# No necesitas dotenv aquí, usamos st.secrets
# from dotenv import load_dotenv
from pathlib import Path
import streamlit as st # Importar streamlit para secrets

# Ya no necesitamos env_loaded
# env_loaded = False

class GeminiAnalyzer:
    """Analizador de datos de vino usando Gemini AI (versión deploy)"""

    def __init__(self):
        """Inicializa el analizador con la API key desde st.secrets"""
        # global env_loaded # No necesario

        # Cargar API key directamente desde st.secrets
        api_key = st.secrets.get("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "❌ No se encontró GEMINI_API_KEY en los Secrets de Streamlit. "
                "Asegúrate de haberla configurado en la configuración de tu app."
            )

        try:
            genai.configure(api_key=api_key)
            # Usaremos el modelo flash consistentemente
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
            print("✅ Gemini Analyzer (deploy) inicializado correctamente.")
        except Exception as e:
            print(f"❌ Error al configurar la API de Gemini (deploy): {e}")
            raise ValueError("No se pudo configurar la API de Gemini. Verifica tu API Key en st.secrets.")

    def analizar_alertas_inteligentes(self, df: pd.DataFrame, alertas_basicas: List[Dict]) -> str:
        """
        Analiza las alertas básicas y genera insights más profundos con IA
        """
        if not self.model: return "⚠️ Modelo Gemini no inicializado."

        contexto = self._preparar_contexto_mercado(df, alertas_basicas)

        prompt = f"""
Eres un experto analista del mercado de vinos en México, especializado en comercio electrónico. Analiza los siguientes datos y alertas de un dashboard para generar insights estratégicos claros y accionables para un dueño de negocio.

CONTEXTO DEL MERCADO (según filtros actuales):
{contexto}

ALERTAS RECIENTES DETECTADAS:
{self._formatear_alertas(alertas_basicas)}

Basado EXCLUSIVAMENTE en la información proporcionada, responde de forma breve y enfocada:

1. **Prioridades Urgentes (máx. 2):** ¿Cuáles son las alertas más críticas que requieren acción inmediata y por qué?
2. **Insight Estratégico:** ¿Qué patrón o tendencia clave se observa al combinar las alertas con el contexto general?
3. **Recomendación Accionable:** Propón una acción concreta que el negocio debería tomar de inmediato.
4. **Oportunidad Potencial (opcional):** Si existen alertas de oportunidad (💎, 🌟), sugiere una idea breve para aprovechar ese nicho.

Usa un lenguaje de negocios claro, directo y enfocado en decisiones.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Error al generar análisis IA: {str(e)}"

    def analizar_producto_especifico(self, producto_nombre: str, df: pd.DataFrame) -> str:
        """Analiza un producto específico y su posicionamiento."""
        if not self.model: return "⚠️ Modelo Gemini no inicializado."
        # Usar iloc[0] puede dar error si producto_df está vacío tras el filtrado inicial
        producto_df = df[df['nombre'].str.contains(producto_nombre, case=False, na=False)]
        if producto_df.empty: return f"❌ No se encontró '{producto_nombre}' con los filtros actuales."

        producto = producto_df.iloc[0] # Ahora es seguro usar iloc[0]

        # Asegurar que las columnas existen antes de filtrar competidores
        competidores = pd.DataFrame() # DataFrame vacío por defecto
        if 'tipo_vino' in df.columns and 'pais_origen' in df.columns and 'nombre' in df.columns:
            if producto['tipo_vino'] and producto['pais_origen']: # Evitar filtrar con NaN
                competidores = df[
                    (df['tipo_vino'] == producto['tipo_vino']) &
                    (df['pais_origen'] == producto['pais_origen']) &
                    (df['nombre'] != producto['nombre'])
                ].head(5)

        # Preparar estadísticas del segmento (manejar posible ausencia de segmento)
        precio_prom_segmento = np.nan
        precio_med_segmento = np.nan
        segmento_producto = producto.get('segmento_precio', 'N/A')
        if 'segmento_precio' in df.columns and segmento_producto != 'N/A':
            df_segmento = df[df['segmento_precio'] == segmento_producto]
            if not df_segmento.empty:
                precio_prom_segmento = df_segmento['precio_actual'].mean()
                precio_med_segmento = df_segmento['precio_actual'].median()

        prompt = f"""
Analiza este producto de vino y su posicionamiento competitivo (según filtros actuales):

PRODUCTO: {producto.get('nombre','N/A')} (${producto.get('precio_actual', 0.0):.2f}, {producto.get('tipo_vino','N/A')}, {producto.get('pais_origen','N/A')}, Segmento: {segmento_producto})
Tienda: {producto.get('tienda','N/A')}
Descuento: {'Sí' if producto.get('tiene_descuento', False) else 'No'}

COMPETIDORES DIRECTOS (mismo tipo y origen, si encontrados):
{self._formatear_competidores(competidores)}

PRECIOS DEL SEGMENTO '{segmento_producto}':
Promedio: ${precio_prom_segmento:.2f}
Mediana: ${precio_med_segmento:.2f}

Proporciona:
1. Evaluación del precio (vs segmento y competidores).
2. Posible ventaja competitiva (si la hay).
3. Recomendación breve de pricing.
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e: return f"⚠️ Error al analizar producto: {str(e)}"

    def generar_recomendacion_catalogo(self, df: pd.DataFrame, presupuesto: float) -> str:
        """Genera recomendación de catálogo inicial."""
        if not self.model: return "⚠️ Modelo Gemini no inicializado."
        # Asegurar columnas necesarias
        if not all(col in df.columns for col in ['segmento_precio', 'precio_actual', 'nombre', 'tiene_descuento']):
             return "⚠️ Faltan columnas ('segmento_precio', 'precio_actual', 'nombre', 'tiene_descuento') para generar recomendación."

        # Usar observed=True y manejar posible división por cero en lambda
        analisis_segmentos = df.groupby('segmento_precio', observed=True).agg(
            precio_prom=('precio_actual', 'mean'),
            precio_med=('precio_actual', 'median'),
            num_prod=('nombre', 'count'),
            pct_desc=('tiene_descuento', lambda x: (x.astype(bool).sum() / len(x)) * 100 if len(x) > 0 else 0)
        ).round(2)

        prompt = f"""
Eres un comprador experto de vinos para una nueva tienda online en México con presupuesto inicial de ${presupuesto:,.2f} MXN.

ANÁLISIS DE SEGMENTOS DEL MERCADO ACTUAL (según filtros):
{analisis_segmentos.to_string()}

PRODUCTOS POPULARES POR SEGMENTO (según filtros):
{self._obtener_top_productos(df)}

Diseña una estrategia de catálogo inicial (15-20 tipos de producto, no marcas específicas):
1. Prioriza segmentos con buen balance precio/volumen encontrados en los datos.
2. Balancea vinos populares (más frecuentes) con algunos nichos interesantes (menos frecuentes).
3. Asigna un % del presupuesto a cada segmento principal identificado.
4. Justifica brevemente la selección basada en los datos provistos.
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e: return f"⚠️ Error al generar recomendación: {str(e)}"

    def explicar_alerta(self, alerta: Dict, df: pd.DataFrame) -> str:
        """
        Explica una alerta específica en lenguaje claro con contexto.
        """
        if not self.model: return "⚠️ Modelo Gemini no inicializado." # Verificación añadida

        prompt = f"""
Eres un consultor experto en retail de vinos. Explica esta alerta de forma clara y accionable para un dueño de negocio:

**ALERTA:**
Tipo: {alerta.get('tipo', 'N/A')}
Nivel: {alerta.get('nivel', 'N/A')}
Mensaje: {alerta.get('mensaje', 'N/A')}
Detalle: {alerta.get('detalle', 'N/A')}

**CONTEXTO DEL MERCADO (FILTRADO ACTUAL):**
- Total productos en vista: {len(df)}
- Precio promedio en vista: ${df['precio_actual'].mean():.2f} MXN
- Rango de precios en vista: ${df['precio_actual'].min():.2f} - ${df['precio_actual'].max():.2f} MXN

Proporciona:
1. **¿Qué significa esto?** (explicación simple y directa)
2. **¿Por qué es importante?** (impacto potencial en el negocio)
3. **¿Qué debo hacer?** (siguiente paso concreto y específico)

Sé muy conciso (máximo 3 frases por punto) y enfócate en la acción.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Error al generar explicación IA: {str(e)}"

    def analizar_pestana_actual(self, df: pd.DataFrame, pestana: str, alertas: list = None) -> str:
        """
        Analiza el contexto de la pestaña actual y genera insights específicos.
        """
        if not self.model: return "⚠️ Modelo Gemini no inicializado."
        if df.empty: return "⚠️ No hay datos filtrados para analizar esta pestaña."

        # Contexto base (manejar posible división por cero)
        num_productos = len(df)
        pct_descuento = (df['tiene_descuento'].astype(bool).sum() / num_productos * 100) if num_productos > 0 else 0

        contexto = f"""
DATOS DE LA PESTAÑA "{pestana.upper()}":
- Productos analizados: {num_productos}
- Precio promedio: ${df['precio_actual'].mean():.2f}
- Precio mediano: ${df['precio_actual'].median():.2f}
- Rango: ${df['precio_actual'].min():.2f} - ${df['precio_actual'].max():.2f}
- % con descuento: {pct_descuento:.1f}%
- Tiendas: {df['tienda'].nunique()}
"""

        # Contexto específico por pestaña
        prompt = "" # Inicializar prompt
        if pestana in ['precios', 'competencia']:
            std_precio = df['precio_actual'].std()
            cv = (std_precio / df['precio_actual'].mean() * 100) if df['precio_actual'].mean() != 0 else 0
            contexto += f"""
MÉTRICAS DE PRECIOS:
- Desviación estándar: ${std_precio:.2f}
- Coeficiente de variación: {cv:.1f}%
- Segmentos presentes: {df['segmento_precio'].nunique() if 'segmento_precio' in df.columns else 'N/A'}
"""
            prompt = f"""
Analiza esta vista de PRECIOS Y COMPETENCIA:
{contexto}
Proporciona: 1. Oportunidad Principal, 2. Riesgo Clave, 3. Benchmark vs mercado típico, 4. Acción Rápida HOY. Máx 200 palabras, accionable.
"""

        elif pestana in ['catalogo', 'resumen']:
            top_paises = df['pais_origen'].value_counts().head(3) if 'pais_origen' in df.columns else pd.Series()
            top_tipos = df['tipo_vino'].value_counts().head(3) if 'tipo_vino' in df.columns else pd.Series()
            contexto += f"""
COMPOSICIÓN DEL CATÁLOGO:
Top 3 Países: {', '.join([f"{p} ({c})" for p, c in top_paises.items()]) if not top_paises.empty else 'N/A'}
Top 3 Tipos: {', '.join([f"{t} ({c})" for t, c in top_tipos.items()]) if not top_tipos.empty else 'N/A'}
"""
            prompt = f"""
Analiza esta vista de CATÁLOGO Y VARIEDAD:
{contexto}
Proporciona: 1. Gap de Catálogo, 2. Oportunidad de Diferenciación, 3. Balance a ajustar, 4. Producto Sugerido a agregar. Máx 200 palabras, enfócate en crecimiento.
"""

        elif pestana in ['nichos', 'uvas']:
            if 'uva_varietal' in df.columns:
                uvas_validas = df[~df['uva_varietal'].isin(['No especificado', 'Tinto', 'Blanco', '', None]) & df['uva_varietal'].notna()]
                top_uvas = uvas_validas['uva_varietal'].value_counts().head(5) if not uvas_validas.empty else pd.Series()
                contexto += f"""
ANÁLISIS DE UVAS:
Uvas más frecuentes: {', '.join([f"{u} ({c})" for u, c in top_uvas.items()]) if not top_uvas.empty else 'N/A'}
"""
            prompt = f"""
Analiza esta vista de NICHOS Y OPORTUNIDADES (Uvas):
{contexto}
Proporciona: 1. Nicho Ganador (potencial), 2. Nicho Saturado (evitar), 3. Estrategia de Entrada a nicho prometedor, 4. Proyección de Rentabilidad (cualitativa). Máx 200 palabras, enfócate en oportunidades.
"""

        else: # Análisis genérico para otras pestañas
            prompt = f"""
Analiza estos datos del mercado de vinos:
{contexto}
Proporciona insights clave: 1. Observación Principal, 2. Riesgo u Oportunidad, 3. Recomendación Accionable. Máximo 150 palabras.
"""

        # Agregar alertas si existen
        if alertas and len(alertas) > 0:
            alertas_texto = "\n".join([f"- [{a.get('nivel','INFO').upper()}] {a.get('mensaje','N/A')}" for a in alertas[:3]])
            prompt += f"\n\nALERTAS ACTIVAS RELEVANTES:\n{alertas_texto}\n\nConsidera estas alertas en tu análisis."

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Error al analizar pestaña: {str(e)}"

    def responder_pregunta_predefinida(self, pregunta: str, df: pd.DataFrame) -> str:
        """
        Responde preguntas predefinidas comunes sobre el negocio.
        """
        if not self.model: return "⚠️ Modelo Gemini no inicializado."
        if df.empty: return "⚠️ No hay datos filtrados para responder la pregunta."

        # Preparar contexto rico
        num_productos = len(df)
        pct_descuento = (df['tiene_descuento'].astype(bool).sum() / num_productos * 100) if num_productos > 0 else 0
        contexto = f"""
DATOS DEL MERCADO (FILTRADOS):
- Total productos: {num_productos}
- Precio promedio: ${df['precio_actual'].mean():.2f}
- Precio mediano: ${df['precio_actual'].median():.2f}
- Tiendas: {df['tienda'].nunique()}
- % descuento: {pct_descuento:.1f}%
- Tipos de vino: {df['tipo_vino'].nunique() if 'tipo_vino' in df.columns else 'N/A'}
- Países: {df['pais_origen'].nunique() if 'pais_origen' in df.columns else 'N/A'}
"""

        # Contexto adicional según la pregunta
        if "competidor" in pregunta.lower() and 'tienda' in df.columns:
            competencia = df.groupby('tienda', observed=True).agg(
                 productos=('nombre', 'count'),
                 precio_prom=('precio_actual', 'mean')
            ).sort_values('productos', ascending=False)
            contexto += "\n\nTOP COMPETIDORES (en esta vista):\n"
            for tienda, row in competencia.head(3).iterrows():
                contexto += f"- {tienda}: {row['productos']} productos, ${row['precio_prom']:.2f} promedio\n"

        if ("catálogo" in pregunta.lower() or "agregar" in pregunta.lower()) and 'tipo_vino' in df.columns and 'pais_origen' in df.columns:
            combinaciones = df.groupby(['tipo_vino', 'pais_origen'], observed=True).size()
            nichos_pequenos = combinaciones[combinaciones < 5].head(5)
            if not nichos_pequenos.empty:
                contexto += "\n\nNICHOS CON POCA OFERTA (en esta vista):\n"
                for (tipo, pais), count in nichos_pequenos.items():
                    contexto += f"- {tipo} de {pais}: {count} productos\n"

        if ("precio" in pregunta.lower() or "margen" in pregunta.lower()) and 'segmento_precio' in df.columns:
            por_segmento = df.groupby('segmento_precio', observed=True)['precio_actual'].agg(['mean', 'count'])
            contexto += "\n\nPRECIOS POR SEGMENTO (en esta vista):\n"
            for seg, row in por_segmento.iterrows():
                contexto += f"- {seg}: ${row['mean']:.2f} ({row['count']} productos)\n"

        prompt = f"""
Responde esta pregunta de un emprendedor de vinos en México, basándote únicamente en los datos filtrados proporcionados:

**PREGUNTA:** {pregunta}

**CONTEXTO DEL MERCADO (FILTRADO):**
{contexto}

Proporciona una respuesta:
1. **Directa y específica** a la pregunta.
2. **Basada estrictamente en los datos** del contexto.
3. **Accionable** (si aplica, qué hacer).
4. **Con números concretos** del contexto si respaldan la respuesta.

Sé conciso (máximo 200 palabras). Si los datos no son suficientes para responder, indícalo claramente.
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠️ Error al responder pregunta IA: {str(e)}"


    # --- Métodos de ayuda ---
    def _preparar_contexto_mercado(self, df: pd.DataFrame, alertas: List[Dict]) -> str:
        # (Sin cambios respecto a la versión local)
        if df.empty: return "No hay datos filtrados."
        total_productos = len(df)
        # Calcular métricas solo si hay datos y columnas
        precio_promedio = df['precio_actual'].mean() if 'precio_actual' in df.columns and not df.empty else 0
        precio_mediano = df['precio_actual'].median() if 'precio_actual' in df.columns and not df.empty else 0
        tiendas = df['tienda'].nunique() if 'tienda' in df.columns and not df.empty else 0
        segmentos = df['segmento_precio'].value_counts().to_dict() if 'segmento_precio' in df.columns and not df.empty else {}
        tipos = df['tipo_vino'].value_counts().head(5).to_dict() if 'tipo_vino' in df.columns and not df.empty else {}

        return f"""
- Productos en vista: {total_productos:,} | Tiendas en vista: {tiendas}
- Precio Promedio: ${precio_promedio:,.2f} | Mediana: ${precio_mediano:,.2f} MXN
- Distribución por Segmento: {segmentos}
- Top 5 Tipos de Vino: {tipos}
- Alertas Activas (Globales): {len(alertas)}
"""

    def _formatear_alertas(self, alertas: List[Dict]) -> str:
        # (Sin cambios respecto a la versión local)
        if not alertas: return "No hay alertas activas."
        texto = ""
        # Usar .get() para evitar KeyErrors
        for i, alerta in enumerate(alertas[:10], 1): # Limitar a 10
            texto += f"{i}. [{alerta.get('nivel', 'INFO').upper()}] {alerta.get('mensaje', 'N/A')}: {alerta.get('detalle', 'N/A')}\n"
        return texto

    def _formatear_competidores(self, competidores: pd.DataFrame) -> str:
        # (Sin cambios respecto a la versión local)
        if competidores.empty: return "No se encontraron competidores directos con los filtros actuales."
        texto = ""
        for idx, comp in competidores.iterrows():
            # Usar .get() por si alguna columna falta inesperadamente
            texto += f"- {comp.get('nombre','N/A')} ({comp.get('tienda','N/A')}): ${comp.get('precio_actual', 0.0):.2f}\n"
        return texto

    def _obtener_top_productos(self, df: pd.DataFrame) -> str:
        # (Sin cambios respecto a la versión local)
        texto = ""
        if 'segmento_precio' not in df.columns or 'tipo_vino' not in df.columns:
             return " (Faltan columnas 'segmento_precio' o 'tipo_vino')"

        # Usar dropna() antes de value_counts
        segmentos_populares = df['segmento_precio'].dropna().value_counts().head(3).index
        for segmento in segmentos_populares:
            df_seg = df[df['segmento_precio'] == segmento]
            if not df_seg.empty:
                 # Usar dropna() también aquí
                 top = df_seg['tipo_vino'].dropna().value_counts().head(3)
                 if not top.empty:
                     texto += f"\nSegmento '{segmento}':\n"
                     for tipo, count in top.items():
                         texto += f"  - {tipo}: {count} productos\n"
        return texto if texto else " (No hay datos suficientes por segmento)"