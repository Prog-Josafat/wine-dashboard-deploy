# wine_scraper/utils/gemini_analyzer.py
import os
import google.generativeai as genai
from typing import Dict, List, Any
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

env_loaded = False

class GeminiAnalyzer:
    """Analizador de datos de vino usando Gemini AI"""
    
    def __init__(self):
        """Inicializa el analizador con la API key desde variables de entorno"""
        global env_loaded
        
        if not env_loaded:
            dotenv_path = Path(__file__).resolve().parents[2] / '.env'
            if dotenv_path.exists():
                load_dotenv(dotenv_path=dotenv_path)
                env_loaded = True

        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("GEMINI_API_KEY")
            except ImportError:
                 pass
            except Exception:
                 pass

        if not api_key:
            raise ValueError(
                "❌ No se encontró GEMINI_API_KEY. "
                "Asegúrate de que esté configurada en tu archivo .env "
                "(ubicado en la raíz del proyecto, ej: 'wine-scraper-project/.env')"
            )
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
            print("✅ Gemini Analyzer inicializado correctamente.")
        except Exception as e:
            print(f"❌ Error al configurar la API de Gemini: {e}")
            raise ValueError("No se pudo configurar la API de Gemini. Verifica tu API Key.")

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
        producto_df = df[df['nombre'].str.contains(producto_nombre, case=False, na=False)]
        if producto_df.empty: return f"❌ No se encontró '{producto_nombre}'."
        producto = producto_df.iloc[0]
        competidores = df[(df['tipo_vino'] == producto['tipo_vino']) & (df['pais_origen'] == producto['pais_origen']) & (df['nombre'] != producto['nombre'])].head(5)
        
        prompt = f"""
Analiza este producto de vino y su posicionamiento competitivo:

PRODUCTO: {producto['nombre']} (${producto['precio_actual']:.2f}, {producto['tipo_vino']}, {producto['pais_origen']}, Segmento: {producto['segmento_precio']})
Tienda: {producto['tienda']}
Descuento: {'Sí' if producto['tiene_descuento'] else 'No'}

COMPETIDORES DIRECTOS (mismo tipo y origen):
{self._formatear_competidores(competidores)}

PRECIOS DEL SEGMENTO '{producto['segmento_precio']}':
Promedio: ${df[df['segmento_precio'] == producto['segmento_precio']]['precio_actual'].mean():.2f}
Mediana: ${df[df['segmento_precio'] == producto['segmento_precio']]['precio_actual'].median():.2f}

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
        analisis_segmentos = df.groupby('segmento_precio').agg(
            precio_prom=('precio_actual', 'mean'),
            precio_med=('precio_actual', 'median'),
            num_prod=('nombre', 'count'),
            pct_desc=('tiene_descuento', lambda x: (x.sum() / len(x)) * 100 if len(x) > 0 else 0)
        ).round(2)

        prompt = f"""
Eres un comprador experto de vinos para una nueva tienda online en México con presupuesto inicial de ${presupuesto:,.2f} MXN.

ANÁLISIS DE SEGMENTOS DEL MERCADO ACTUAL:
{analisis_segmentos.to_string()}

PRODUCTOS POPULARES POR SEGMENTO:
{self._obtener_top_productos(df)}

Diseña una estrategia de catálogo inicial (15-20 tipos de producto, no marcas):
1. Prioriza segmentos con buen balance precio/volumen.
2. Balancea vinos populares con algunos nichos interesantes.
3. Asigna un % del presupuesto a cada segmento principal.
4. Justifica brevemente la selección.
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e: return f"⚠️ Error al generar recomendación: {str(e)}"

    def _preparar_contexto_mercado(self, df: pd.DataFrame, alertas: List[Dict]) -> str:
        total_productos = len(df)
        precio_promedio = df['precio_actual'].mean()
        precio_mediano = df['precio_actual'].median()
        tiendas = df['tienda'].nunique()
        segmentos = df['segmento_precio'].value_counts().to_dict()
        tipos = df['tipo_vino'].value_counts().head(5).to_dict()
        return f"""
- Productos totales: {total_productos:,} | Tiendas activas: {tiendas}
- Precio Promedio: ${precio_promedio:,.2f} | Mediana: ${precio_mediano:,.2f} MXN
- Distribución por Segmento: {segmentos}
- Top 5 Tipos de Vino: {tipos}
- Alertas Activas: {len(alertas)}
"""

    def _formatear_alertas(self, alertas: List[Dict]) -> str:
        if not alertas: return "No hay alertas activas."
        texto = ""
        for i, alerta in enumerate(alertas[:10], 1):
            texto += f"{i}. [{alerta.get('nivel', 'INFO').upper()}] {alerta.get('mensaje', 'N/A')}: {alerta.get('detalle', 'N/A')}\n"
        return texto

    def _formatear_competidores(self, competidores: pd.DataFrame) -> str:
        if competidores.empty: return "No se encontraron competidores directos."
        texto = ""
        for idx, comp in competidores.iterrows():
            texto += f"- {comp['nombre']} ({comp['tienda']}): ${comp['precio_actual']:.2f}\n"
        return texto

    def _obtener_top_productos(self, df: pd.DataFrame) -> str:
        texto = ""
        segmentos_populares = df['segmento_precio'].value_counts().head(3).index
        for segmento in segmentos_populares:
            df_seg = df[df['segmento_precio'] == segmento]
            if not df_seg.empty:
                 top = df_seg['tipo_vino'].value_counts().head(3)
                 texto += f"\nSegmento '{segmento}':\n"
                 for tipo, count in top.items():
                     texto += f"  - {tipo}: {count} productos\n"
        return texto