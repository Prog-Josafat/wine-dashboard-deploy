# wine_scraper/utils/data_quality.py
import pandas as pd
import logging

logger = logging.getLogger("wine_scraper")

class DataQuality:
    """Gestiona la calidad y clasificación de datos sin eliminar registros"""
    
    def __init__(self):
        self.required_columns = [
            'tienda', 'nombre', 'precio_actual', 'precio_anterior', 
            'descuento_porcentaje', 'tamaño_botella', 'tipo_vino', 
            'pais_origen', 'region_origen', 'uva_varietal',
            'url_producto', 'fecha_scraping'
        ]
    
    def classify_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clasifica cada registro según su calidad de datos.
        NO elimina registros, solo los etiqueta.
        """
        df = df.copy()
        
        df['calidad_datos'] = 'completo'
        
        df.loc[df['precio_actual'].isna() | (df['precio_actual'] <= 0), 'calidad_datos'] = 'sin_precio'
        df.loc[df['nombre'].isna() | (df['nombre'].str.len() < 3), 'calidad_datos'] = 'sin_nombre'
        
        mask_parcial = (
            (df['precio_actual'].notna()) & 
            (df['precio_actual'] > 0) &
            (df['nombre'].notna()) &
            (
                df['tipo_vino'].isna() | 
                df['pais_origen'].isna() |
                df['tamaño_botella'].isna()
            )
        )
        df.loc[mask_parcial, 'calidad_datos'] = 'parcial'
        
        mask_completo = (
            (df['precio_actual'].notna()) & 
            (df['precio_actual'] > 0) &
            (df['nombre'].notna()) &
            (df['tipo_vino'].notna()) &
            (df['pais_origen'].notna())
        )
        df.loc[mask_completo, 'calidad_datos'] = 'completo'
        
        logger.info(f"\n📊 Clasificación de calidad:")
        logger.info(f"  ✅ Completo: {(df['calidad_datos'] == 'completo').sum()}")
        logger.info(f"  ⚠️  Parcial: {(df['calidad_datos'] == 'parcial').sum()}")
        logger.info(f"  ❌ Sin precio: {(df['calidad_datos'] == 'sin_precio').sum()}")
        logger.info(f"  ❌ Sin nombre: {(df['calidad_datos'] == 'sin_nombre').sum()}")
        
        return df
    
    def get_dataset_for_analysis(self, df: pd.DataFrame, tipo_analisis: str) -> pd.DataFrame:
        """
        Devuelve el subset de datos apropiado según el tipo de análisis.
        
        Args:
            tipo_analisis: 'precio', 'competencia', 'catalogo', 'geografia', 'todo'
        
        Returns:
            DataFrame filtrado
        """
        if tipo_analisis == 'precio':
            subset = df[
                (df['precio_actual'].notna()) & 
                (df['precio_actual'] > 0)
            ]
            logger.info(f"🔍 Dataset para análisis de precios: {len(subset)}/{len(df)} registros")
            
        elif tipo_analisis == 'competencia':
            subset = df[
                (df['precio_actual'].notna()) & 
                (df['precio_actual'] > 0) &
                (df['tienda'].notna()) &
                (df['nombre'].notna())
            ]
            logger.info(f"🔍 Dataset para análisis competitivo: {len(subset)}/{len(df)} registros")
            
        elif tipo_analisis == 'catalogo':
            subset = df[
                (df['tipo_vino'].notna()) & 
                (df['pais_origen'].notna())
            ]
            logger.info(f"🔍 Dataset para análisis de catálogo: {len(subset)}/{len(df)} registros")
            
        elif tipo_analisis == 'geografia':
            subset = df[df['pais_origen'].notna()]
            logger.info(f"🔍 Dataset para análisis geográfico: {len(subset)}/{len(df)} registros")
            
        else:
            subset = df
            logger.info(f"🔍 Dataset completo: {len(subset)} registros")
        
        return subset
    
    def generar_reporte_calidad(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera un reporte detallado de calidad de datos por tienda"""
        
        reporte_general = {
            'total_registros': len(df),
            'con_precio': df['precio_actual'].notna().sum(),
            'con_tipo': df['tipo_vino'].notna().sum(),
            'con_pais': df['pais_origen'].notna().sum(),
            'con_tamaño': df['tamaño_botella'].notna().sum(),
            'completos': df[
                df[['precio_actual', 'tipo_vino', 'pais_origen']].notna().all(axis=1)
            ].shape[0]
        }
        
        reporte_tiendas = []
        for tienda in df['tienda'].unique():
            df_t = df[df['tienda'] == tienda]
            
            reporte_tiendas.append({
                'tienda': tienda,
                'total': len(df_t),
                'con_precio': df_t['precio_actual'].notna().sum(),
                'con_tipo': df_t['tipo_vino'].notna().sum(),
                'con_pais': df_t['pais_origen'].notna().sum(),
                'completitud_%': round(
                    (df_t[['precio_actual', 'tipo_vino', 'pais_origen']].notna().all(axis=1).sum() / len(df_t)) * 100, 
                    2
                )
            })
        
        logger.info("\n" + "="*60)
        logger.info("📋 REPORTE DE CALIDAD DE DATOS")
        logger.info("="*60)
        logger.info(f"\n📊 Resumen General:")
        for key, value in reporte_general.items():
            logger.info(f"  {key}: {value}")
        
        logger.info(f"\n🏪 Calidad por Tienda:")
        df_reporte = pd.DataFrame(reporte_tiendas).sort_values('completitud_%', ascending=False)
        logger.info(f"\n{df_reporte.to_string(index=False)}")
        logger.info("="*60 + "\n")
        
        return df_reporte
    
    def fill_missing_with_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rellena valores faltantes con valores por defecto legibles.
        NO inventa datos, solo hace explícita la falta de información.
        """
        df = df.copy()
        
        df['tipo_vino'] = df['tipo_vino'].fillna('No especificado')
        df['pais_origen'] = df['pais_origen'].fillna('No especificado')
        df['region_origen'] = df['region_origen'].fillna('No especificado')
        df['uva_varietal'] = df['uva_varietal'].fillna('No especificado')
        df['tamaño_botella'] = df['tamaño_botella'].fillna('No especificado')
        
        logger.info("✓ Valores faltantes marcados como 'No especificado'")
        return df
    
    def calcular_confianza_analisis(self, df_original: pd.DataFrame, df_filtrado: pd.DataFrame) -> dict:
        """Calcula el nivel de confianza de un análisis basado en datos disponibles"""
        
        n_total = len(df_original)
        n_usado = len(df_filtrado)
        
        confianza = {
            'registros_totales': n_total,
            'registros_usados': n_usado,
            'porcentaje_cobertura': round((n_usado / n_total) * 100, 2) if n_total > 0 else 0,
            'nivel_confianza': ''
        }
        
        if confianza['porcentaje_cobertura'] >= 80:
            confianza['nivel_confianza'] = 'Alto'
        elif confianza['porcentaje_cobertura'] >= 60:
            confianza['nivel_confianza'] = 'Medio'
        else:
            confianza['nivel_confianza'] = 'Bajo'
        
        return confianza