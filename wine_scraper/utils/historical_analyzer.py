# wine_scraper/utils/historical_analyzer.py
"""
Módulo para análisis histórico de datos consolidados
Analiza tendencias, rotación y patrones temporales
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class HistoricalAnalyzer:
    """Analizador de datos históricos de vinos"""
    
    def __init__(self, base_path: str = './data/consolidated'):
        """
        Inicializa el analizador histórico
        
        Args:
            base_path: Ruta a la carpeta de datos consolidados
        """
        self.base_path = Path(base_path)
        self.snapshots = self._load_snapshots()
    
    def _load_snapshots(self) -> Dict[str, pd.DataFrame]:
        """
        Carga todos los snapshots disponibles
        
        Returns:
            Dict con {fecha: DataFrame}
        """
        snapshots = {}
        
        if not self.base_path.exists():
            return snapshots
        
        date_dirs = sorted([
            d for d in self.base_path.iterdir() 
            if d.is_dir() and d.name.isdigit()
        ])
        
        for date_dir in date_dirs:
            csv_file = date_dir / 'datos_completos_listos.csv'
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    snapshots[date_dir.name] = df
                except Exception as e:
                    print(f"⚠️ Error cargando {date_dir.name}: {e}")
        
        return snapshots
    
    def get_available_dates(self) -> List[str]:
        """Retorna lista de fechas disponibles ordenadas"""
        return sorted(self.snapshots.keys(), reverse=True)
    
    def get_price_evolution(self, product_filter: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calcula evolución de precios a lo largo del tiempo
        
        Args:
            product_filter: Dict con filtros {'tipo_vino': 'Tinto', 'pais_origen': 'Chile'}
        
        Returns:
            DataFrame con columnas: fecha, precio_promedio, precio_mediano, etc.
        """
        if len(self.snapshots) < 2:
            return pd.DataFrame()
        
        evolution = []
        
        for fecha, df in self.snapshots.items():
            df_filtered = df.copy()
            if product_filter:
                for col, val in product_filter.items():
                    if col in df.columns:
                        df_filtered = df_filtered[df_filtered[col] == val]
            
            if len(df_filtered) == 0:
                continue
            
            evolution.append({
                'fecha': fecha,
                'fecha_dt': datetime.strptime(fecha, '%Y%m%d'),
                'precio_promedio': df_filtered['precio_actual'].mean(),
                'precio_mediano': df_filtered['precio_actual'].median(),
                'precio_min': df_filtered['precio_actual'].min(),
                'precio_max': df_filtered['precio_actual'].max(),
                'num_productos': len(df_filtered),
                'pct_descuento': (df_filtered['tiene_descuento'].sum() / len(df_filtered)) * 100
            })
        
        df_evolution = pd.DataFrame(evolution)
        
        if len(df_evolution) > 1:
            df_evolution['variacion_precio'] = df_evolution['precio_promedio'].pct_change() * 100
            df_evolution['variacion_productos'] = df_evolution['num_productos'].pct_change() * 100
        
        return df_evolution
    
    def calculate_rotation_rate(self) -> Dict:
        """
        Calcula tasa de rotación de productos entre snapshots
        
        Returns:
            Dict con métricas de rotación
        """
        if len(self.snapshots) < 2:
            return {'error': 'Se necesitan al menos 2 snapshots'}
        
        dates = sorted(self.snapshots.keys())
        
        df_anterior = self.snapshots[dates[-2]]
        df_actual = self.snapshots[dates[-1]]
        
        df_anterior['product_id'] = df_anterior['nombre'] + '|' + df_anterior['tienda']
        df_actual['product_id'] = df_actual['nombre'] + '|' + df_actual['tienda']
        
        set_anterior = set(df_anterior['product_id'])
        set_actual = set(df_actual['product_id'])
        
        productos_nuevos = set_actual - set_anterior
        productos_descontinuados = set_anterior - set_actual
        productos_permanentes = set_actual & set_anterior
        
        tasa_rotacion = (len(productos_descontinuados) / len(set_anterior)) * 100
        tasa_incorporacion = (len(productos_nuevos) / len(set_actual)) * 100
        
        return {
            'fecha_anterior': dates[-2],
            'fecha_actual': dates[-1],
            'productos_anterior': len(set_anterior),
            'productos_actual': len(set_actual),
            'productos_nuevos': len(productos_nuevos),
            'productos_descontinuados': len(productos_descontinuados),
            'productos_permanentes': len(productos_permanentes),
            'tasa_rotacion_pct': round(tasa_rotacion, 2),
            'tasa_incorporacion_pct': round(tasa_incorporacion, 2),
            'cambio_neto': len(set_actual) - len(set_anterior)
        }
    
    def get_rotation_by_store(self) -> pd.DataFrame:
        """
        Calcula rotación desglosada por tienda
        
        Returns:
            DataFrame con rotación por tienda
        """
        if len(self.snapshots) < 2:
            return pd.DataFrame()
        
        dates = sorted(self.snapshots.keys())
        df_anterior = self.snapshots[dates[-2]]
        df_actual = self.snapshots[dates[-1]]
        
        df_anterior['product_id'] = df_anterior['nombre'] + '|' + df_anterior['tienda']
        df_actual['product_id'] = df_actual['nombre'] + '|' + df_actual['tienda']
        
        rotation_data = []
        
        for tienda in df_actual['tienda'].unique():
            ant = set(df_anterior[df_anterior['tienda'] == tienda]['product_id'])
            act = set(df_actual[df_actual['tienda'] == tienda]['product_id'])
            
            if len(ant) == 0:
                continue
            
            nuevos = len(act - ant)
            descontinuados = len(ant - act)
            tasa = (descontinuados / len(ant)) * 100
            
            rotation_data.append({
                'tienda': tienda,
                'productos_anteriores': len(ant),
                'productos_actuales': len(act),
                'nuevos': nuevos,
                'descontinuados': descontinuados,
                'tasa_rotacion': round(tasa, 2)
            })
        
        return pd.DataFrame(rotation_data).sort_values('tasa_rotacion', ascending=False)
    
    def detect_price_trends(self, min_snapshots: int = 3) -> Dict[str, str]:
        """
        Detecta tendencias de precio por categoría
        
        Args:
            min_snapshots: Mínimo de snapshots necesarios
        
        Returns:
            Dict con tendencias por tipo de vino, país, etc.
        """
        if len(self.snapshots) < min_snapshots:
            return {'error': f'Se necesitan al menos {min_snapshots} snapshots'}
        
        trends = {}
        
        tipo_trends = {}
        for tipo in ['Tinto', 'Blanco', 'Rosado', 'Espumoso']:
            evolution = self.get_price_evolution({'tipo_vino': tipo})
            if len(evolution) >= min_snapshots:
                variacion_total = (
                    (evolution['precio_promedio'].iloc[-1] - evolution['precio_promedio'].iloc[0]) /
                    evolution['precio_promedio'].iloc[0] * 100
                )
                
                if variacion_total > 5:
                    tendencia = '📈 Alza'
                elif variacion_total < -5:
                    tendencia = '📉 Baja'
                else:
                    tendencia = '➡️ Estable'
                
                tipo_trends[tipo] = {
                    'tendencia': tendencia,
                    'variacion_pct': round(variacion_total, 2)
                }
        
        trends['por_tipo'] = tipo_trends
        
        dates = sorted(self.snapshots.keys())
        df_latest = self.snapshots[dates[-1]]
        top_paises = df_latest['pais_origen'].value_counts().head(5).index
        
        pais_trends = {}
        for pais in top_paises:
            evolution = self.get_price_evolution({'pais_origen': pais})
            if len(evolution) >= min_snapshots:
                variacion_total = (
                    (evolution['precio_promedio'].iloc[-1] - evolution['precio_promedio'].iloc[0]) /
                    evolution['precio_promedio'].iloc[0] * 100
                )
                
                if variacion_total > 5:
                    tendencia = '📈 Alza'
                elif variacion_total < -5:
                    tendencia = '📉 Baja'
                else:
                    tendencia = '➡️ Estable'
                
                pais_trends[pais] = {
                    'tendencia': tendencia,
                    'variacion_pct': round(variacion_total, 2)
                }
        
        trends['por_pais'] = pais_trends
        
        return trends
    
    def get_seasonality_insights(self) -> Dict:
        """
        Detecta patrones estacionales básicos
        
        Returns:
            Dict con insights de estacionalidad
        """
        if len(self.snapshots) < 3:
            return {'error': 'Se necesitan al menos 3 snapshots'}
        
        evolution = self.get_price_evolution()
        
        if len(evolution) == 0:
            return {'error': 'No hay datos de evolución'}
        
        evolution['mes'] = evolution['fecha_dt'].dt.month
        
        actividad_por_mes = evolution.groupby('mes').agg({
            'num_productos': 'mean',
            'pct_descuento': 'mean'
        }).round(2)
        
        mes_mas_productos = actividad_por_mes['num_productos'].idxmax()
        mes_mas_descuentos = actividad_por_mes['pct_descuento'].idxmax()
        
        meses_nombres = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        
        return {
            'mes_mayor_catalogo': meses_nombres.get(mes_mas_productos, 'N/A'),
            'mes_mas_descuentos': meses_nombres.get(mes_mas_descuentos, 'N/A'),
            'variacion_promedio_mensual': evolution['variacion_precio'].mean(),
            'datos_disponibles': len(evolution)
        }
    
    def get_summary_statistics(self) -> Dict:
        """
        Genera estadísticas resumen de todos los snapshots
        
        Returns:
            Dict con estadísticas clave
        """
        if not self.snapshots:
            return {'error': 'No hay snapshots disponibles'}
        
        dates = sorted(self.snapshots.keys())
        
        return {
            'num_snapshots': len(self.snapshots),
            'fecha_mas_antigua': dates[0],
            'fecha_mas_reciente': dates[-1],
            'dias_entre_snapshots': (
                datetime.strptime(dates[-1], '%Y%m%d') - 
                datetime.strptime(dates[0], '%Y%m%d')
            ).days,
            'productos_promedio': int(np.mean([len(df) for df in self.snapshots.values()])),
            'precio_promedio_historico': round(
                np.mean([df['precio_actual'].mean() for df in self.snapshots.values()]), 2
            )
        }