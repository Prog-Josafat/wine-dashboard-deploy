# wine_scraper/utils/predictor.py
"""
Módulo de predicciones básicas usando datos históricos
Predicciones simples pero efectivas sin ML complejo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class SimplePredictor:
    """Predictor simple basado en tendencias históricas"""
    
    def __init__(self, historical_analyzer):
        """
        Inicializa el predictor
        
        Args:
            historical_analyzer: Instancia de HistoricalAnalyzer con datos cargados
        """
        self.historical = historical_analyzer
        self.snapshots = historical_analyzer.snapshots
    
    def predict_next_period_prices(self, segment: str = None) -> Dict:
        """
        Predice precios del próximo período basado en tendencia
        
        Args:
            segment: Segmento específico o None para global
        
        Returns:
            Dict con predicción y métricas de confianza
        """
        if len(self.snapshots) < 3:
            return {'error': 'Se necesitan al menos 3 snapshots para predicción'}
        
        filter_dict = {'segmento_precio': segment} if segment else None
        evolution = self.historical.get_price_evolution(filter_dict)
        
        if len(evolution) < 3:
            return {'error': 'Datos insuficientes para este segmento'}
        
        precios = evolution['precio_promedio'].values
        x = np.arange(len(precios))
        
        coeffs = np.polyfit(x, precios, 1)
        tendencia = coeffs[0]
        
        prediccion = precios[-1] + tendencia

        std_cambios = evolution['variacion_precio'].std() if 'variacion_precio' in evolution.columns else 0
        margen_error = std_cambios * 1.96 
        
        dates = sorted(self.snapshots.keys())
        if len(dates) >= 2:
            dias_entre_capturas = (
                datetime.strptime(dates[-1], '%Y%m%d') - 
                datetime.strptime(dates[-2], '%Y%m%d')
            ).days
            fecha_prediccion = datetime.strptime(dates[-1], '%Y%m%d') + timedelta(days=dias_entre_capturas)
        else:
            dias_entre_capturas = 14
            fecha_prediccion = datetime.now() + timedelta(days=14)
        
        return {
            'segmento': segment if segment else 'Global',
            'precio_actual': precios[-1],
            'precio_predicho': prediccion,
            'variacion_esperada': prediccion - precios[-1],
            'variacion_pct': ((prediccion - precios[-1]) / precios[-1]) * 100,
            'margen_error': margen_error,
            'rango_prediccion': (prediccion - margen_error, prediccion + margen_error),
            'tendencia': 'alcista' if tendencia > 0 else 'bajista',
            'confianza': self._calculate_confidence(evolution),
            'fecha_prediccion': fecha_prediccion.strftime('%Y-%m-%d'),
            'dias_hasta_prediccion': dias_entre_capturas,
            'basado_en_snapshots': len(precios)
        }
    
    def _calculate_confidence(self, evolution: pd.DataFrame) -> str:
        """Calcula nivel de confianza de la predicción"""
        if len(evolution) < 3:
            return 'baja'
        elif len(evolution) < 5:
            return 'media'
        else:
            if 'variacion_precio' in evolution.columns:
                cambios = evolution['variacion_precio'].dropna()
                if len(cambios) > 0:
                    misma_direccion = (cambios > 0).sum() / len(cambios)
                    if misma_direccion > 0.7 or misma_direccion < 0.3:
                        return 'alta'
            return 'media'
    
    def predict_rotation_trend(self) -> Dict:
        """
        Predice tendencia de rotación de productos
        
        Returns:
            Dict con predicción de rotación
        """
        if len(self.snapshots) < 3:
            return {'error': 'Se necesitan al menos 3 snapshots'}
        
        dates = sorted(self.snapshots.keys())
        rotaciones = []
        
        for i in range(len(dates) - 1):
            df_ant = self.snapshots[dates[i]]
            df_act = self.snapshots[dates[i + 1]]
            
            df_ant['product_id'] = df_ant['nombre'] + '|' + df_ant['tienda']
            df_act['product_id'] = df_act['nombre'] + '|' + df_act['tienda']
            
            set_ant = set(df_ant['product_id'])
            set_act = set(df_act['product_id'])
            
            descontinuados = len(set_ant - set_act)
            tasa = (descontinuados / len(set_ant)) * 100 if len(set_ant) > 0 else 0
            
            rotaciones.append(tasa)
        
        tasa_promedio = np.mean(rotaciones)
        tasa_reciente = rotaciones[-1]
        
        if tasa_reciente > tasa_promedio * 1.2:
            tendencia = 'acelerándose'
        elif tasa_reciente < tasa_promedio * 0.8:
            tendencia = 'desacelerándose'
        else:
            tendencia = 'estable'
        
        return {
            'tasa_rotacion_promedio': round(tasa_promedio, 2),
            'tasa_rotacion_reciente': round(tasa_reciente, 2),
            'tendencia': tendencia,
            'periodos_analizados': len(rotaciones),
            'prediccion_proximo_periodo': round(tasa_reciente, 2),
            'interpretacion': self._interpret_rotation(tasa_reciente)
        }
    
    def _interpret_rotation(self, tasa: float) -> str:
        """Interpreta la tasa de rotación"""
        if tasa < 5:
            return "Muy baja - Catálogo muy estable"
        elif tasa < 10:
            return "Baja - Rotación saludable"
        elif tasa < 15:
            return "Normal - Renovación activa"
        elif tasa < 25:
            return "Alta - Cambios significativos"
        else:
            return "Muy alta - Posible volatilidad"
    
    def forecast_seasonal_patterns(self) -> Dict:
        """
        Detecta y proyecta patrones estacionales básicos
        
        Returns:
            Dict con patrones estacionales detectados
        """
        if len(self.snapshots) < 4:
            return {'error': 'Se necesitan al menos 4 snapshots para detectar estacionalidad'}
        
        monthly_data = {}
        
        for date, df in self.snapshots.items():
            mes = int(date[4:6])
            
            if mes not in monthly_data:
                monthly_data[mes] = {
                    'precios': [],
                    'descuentos': [],
                    'productos': []
                }
            
            monthly_data[mes]['precios'].append(df['precio_actual'].mean())
            monthly_data[mes]['descuentos'].append((df['tiene_descuento'].sum() / len(df)) * 100)
            monthly_data[mes]['productos'].append(len(df))
        
        monthly_avg = {}
        for mes, data in monthly_data.items():
            monthly_avg[mes] = {
                'precio_promedio': np.mean(data['precios']),
                'pct_descuento': np.mean(data['descuentos']),
                'num_productos': np.mean(data['productos'])
            }
        
        if monthly_avg:
            precios_por_mes = {m: d['precio_promedio'] for m, d in monthly_avg.items()}
            descuentos_por_mes = {m: d['pct_descuento'] for m, d in monthly_avg.items()}
            
            mes_precio_mas_alto = max(precios_por_mes, key=precios_por_mes.get)
            mes_precio_mas_bajo = min(precios_por_mes, key=precios_por_mes.get)
            mes_mas_descuentos = max(descuentos_por_mes, key=descuentos_por_mes.get)
            
            meses_nombres = {
                1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
            }
            
            return {
                'meses_analizados': len(monthly_avg),
                'mes_precios_altos': meses_nombres.get(mes_precio_mas_alto, 'N/A'),
                'mes_precios_bajos': meses_nombres.get(mes_precio_mas_bajo, 'N/A'),
                'mes_mas_descuentos': meses_nombres.get(mes_mas_descuentos, 'N/A'),
                'variacion_estacional': ((precios_por_mes[mes_precio_mas_alto] - precios_por_mes[mes_precio_mas_bajo]) / precios_por_mes[mes_precio_mas_bajo]) * 100,
                'datos_mensuales': monthly_avg
            }
        
        return {'error': 'No se pudieron calcular patrones estacionales'}
    
    def predict_demand_by_segment(self, price_change: Dict[str, float]) -> Dict:
        """
        Predice cambio en demanda por segmento dado un cambio de precios
        
        Args:
            price_change: Dict {segmento: cambio_porcentual}
        
        Returns:
            Dict con predicciones por segmento
        """
        elasticities = {
            'Económico': -1.5,
            'Medio-Bajo': -1.2,
            'Medio': -0.8,
            'Medio-Alto': -0.5,
            'Premium': -0.3
        }
        
        predictions = {}
        
        dates = sorted(self.snapshots.keys())
        df_latest = self.snapshots[dates[-1]]
        
        for segmento, cambio_precio in price_change.items():
            if segmento not in elasticities:
                continue
            
            elasticidad = elasticities[segmento]
            cambio_demanda = elasticidad * cambio_precio
            
            productos_segmento = len(df_latest[df_latest['segmento_precio'] == segmento])
            productos_proyectados = int(productos_segmento * (1 + cambio_demanda / 100))
            
            predictions[segmento] = {
                'cambio_precio_pct': cambio_precio,
                'elasticidad': elasticidad,
                'cambio_demanda_esperado_pct': cambio_demanda,
                'productos_actuales': productos_segmento,
                'productos_proyectados': productos_proyectados,
                'cambio_unidades': productos_proyectados - productos_segmento
            }
        
        return predictions
    
    def generate_forecast_summary(self) -> str:
        """
        Genera resumen de todas las predicciones en formato texto
        
        Returns:
            str: Resumen completo de predicciones
        """
        summary = "# 🔮 Resumen de Predicciones\n\n"
        
        pred_global = self.predict_next_period_prices()
        if 'error' not in pred_global:
            summary += f"## Predicción de Precios (Global)\n\n"
            summary += f"- **Precio Actual:** ${pred_global['precio_actual']:.2f}\n"
            summary += f"- **Precio Predicho:** ${pred_global['precio_predicho']:.2f}\n"
            summary += f"- **Variación Esperada:** {pred_global['variacion_pct']:+.2f}%\n"
            summary += f"- **Tendencia:** {pred_global['tendencia'].title()}\n"
            summary += f"- **Confianza:** {pred_global['confianza'].title()}\n"
            summary += f"- **Fecha Predicción:** {pred_global['fecha_prediccion']}\n\n"
        
        pred_rot = self.predict_rotation_trend()
        if 'error' not in pred_rot:
            summary += f"## Predicción de Rotación\n\n"
            summary += f"- **Tasa Promedio:** {pred_rot['tasa_rotacion_promedio']:.2f}%\n"
            summary += f"- **Tasa Reciente:** {pred_rot['tasa_rotacion_reciente']:.2f}%\n"
            summary += f"- **Tendencia:** {pred_rot['tendencia'].title()}\n"
            summary += f"- **Interpretación:** {pred_rot['interpretacion']}\n\n"
        
        pred_season = self.forecast_seasonal_patterns()
        if 'error' not in pred_season:
            summary += f"## Patrones Estacionales\n\n"
            summary += f"- **Mes Precios Altos:** {pred_season['mes_precios_altos']}\n"
            summary += f"- **Mes Precios Bajos:** {pred_season['mes_precios_bajos']}\n"
            summary += f"- **Mes Más Descuentos:** {pred_season['mes_mas_descuentos']}\n"
            summary += f"- **Variación Estacional:** {pred_season['variacion_estacional']:.1f}%\n\n"
        
        return summary