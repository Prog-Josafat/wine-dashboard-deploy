# wine_scraper/utils/price_simulator.py
"""
Módulo de simulación de escenarios de precios
Permite simular cambios y calcular impactos
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class PriceSimulator:
    """Simulador de escenarios de precios y márgenes"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa el simulador
        
        Args:
            df: DataFrame con datos actuales del mercado
        """
        self.df = df.copy()
    
    def simulate_price_changes(
        self, 
        changes_by_segment: Dict[str, float],
        costo_operativo_pct: float = 60.0
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Simula cambios de precio por segmento
        
        Args:
            changes_by_segment: Dict {segmento: variacion_porcentaje}
                Ejemplo: {'Económico': -5, 'Premium': 10}
            costo_operativo_pct: Porcentaje del precio que representa el costo
        
        Returns:
            (DataFrame simulado, Dict con métricas de impacto)
        """
        df_sim = self.df.copy()
        
        for segmento, cambio_pct in changes_by_segment.items():
            mask = df_sim['segmento_precio'] == segmento
            df_sim.loc[mask, 'precio_simulado'] = (
                df_sim.loc[mask, 'precio_actual'] * (1 + cambio_pct / 100)
            )
        
        df_sim['precio_simulado'].fillna(df_sim['precio_actual'], inplace=True)
        
        precio_actual_prom = df_sim['precio_actual'].mean()
        precio_simulado_prom = df_sim['precio_simulado'].mean()

        costo_promedio_actual = precio_actual_prom * (costo_operativo_pct / 100)
        costo_promedio_simulado = precio_simulado_prom * (costo_operativo_pct / 100)
        
        margen_actual = ((precio_actual_prom - costo_promedio_actual) / precio_actual_prom) * 100
        margen_simulado = ((precio_simulado_prom - costo_promedio_simulado) / precio_simulado_prom) * 100
        
        impacto = {
            'precio_actual_promedio': precio_actual_prom,
            'precio_simulado_promedio': precio_simulado_prom,
            'variacion_precio_pct': ((precio_simulado_prom - precio_actual_prom) / precio_actual_prom) * 100,
            'variacion_precio_abs': precio_simulado_prom - precio_actual_prom,
            'margen_actual_pct': margen_actual,
            'margen_simulado_pct': margen_simulado,
            'variacion_margen_pp': margen_simulado - margen_actual,
            'productos_afectados': len(df_sim[df_sim['precio_simulado'] != df_sim['precio_actual']]),
            'total_productos': len(df_sim)
        }
        
        return df_sim, impacto
    
    def simulate_discount_strategy(
        self,
        target_discount_pct: float,
        segments_to_affect: list = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Simula una estrategia de descuentos
        
        Args:
            target_discount_pct: Porcentaje objetivo de productos con descuento
            segments_to_affect: Lista de segmentos a afectar (None = todos)
        
        Returns:
            (DataFrame simulado, Dict con métricas)
        """
        df_sim = self.df.copy()
        
        if segments_to_affect:
            mask = df_sim['segmento_precio'].isin(segments_to_affect)
            df_to_modify = df_sim[mask]
        else:
            df_to_modify = df_sim

        current_discount_pct = (df_to_modify['tiene_descuento'].sum() / len(df_to_modify)) * 100
        target_count = int(len(df_to_modify) * (target_discount_pct / 100))
        current_count = df_to_modify['tiene_descuento'].sum()
        
        change_needed = target_count - current_count
        
        if change_needed > 0:
            productos_sin_descuento = df_to_modify[~df_to_modify['tiene_descuento']]
            if len(productos_sin_descuento) >= change_needed:
                indices_to_discount = productos_sin_descuento.sample(change_needed).index
                df_sim.loc[indices_to_discount, 'tiene_descuento_simulado'] = True
                df_sim.loc[indices_to_discount, 'precio_simulado'] = (
                    df_sim.loc[indices_to_discount, 'precio_actual'] * 0.85
                )
        elif change_needed < 0:
            productos_con_descuento = df_to_modify[df_to_modify['tiene_descuento']]
            if len(productos_con_descuento) >= abs(change_needed):
                indices_to_remove = productos_con_descuento.sample(abs(change_needed)).index
                df_sim.loc[indices_to_remove, 'tiene_descuento_simulado'] = False
                df_sim.loc[indices_to_remove, 'precio_simulado'] = (
                    df_sim.loc[indices_to_remove, 'precio_actual']
                )
        
        df_sim['tiene_descuento_simulado'].fillna(df_sim['tiene_descuento'], inplace=True)
        df_sim['precio_simulado'].fillna(df_sim['precio_actual'], inplace=True)
        
        new_discount_pct = (df_sim['tiene_descuento_simulado'].sum() / len(df_sim)) * 100
        impacto_precio = df_sim['precio_simulado'].mean() - df_sim['precio_actual'].mean()
        
        impacto = {
            'descuento_actual_pct': (df_sim['tiene_descuento'].sum() / len(df_sim)) * 100,
            'descuento_simulado_pct': new_discount_pct,
            'productos_con_descuento_actual': df_sim['tiene_descuento'].sum(),
            'productos_con_descuento_simulado': df_sim['tiene_descuento_simulado'].sum(),
            'impacto_precio_promedio': impacto_precio,
            'productos_modificados': abs(change_needed) if abs(change_needed) <= len(df_to_modify) else 0
        }
        
        return df_sim, impacto
    
    def calculate_breakeven(
        self,
        costos_fijos_mes: float,
        costo_variable_unitario: float,
        precio_venta_promedio: Optional[float] = None
    ) -> Dict:
        """
        Calcula punto de equilibrio
        
        Args:
            costos_fijos_mes: Costos fijos mensuales
            costo_variable_unitario: Costo variable por botella
            precio_venta_promedio: Precio de venta (None = usar promedio del df)
        
        Returns:
            Dict con análisis de punto de equilibrio
        """
        if precio_venta_promedio is None:
            precio_venta_promedio = self.df['precio_actual'].mean()
        
        margen_contribucion = precio_venta_promedio - costo_variable_unitario
        
        if margen_contribucion <= 0:
            return {
                'error': 'Margen de contribución negativo o cero',
                'margen_contribucion': margen_contribucion
            }
        
        unidades_equilibrio = costos_fijos_mes / margen_contribucion
        
        return {
            'precio_venta_promedio': precio_venta_promedio,
            'costo_variable_unitario': costo_variable_unitario,
            'margen_contribucion': margen_contribucion,
            'margen_contribucion_pct': (margen_contribucion / precio_venta_promedio) * 100,
            'costos_fijos_mes': costos_fijos_mes,
            'unidades_equilibrio_mes': unidades_equilibrio,
            'unidades_equilibrio_dia': unidades_equilibrio / 30,
            'ingresos_equilibrio': unidades_equilibrio * precio_venta_promedio
        }
    
    def simulate_mix_change(
        self,
        target_mix: Dict[str, float],
        total_inventory: int = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Simula cambio en el mix de productos por segmento
        
        Args:
            target_mix: Dict {segmento: porcentaje_deseado}
                Ejemplo: {'Económico': 30, 'Medio': 50, 'Premium': 20}
            total_inventory: Total de productos (None = usar actual)
        
        Returns:
            (DataFrame simulado, Dict con métricas)
        """
        if total_inventory is None:
            total_inventory = len(self.df)
        
        current_mix = (self.df['segmento_precio'].value_counts(normalize=True) * 100).to_dict()
        
        target_counts = {}
        for segmento, pct in target_mix.items():
            target_counts[segmento] = int(total_inventory * (pct / 100))
        
        changes = {}
        for segmento in target_mix.keys():
            current_count = len(self.df[self.df['segmento_precio'] == segmento])
            target_count = target_counts.get(segmento, 0)
            changes[segmento] = target_count - current_count
        
        df_actual = self.df.copy()
        precio_promedio_por_segmento = df_actual.groupby('segmento_precio')['precio_actual'].mean()
        
        precio_actual_ponderado = sum(
            precio_promedio_por_segmento.get(seg, 0) * (current_mix.get(seg, 0) / 100)
            for seg in precio_promedio_por_segmento.index
        )
        
        precio_simulado_ponderado = sum(
            precio_promedio_por_segmento.get(seg, 0) * (target_mix.get(seg, 0) / 100)
            for seg in target_mix.keys()
        )
        
        impacto = {
            'mix_actual': current_mix,
            'mix_objetivo': target_mix,
            'cambios_por_segmento': changes,
            'precio_actual_ponderado': precio_actual_ponderado,
            'precio_simulado_ponderado': precio_simulado_ponderado,
            'variacion_precio_pct': ((precio_simulado_ponderado - precio_actual_ponderado) / precio_actual_ponderado) * 100,
            'total_inventario': total_inventory
        }
        
        return df_actual, impacto
    
    def estimate_demand_elasticity(
        self,
        price_change_pct: float,
        segment: str = None
    ) -> Dict:
        """
        Estima elasticidad de demanda básica
        
        Args:
            price_change_pct: Cambio porcentual en precio
            segment: Segmento específico (None = global)
        
        Returns:
            Dict con estimación de elasticidad
        """
        elasticities = {
            'Económico': -1.5,
            'Medio-Bajo': -1.2,
            'Medio': -0.8,
            'Medio-Alto': -0.5,
            'Premium': -0.3
        }
        
        if segment and segment in elasticities:
            elasticity = elasticities[segment]
        else:
            segment_counts = self.df['segmento_precio'].value_counts(normalize=True)
            elasticity = sum(
                elasticities.get(seg, -0.8) * segment_counts.get(seg, 0)
                for seg in elasticities.keys()
            )
        
        demand_change_pct = elasticity * price_change_pct
        
        current_units = len(self.df[self.df['segmento_precio'] == segment]) if segment else len(self.df)
        projected_units = current_units * (1 + demand_change_pct / 100)
        
        return {
            'segmento': segment if segment else 'Global',
            'elasticidad': elasticity,
            'cambio_precio_pct': price_change_pct,
            'cambio_demanda_esperado_pct': demand_change_pct,
            'unidades_actuales': current_units,
            'unidades_proyectadas': int(projected_units),
            'interpretacion': self._interpret_elasticity(elasticity)
        }
    
    def _interpret_elasticity(self, elasticity: float) -> str:
        """Interpreta el valor de elasticidad"""
        if elasticity < -1.5:
            return "MUY ELÁSTICO: Los clientes son muy sensibles al precio"
        elif elasticity < -1.0:
            return "ELÁSTICO: Los clientes reaccionan significativamente a cambios de precio"
        elif elasticity < -0.5:
            return "MODERADO: Sensibilidad media al precio"
        else:
            return "INELÁSTICO: Los clientes son poco sensibles al precio"