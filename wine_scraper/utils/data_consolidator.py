# wine_scraper/utils/data_consolidator.py
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger("wine_scraper")

class DataConsolidator:
    """Unifica múltiples archivos CSV en un solo dataset consolidado"""
    def __init__(self, data_dir='./data', create_new_dir=True):
        self.data_dir = Path(data_dir)
        self.consolidated_base = self.data_dir / 'consolidated'
        self.consolidated_base.mkdir(exist_ok=True)
        
        if create_new_dir:
            # Comportamiento original: crear carpeta con fecha actual (para el scraper)
            timestamp = datetime.now().strftime("%Y%m%d")
            self.output_dir = self.consolidated_base / timestamp
            self.output_dir.mkdir(exist_ok=True)
            logger.info(f"Usando (y creando si es necesario) el directorio de salida: {self.output_dir}")
        else:
            # Nuevo comportamiento: buscar la última carpeta existente (para el dashboard)
            latest_dir = self.get_latest_consolidated_dir()
            if latest_dir:
                self.output_dir = latest_dir
            else:
                # Si no hay ninguna carpeta, crea una para hoy como respaldo
                logger.warning("No se encontraron directorios existentes. Creando uno para hoy.")
                timestamp = datetime.now().strftime("%Y%m%d")
                self.output_dir = self.consolidated_base / timestamp
                self.output_dir.mkdir(exist_ok=True)

    def consolidate_all_scrapers(self, use_latest=True):
        """
        Consolida datos de todas las tiendas en un solo archivo.
        
        Args:
            use_latest: Si True, usa solo el archivo más reciente de cada tienda.
                        Si False, consolida TODOS los archivos.
        
        Returns:
            DataFrame consolidado
        """
        all_data = []
        
        # Recorrer cada carpeta de tienda
        for tienda_dir in self.data_dir.iterdir():
            if not tienda_dir.is_dir() or tienda_dir.name == 'consolidated':
                continue
            
            csv_files = list(tienda_dir.glob('*.csv'))
            
            if not csv_files:
                logger.warning(f"No se encontraron archivos CSV en {tienda_dir.name}")
                continue
            
            if use_latest:
                # Usar solo el archivo más reciente
                csv_files = [max(csv_files, key=lambda p: p.stat().st_mtime)]
            
            logger.info(f"Procesando {len(csv_files)} archivo(s) de {tienda_dir.name}")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    all_data.append(df)
                    logger.info(f"  ✓ Cargado: {csv_file.name} ({len(df)} registros)")
                except Exception as e:
                    logger.error(f"  ✗ Error leyendo {csv_file.name}: {e}")
        
        if not all_data:
            logger.error("No se encontraron datos para consolidar")
            return pd.DataFrame()
        
        # Concatenar todos los DataFrames
        df_consolidated = pd.concat(all_data, ignore_index=True)
        logger.info(f"\n✅ Total consolidado: {len(df_consolidated)} registros de {len(all_data)} archivos")
        
        return df_consolidated
    
    def save_consolidated(self, df, suffix=''):
        """Guarda el DataFrame consolidado con timestamp en carpeta de fecha"""
        if df.empty:
            logger.warning("No hay datos para guardar")
            return None
        
        # Nombre más simple sin timestamp adicional (ya está en la carpeta)
        filename = f"consolidated_wine_data{suffix}.csv"
        filepath = self.output_dir / filename
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"💾 Archivo consolidado guardado: {filepath}")
        
        return filepath
    
    def get_latest_consolidated_dir(self):
        """Obtiene la carpeta consolidada más reciente"""
        date_dirs = [d for d in self.consolidated_base.iterdir() if d.is_dir() and d.name.isdigit()]
        
        if not date_dirs:
            # Se elimina el warning de aquí porque ya se maneja en el init
            return None
        
        latest_dir = max(date_dirs, key=lambda d: d.name)
        logger.info(f"📂 Usando carpeta consolidada: {latest_dir.name}")
        return latest_dir
    
    def get_latest_consolidated(self):
        """Obtiene el archivo consolidado más reciente"""
        latest_dir = self.get_latest_consolidated_dir()
        
        if not latest_dir:
            return None
        
        # Buscar el archivo consolidado en esa carpeta
        csv_files = list(latest_dir.glob('consolidated_*.csv'))
        
        if not csv_files:
            logger.warning(f"No hay archivos consolidados en {latest_dir.name}")
            return None
        
        latest = csv_files[0]  # Solo debería haber uno por carpeta
        logger.info(f"📂 Usando archivo: {latest.name}")
        return pd.read_csv(latest, encoding='utf-8-sig')
    
    def consolidate_and_save(self, use_latest=True):
        """Método todo-en-uno: consolida y guarda"""
        df = self.consolidate_all_scrapers(use_latest=use_latest)
        if not df.empty:
            filepath = self.save_consolidated(df)
            return df, filepath
        return None, None