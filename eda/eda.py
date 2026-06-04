"""
Módulo de Análisis Exploratorio de Datos (EDA)
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataLoader:
    """Carga y análisis automático de datos."""

    @staticmethod
    def cargar_csv_de_directorio(directorio: str) -> pd.DataFrame:
        """Carga el primer CSV del directorio y muestra un análisis básico."""
        if not os.path.isdir(directorio):
            raise ValueError(f"La ruta no es un directorio válido: {directorio}")

        csv_archivos = [f for f in os.listdir(directorio) if f.endswith(".csv")]
        if not csv_archivos:
            raise FileNotFoundError("No se encontraron archivos CSV en el directorio.")

        ruta_csv = os.path.join(directorio, csv_archivos[0])
        try:
            df = pd.read_csv(ruta_csv)
            print(f"\n{'#' * 50}")
            print(f"# Archivo CSV cargado: {ruta_csv} #")
            print(f"{'#' * 50}")
            DataLoader._mostrar_analisis_basico(df)
            return df
        except Exception as e:
            raise RuntimeError(f"Error al cargar el archivo CSV: {e}") from e

    @staticmethod
    def _mostrar_analisis_basico(df: pd.DataFrame) -> None:
        """Muestra análisis básico del DataFrame (compatible con scripts)."""
        print("\n" + "=" * 50)
        print("DataFrame completo".center(50))
        print("=" * 50)
        print(df.to_string())

        print("\n" + "=" * 50)
        print("Primeras filas".center(50))
        print("=" * 50)
        print(df.head().to_string())

        print("\n" + "=" * 50)
        print("Información del DataFrame".center(50))
        print("=" * 50)
        df.info()

        print("\n" + "=" * 50)
        print("Estadísticas descriptivas".center(50))
        print("=" * 50)
        print(df.describe().to_string())

        print("\n" + "=" * 50)
        print("Valores únicos por columna".center(50))
        print("=" * 50)
        for columna in df.columns:
            print(f"\n{columna} - VALORES ÚNICOS:")
            print(df[columna].value_counts())

    @staticmethod
    def cargar_multiple_csv(directorio: str, pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
        """Carga múltiples archivos CSV que coincidan con un patrón."""
        archivos = glob.glob(os.path.join(directorio, pattern))
        dataframes = {}

        for archivo in archivos:
            nombre = os.path.basename(archivo)
            try:
                dataframes[nombre] = pd.read_csv(archivo)
                print(f"Cargado: {nombre}")
            except Exception as e:
                print(f"Error al cargar {nombre}: {e}")

        return dataframes


class NullAnalyzer:
    """Análisis de valores nulos."""

    @staticmethod
    def null_analysis(
        df: pd.DataFrame, show_plot: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """Analiza valores nulos y devuelve estadísticas y recomendaciones."""
        null_info = (
            pd.DataFrame(
                {
                    "Columna": df.columns,
                    "Nulos": df.isnull().sum(),
                    "Porcentaje": (df.isnull().sum() / len(df)) * 100,
                }
            )
            .set_index("Columna")
            .sort_values(by="Nulos", ascending=False)
        )

        recomendaciones = {}
        for col in df.columns:
            pct_nulos = (df[col].isnull().sum() / len(df)) * 100
            if pct_nulos == 0:
                continue
            if pct_nulos < 5:
                recomendaciones[col] = "Considerar imputación por media/mediana"
            elif pct_nulos < 30:
                recomendaciones[col] = (
                    "Evaluar importancia de la variable y técnicas avanzadas de imputación"
                )
            else:
                recomendaciones[col] = "Considerar eliminar la columna"

        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.bar(null_info.index, null_info["Nulos"], color="skyblue")
            plt.title("Valores Nulos por Columna")
            plt.xlabel("Columnas")
            plt.ylabel("Cantidad de Nulos")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        return null_info, recomendaciones

    @staticmethod
    def analizar_patrones_nulos(df: pd.DataFrame) -> pd.DataFrame:
        """Detecta patrones de valores nulos entre columnas."""
        null_matrix = df.isnull().astype(int)
        null_corr = null_matrix.corr()

        print("Patrones de valores nulos correlacionados:")
        for col1 in null_corr.columns:
            for col2 in null_corr.index:
                if col1 < col2 and abs(null_corr.loc[col2, col1]) > 0.5:
                    print(f"{col1} - {col2}: {null_corr.loc[col2, col1]:.2f}")

        return null_corr


class DuplicateHandler:
    """Gestión de duplicados."""

    @staticmethod
    def find_duplicates(
        df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Encuentra filas duplicadas."""
        duplicados = df[df.duplicated(subset=subset, keep=False)]
        print(f"Se encontraron {len(duplicados)} filas duplicadas")
        return duplicados

    @staticmethod
    def remove_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """Elimina filas duplicadas."""
        len_original = len(df)
        df_limpio = df.drop_duplicates(subset=subset, keep=keep)
        print(f"Se eliminaron {len_original - len(df_limpio)} filas duplicadas")
        return df_limpio

    @staticmethod
    def get_duplicate_stats(
        df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> Dict:
        """Estadísticas de duplicación."""
        dup_mask = df.duplicated(subset=subset, keep=False)
        dup_df = df[dup_mask]
        group_cols = subset if subset else list(df.columns)

        dup_sizes = dup_df.groupby(group_cols, dropna=False).size()
        max_dup = int(dup_sizes.max()) if len(dup_sizes) else 0

        stats = {
            "total_rows": len(df),
            "duplicate_rows": int(dup_mask.sum()),
            "unique_rows": len(df.drop_duplicates(subset=subset)),
            "duplicate_groups": len(dup_sizes),
            "max_duplicates": max_dup,
        }
        stats["duplicate_percentage"] = (
            stats["duplicate_rows"] / stats["total_rows"] * 100 if stats["total_rows"] else 0
        )
        return stats
