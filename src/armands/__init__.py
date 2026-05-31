"""
ArmanDS (armands) - Biblioteca para EDA, visualización, preprocesamiento y ML.
"""

from .eda import DataLoader, DuplicateHandler, NullAnalyzer
from .ml import ModelEvaluator, ModelOptimizer, ModelPersistence
from .preprocesamiento import DataCleaner, FeatureEngineer
from .visualizaciones import DataVisualizer

__version__ = "0.2.0"

__all__ = [
    "DataLoader",
    "NullAnalyzer",
    "DuplicateHandler",
    "DataVisualizer",
    "DataCleaner",
    "FeatureEngineer",
    "ModelEvaluator",
    "ModelOptimizer",
    "ModelPersistence",
]
