"""
ArmanDS (pip package name: armands).

Library for EDA, visualization, preprocessing, and machine learning.
"""

from .eda import DataLoader, DuplicateHandler, NullAnalyzer
from .ml import ModelEvaluator, ModelOptimizer, ModelPersistence
from .preprocessing import DataCleaner, FeatureEngineer
from .visualizations import DataVisualizer

__version__ = "0.2.0"

__all__ = [
    "__version__",
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
