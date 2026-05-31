"""
DataVisualizer: fachada unificada para todas las visualizaciones.
"""

from . import correlacion, regresiones


class DataVisualizer:
    """Visualizaciones avanzadas de datos."""

    plot_correlation_matrix = staticmethod(correlacion.plot_correlation_matrix)
    plot_categorical_analysis = staticmethod(correlacion.plot_categorical_analysis)
    plot_distribution = staticmethod(regresiones.plot_distribution)
    plot_time_series = staticmethod(regresiones.plot_time_series)


__all__ = ["DataVisualizer", "correlacion", "regresiones"]
