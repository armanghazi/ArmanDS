"""
DataVisualizer: unified facade for all visualizations.
"""

from . import correlation, plots


class DataVisualizer:
    """Advanced data visualizations."""

    plot_correlation_matrix = staticmethod(correlation.plot_correlation_matrix)
    plot_categorical_analysis = staticmethod(correlation.plot_categorical_analysis)
    plot_distribution = staticmethod(plots.plot_distribution)
    plot_time_series = staticmethod(plots.plot_time_series)


__all__ = ["DataVisualizer", "correlation", "plots"]
