"""
Distribuciones, series temporales y gráficos relacionados con regresión.
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


def plot_distribution(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    interactive: bool = False,
) -> None:
    """Histogramas y boxplots para columnas numéricas."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        raise ValueError("No hay columnas numéricas para visualizar.")

    n_cols = len(columns)

    if interactive:
        fig = make_subplots(
            rows=n_cols,
            cols=2,
            subplot_titles=[f"Histograma de {col}" for col in columns]
            + [f"Box Plot de {col}" for col in columns],
        )
        for i, col in enumerate(columns, 1):
            fig.add_trace(go.Histogram(x=df[col], name=col), row=i, col=1)
            fig.add_trace(go.Box(y=df[col], name=col), row=i, col=2)
        fig.update_layout(height=300 * n_cols, showlegend=False)
        fig.show()
    else:
        if n_cols == 1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(data=df, x=columns[0], ax=axes[0])
            axes[0].set_title(f"Histograma de {columns[0]}")
            sns.boxplot(data=df, y=columns[0], ax=axes[1])
            axes[1].set_title(f"Box Plot de {columns[0]}")
        else:
            fig, axes = plt.subplots(n_cols, 2, figsize=(12, 4 * n_cols))
            for i, col in enumerate(columns):
                sns.histplot(data=df, x=col, ax=axes[i, 0])
                axes[i, 0].set_title(f"Histograma de {col}")
                sns.boxplot(data=df, y=col, ax=axes[i, 1])
                axes[i, 1].set_title(f"Box Plot de {col}")
        plt.tight_layout()
        plt.show()


def plot_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_columns: Union[str, List[str]],
    freq: str = "D",
    interactive: bool = False,
) -> None:
    """Visualización de series temporales."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    if isinstance(value_columns, str):
        value_columns = [value_columns]

    df_grouped = df.groupby(pd.Grouper(key=date_column, freq=freq))[value_columns].mean()

    if interactive:
        fig = go.Figure()
        for col in value_columns:
            fig.add_trace(
                go.Scatter(x=df_grouped.index, y=df_grouped[col], name=col)
            )
        fig.update_layout(
            title="Análisis de Series Temporales",
            xaxis_title="Fecha",
            yaxis_title="Valor",
        )
        fig.show()
    else:
        plt.figure(figsize=(12, 6))
        for col in value_columns:
            plt.plot(df_grouped.index, df_grouped[col], label=col)
        plt.title("Análisis de Series Temporales")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
