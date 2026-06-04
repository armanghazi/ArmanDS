"""
Visualizaciones de correlación y variables categóricas.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: tuple = (10, 8),
    method: str = "pearson",
    interactive: bool = False,
) -> None:
    """Matriz de correlación (estática o interactiva con Plotly)."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No hay columnas numéricas para calcular correlación.")

    corr = numeric_df.corr(method=method)

    if interactive:
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(
            title=f"Matriz de Correlación ({method})",
            width=figsize[0] * 100,
            height=figsize[1] * 100,
        )
        fig.show()
    else:
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", center=0)
        plt.title(f"Matriz de Correlación ({method})")
        plt.tight_layout()
        plt.show()


def plot_categorical_analysis(
    df: pd.DataFrame,
    cat_column: str,
    value_column: Optional[str] = None,
    top_n: int = 10,
    interactive: bool = False,
) -> None:
    """Análisis de variables categóricas."""
    if value_column:
        data = (
            df.groupby(cat_column)[value_column]
            .agg(["count", "mean", "sum"])
            .sort_values("sum", ascending=False)
            .head(top_n)
        )
    else:
        data = df[cat_column].value_counts().head(top_n)

    if interactive:
        if value_column:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=[
                    f"Conteo por {cat_column}",
                    f"Promedio de {value_column} por {cat_column}",
                ],
            )
            fig.add_trace(
                go.Bar(x=data.index.astype(str), y=data["count"], name="Conteo"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(x=data.index.astype(str), y=data["mean"], name="Promedio"),
                row=2,
                col=1,
            )
        else:
            fig = go.Figure(data=go.Bar(x=data.index.astype(str), y=data.values))
            fig.update_layout(title=f"Distribución de {cat_column}")
        fig.show()
    else:
        if value_column:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            data["count"].plot(kind="bar", ax=ax1)
            ax1.set_title(f"Conteo por {cat_column}")
            ax1.tick_params(axis="x", rotation=45)
            data["mean"].plot(kind="bar", ax=ax2)
            ax2.set_title(f"Promedio de {value_column} por {cat_column}")
            ax2.tick_params(axis="x", rotation=45)
        else:
            plt.figure(figsize=(12, 6))
            data.plot(kind="bar")
            plt.title(f"Distribución de {cat_column}")
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
