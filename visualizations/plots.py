"""
Distribution histograms, time series, and statistical plots.
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

from ._plot import show_or_close


def plot_distribution(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    interactive: bool = False,
    show: bool = True,
) -> None:
    """Plot histograms and boxplots for numeric columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns to visualize.")

    n_cols = len(columns)

    if interactive:
        fig = make_subplots(
            rows=n_cols,
            cols=2,
            subplot_titles=[f"Histogram of {col}" for col in columns]
            + [f"Box Plot of {col}" for col in columns],
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
            axes[0].set_title(f"Histogram of {columns[0]}")
            sns.boxplot(data=df, y=columns[0], ax=axes[1])
            axes[1].set_title(f"Box Plot of {columns[0]}")
        else:
            fig, axes = plt.subplots(n_cols, 2, figsize=(12, 4 * n_cols))
            for i, col in enumerate(columns):
                sns.histplot(data=df, x=col, ax=axes[i, 0])
                axes[i, 0].set_title(f"Histogram of {col}")
                sns.boxplot(data=df, y=col, ax=axes[i, 1])
                axes[i, 1].set_title(f"Box Plot of {col}")
        plt.tight_layout()
        show_or_close(show)


def plot_time_series(
    df: pd.DataFrame,
    date_column: str,
    value_columns: Union[str, List[str]],
    freq: str = "D",
    interactive: bool = False,
    show: bool = True,
) -> None:
    """Visualize time series data."""
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
            title="Time Series Analysis",
            xaxis_title="Date",
            yaxis_title="Value",
        )
        fig.show()
    else:
        plt.figure(figsize=(12, 6))
        for col in value_columns:
            plt.plot(df_grouped.index, df_grouped[col], label=col)
        plt.title("Time Series Analysis")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        show_or_close(show)
