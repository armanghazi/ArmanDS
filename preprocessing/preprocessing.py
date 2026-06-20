"""
Preprocessing: data cleaning and feature engineering.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder, WOEEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class DataCleaner:
    """Data cleaning."""

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        columns: List[str],
        method: str = "zscore",
        threshold: float = 3,
    ) -> pd.DataFrame:
        """Remove outliers using z-score or IQR method."""
        df = df.copy()

        if method == "zscore":
            for col in columns:
                std = df[col].std()
                if std == 0 or pd.isna(std):
                    continue
                z_scores = np.abs((df[col] - df[col].mean()) / std)
                df = df[z_scores < threshold]
        elif method == "iqr":
            for col in columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df = df[
                    ~(
                        (df[col] < (q1 - threshold * iqr))
                        | (df[col] > (q3 + threshold * iqr))
                    )
                ]

        return df

    @staticmethod
    def _fill_categorical(series: pd.Series, strategy: str) -> pd.Series:
        """Impute missing values in a non-numeric column."""
        if not series.isna().any():
            return series
        if strategy in ("mean", "median"):
            strategy = "most_frequent"
        if strategy == "constant":
            return series.fillna("")
        mode = series.mode()
        fill = mode.iloc[0] if not mode.empty else "unknown"
        return series.fillna(fill)

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: Union[str, Dict] = "mean",
        method: str = "simple",
    ) -> pd.DataFrame:
        """Handle missing values in numeric and categorical columns."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        if isinstance(strategy, str):
            if method == "knn":
                if len(numeric_cols) == 0:
                    raise ValueError("KNN imputation requires at least one numeric column.")
                imputer = KNNImputer(n_neighbors=5)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            else:
                for col in numeric_cols:
                    imputer = SimpleImputer(strategy=strategy)
                    df[[col]] = imputer.fit_transform(df[[col]])
                for col in non_numeric_cols:
                    df[col] = DataCleaner._fill_categorical(df[col], strategy)
        else:
            for col, strat in strategy.items():
                if col not in df.columns:
                    continue
                if col in numeric_cols:
                    imputer = SimpleImputer(strategy=strat)
                    df[[col]] = imputer.fit_transform(df[[col]])
                else:
                    df[col] = DataCleaner._fill_categorical(df[col], strat)

        return df

    @staticmethod
    def handle_infinite_values(
        df: pd.DataFrame, replacement: Union[str, float] = "mean"
    ) -> pd.DataFrame:
        """Handle infinite values in numeric columns."""
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            mask = np.isinf(df[col])
            if not mask.any():
                continue
            finite = df.loc[~mask, col]
            if isinstance(replacement, str):
                if replacement == "mean":
                    val = finite.mean()
                elif replacement == "median":
                    val = finite.median()
                elif replacement == "max":
                    val = finite.max()
                else:
                    val = finite.min()
            else:
                val = replacement
            df.loc[mask, col] = val

        return df


class FeatureEngineer:
    """Feature engineering."""

    @staticmethod
    def create_date_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Extract temporal features from a date column."""
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        df[f"{date_column}_year"] = df[date_column].dt.year
        df[f"{date_column}_month"] = df[date_column].dt.month
        df[f"{date_column}_day"] = df[date_column].dt.day
        df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek
        df[f"{date_column}_quarter"] = df[date_column].dt.quarter
        df[f"{date_column}_is_weekend"] = df[date_column].dt.dayofweek.isin([5, 6]).astype(
            int
        )
        df[f"{date_column}_is_month_end"] = df[date_column].dt.is_month_end.astype(int)
        df[f"{date_column}_is_month_start"] = df[date_column].dt.is_month_start.astype(int)

        return df

    @staticmethod
    def create_interaction_features(
        df: pd.DataFrame, columns: List[tuple], operation: str = "multiply"
    ) -> pd.DataFrame:
        """Create interaction features between column pairs."""
        df = df.copy()
        operations = {
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y,
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "mean": lambda x, y: (x + y) / 2,
            "max": lambda x, y: np.maximum(x, y),
            "min": lambda x, y: np.minimum(x, y),
        }

        for col1, col2 in columns:
            new_col = f"{col1}_{operation}_{col2}"
            if operation == "divide":
                df[new_col] = np.divide(
                    df[col1],
                    df[col2],
                    out=np.zeros(len(df), dtype=float),
                    where=df[col2].to_numpy() != 0,
                )
            else:
                df[new_col] = operations[operation](df[col1], df[col2])

        return df

    @staticmethod
    def encode_categorical(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "onehot",
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in columns:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()
                fill = mode_val.iloc[0] if not mode_val.empty else "unknown"
                df[col] = df[col].fillna(fill)

        if method == "onehot":
            df = pd.get_dummies(df, columns=columns, prefix_sep="_")
        elif method == "label":
            for col in columns:
                df[col] = df[col].astype("category").cat.codes
        elif method == "target":
            if not target_column:
                raise ValueError("target_column is required for target encoding.")
            encoder = TargetEncoder()
            df[columns] = encoder.fit_transform(df[columns], df[target_column])
        elif method == "woe":
            if not target_column:
                raise ValueError("target_column is required for WOE encoding.")
            encoder = WOEEncoder()
            df[columns] = encoder.fit_transform(df[columns], df[target_column])
        elif method == "binary":
            for col in columns:
                if df[col].nunique() == 2:
                    df[col] = (df[col] == df[col].value_counts().index[0]).astype(int)

        return df

    @staticmethod
    def scale_features(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "standard",
    ) -> Tuple[pd.DataFrame, object]:
        """Scale numeric features."""
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()

        df[columns] = scaler.fit_transform(df[columns])
        return df, scaler
