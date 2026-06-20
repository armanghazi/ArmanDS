"""
Exploratory Data Analysis (EDA) module.
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..visualizations._plot import show_or_close


class DataLoader:
    """Automatic data loading and analysis."""

    @staticmethod
    def load_csv_from_directory(directory: str) -> pd.DataFrame:
        """Load the first CSV found in the directory and print a basic analysis."""
        if not os.path.isdir(directory):
            raise ValueError(f"Path is not a valid directory: {directory}")

        csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the directory.")

        csv_path = os.path.join(directory, csv_files[0])
        try:
            df = pd.read_csv(csv_path)
            print(f"\n{'#' * 50}")
            print(f"# CSV file loaded: {csv_path} #")
            print(f"{'#' * 50}")
            DataLoader._show_basic_analysis(df)
            return df
        except Exception as e:
            raise RuntimeError(f"Error loading CSV file: {e}") from e

    @staticmethod
    def _show_basic_analysis(df: pd.DataFrame) -> None:
        """Print basic DataFrame analysis."""
        print("\n" + "=" * 50)
        print("Full DataFrame".center(50))
        print("=" * 50)
        print(df.to_string())

        print("\n" + "=" * 50)
        print("First rows".center(50))
        print("=" * 50)
        print(df.head().to_string())

        print("\n" + "=" * 50)
        print("DataFrame info".center(50))
        print("=" * 50)
        df.info()

        print("\n" + "=" * 50)
        print("Descriptive statistics".center(50))
        print("=" * 50)
        print(df.describe().to_string())

        print("\n" + "=" * 50)
        print("Unique values per column".center(50))
        print("=" * 50)
        for col in df.columns:
            print(f"\n{col} - UNIQUE VALUES:")
            print(df[col].value_counts())

    @staticmethod
    def load_multiple_csv(directory: str, pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
        """Load multiple CSV files matching a pattern."""
        files = glob.glob(os.path.join(directory, pattern))
        dataframes = {}

        for filepath in files:
            name = os.path.basename(filepath)
            try:
                dataframes[name] = pd.read_csv(filepath)
                print(f"Loaded: {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")

        return dataframes


class NullAnalyzer:
    """Null value analysis."""

    @staticmethod
    def null_analysis(
        df: pd.DataFrame, show_plot: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """Analyze null values and return statistics and recommendations."""
        null_info = (
            pd.DataFrame(
                {
                    "Column": df.columns,
                    "Nulls": df.isnull().sum(),
                    "Percentage": (df.isnull().sum() / len(df)) * 100,
                }
            )
            .set_index("Column")
            .sort_values(by="Nulls", ascending=False)
        )

        recommendations = {}
        for col in df.columns:
            pct_null = (df[col].isnull().sum() / len(df)) * 100
            if pct_null == 0:
                continue
            if pct_null < 5:
                recommendations[col] = "Consider mean/median imputation"
            elif pct_null < 30:
                recommendations[col] = (
                    "Evaluate variable importance and advanced imputation techniques"
                )
            else:
                recommendations[col] = "Consider dropping the column"

        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.bar(null_info.index, null_info["Nulls"], color="skyblue")
            plt.title("Null Values per Column")
            plt.xlabel("Columns")
            plt.ylabel("Null Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            show_or_close(True)

        return null_info, recommendations

    @staticmethod
    def analyze_null_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect correlated null patterns between columns."""
        null_matrix = df.isnull().astype(int)
        null_corr = null_matrix.corr()

        print("Correlated null value patterns:")
        for col1 in null_corr.columns:
            for col2 in null_corr.index:
                if col1 < col2 and abs(null_corr.loc[col2, col1]) > 0.5:
                    print(f"{col1} - {col2}: {null_corr.loc[col2, col1]:.2f}")

        return null_corr


class DuplicateHandler:
    """Duplicate row management."""

    @staticmethod
    def find_duplicates(
        df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Find duplicate rows."""
        duplicates = df[df.duplicated(subset=subset, keep=False)]
        print(f"Found {len(duplicates)} duplicate rows")
        return duplicates

    @staticmethod
    def remove_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """Remove duplicate rows."""
        original_len = len(df)
        df_clean = df.drop_duplicates(subset=subset, keep=keep)
        print(f"Removed {original_len - len(df_clean)} duplicate rows")
        return df_clean

    @staticmethod
    def get_duplicate_stats(
        df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> Dict:
        """Return duplicate statistics for the DataFrame."""
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
