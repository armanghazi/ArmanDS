"""Tests for preprocessing utilities."""

import unittest

import numpy as np
import pandas as pd

from armands import DataCleaner, FeatureEngineer


class TestDataCleaner(unittest.TestCase):
    def test_global_most_frequent_fills_categorical(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "y"]})
        out = DataCleaner.handle_missing_values(df, strategy="most_frequent")
        self.assertEqual(out["b"].isna().sum(), 0)
        self.assertEqual(out["a"].isna().sum(), 0)

    def test_dict_strategy_per_column(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": ["x", None]})
        out = DataCleaner.handle_missing_values(
            df, strategy={"a": "mean", "b": "most_frequent"}
        )
        self.assertFalse(out.isna().any().any())

    def test_divide_by_zero_in_interaction(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 0.0]})
        out = FeatureEngineer.create_interaction_features(
            df, columns=[("x", "y")], operation="divide"
        )
        self.assertEqual(out["x_divide_y"].iloc[1], 0.0)


if __name__ == "__main__":
    unittest.main()
