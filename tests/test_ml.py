"""Tests for ML utilities."""

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from armands import ModelEvaluator, ModelOptimizer


class TestModelEvaluator(unittest.TestCase):
    def test_roc_with_string_labels(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame({"f1": rng.normal(size=80), "f2": rng.normal(size=80)})
        y = np.where(X["f1"] + X["f2"] > 0, "yes", "no")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            task_type="classification",
            model_name="rf_strings",
        )
        self.assertGreaterEqual(metrics["accuracy_test"], 0.0)


class TestModelOptimizer(unittest.TestCase):
    def test_cv_splits_adapt_to_small_data(self):
        X = pd.DataFrame({"f1": range(8)})
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        results = ModelOptimizer.cross_validate_model(
            RandomForestClassifier(random_state=42),
            X,
            y,
            cv=10,
            scoring="accuracy",
            task_type="classification",
        )
        self.assertIn("accuracy", results)
        self.assertEqual(len(results["accuracy"]["scores"]), 4)


if __name__ == "__main__":
    unittest.main()
