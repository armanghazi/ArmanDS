"""
Módulo de Machine Learning: evaluación, optimización y persistencia de modelos.
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)


class ModelEvaluator:
    """Evaluación de modelos con métricas, gráficos y reportes."""

    def evaluate_model(
        self,
        model: BaseEstimator,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        task_type: str = "classification",
        model_name: str = "Modelo ML",
    ) -> Dict:
        """Evalúa un modelo de clasificación o regresión."""
        results_dir = "resultados"
        os.makedirs(results_dir, exist_ok=True)

        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        if task_type == "classification":
            metrics = self._evaluate_classification(
                y_test, y_pred, y_train, y_pred_train
            )
            self._plot_confusion_matrix(y_test, y_pred, model_name)
            if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
                self._plot_roc_curve(model, X_test, y_test, model_name)
        else:
            metrics = self._evaluate_regression(y_test, y_pred, y_train, y_pred_train)
            self._plot_regression_results(y_test, y_pred, model_name)

        self._generate_report(model, metrics, model_name, task_type)
        return metrics

    def _evaluate_classification(self, y_test, y_pred, y_train, y_pred_train) -> Dict:
        return {
            "accuracy_train": float((y_train == y_pred_train).mean()),
            "accuracy_test": float((y_test == y_pred).mean()),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "classification_report": classification_report(y_test, y_pred),
        }

    def _evaluate_regression(self, y_test, y_pred, y_train, y_pred_train) -> Dict:
        return {
            "mse_train": mean_squared_error(y_train, y_pred_train),
            "mse_test": mean_squared_error(y_test, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }

    def _plot_confusion_matrix(self, y_test, y_pred, model_name: str) -> None:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión - {model_name}")
        plt.ylabel("Valor Real")
        plt.xlabel("Valor Predicho")
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(f"resultados/confusion_matrix_{safe_name}.png")
        plt.close()

    def _plot_roc_curve(self, model, X_test, y_test, model_name: str) -> None:
        from sklearn.metrics import auc, roc_curve

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("Tasa de Falsos Positivos")
        plt.ylabel("Tasa de Verdaderos Positivos")
        plt.title(f"Curva ROC - {model_name}")
        plt.legend(loc="lower right")
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(f"resultados/roc_curve_{safe_name}.png")
        plt.close()

    def _plot_regression_results(self, y_test, y_pred, model_name: str) -> None:
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("Valores Reales")
        plt.ylabel("Predicciones")
        plt.title(f"Predicciones vs Valores Reales - {model_name}")
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(f"resultados/regression_results_{safe_name}.png")
        plt.close()

    def _generate_report(
        self, model, metrics: Dict, model_name: str, task_type: str
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.lower().replace(" ", "_")
        report_path = f"resultados/reporte_{safe_name}_{timestamp}.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== Reporte de Evaluación: {model_name} ===\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if hasattr(model, "get_params"):
                f.write("Parámetros del modelo:\n")
                for param, value in model.get_params().items():
                    f.write(f"- {param}: {value}\n")

            f.write("\nMétricas de Rendimiento\n")
            f.write("-" * 50 + "\n")
            if task_type == "classification":
                f.write(f"Accuracy (Train): {metrics['accuracy_train']:.4f}\n")
                f.write(f"Accuracy (Test): {metrics['accuracy_test']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {metrics['f1']:.4f}\n\n")
                f.write(metrics["classification_report"])
            else:
                f.write(f"MSE (Train): {metrics['mse_train']:.4f}\n")
                f.write(f"MSE (Test): {metrics['mse_test']:.4f}\n")
                f.write(f"RMSE: {metrics['rmse']:.4f}\n")
                f.write(f"MAE: {metrics['mae']:.4f}\n")
                f.write(f"R²: {metrics['r2']:.4f}\n")


class ModelOptimizer:
    """Optimización e hiperparámetros."""

    @staticmethod
    def optimize_hyperparameters(
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Dict,
        cv: int = 5,
        scoring: str = "accuracy",
        method: str = "grid",
        n_iter: int = 100,
    ) -> Tuple[BaseEstimator, Dict]:
        """Optimiza hiperparámetros con Grid Search o Random Search."""
        if method == "grid":
            search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        else:
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1
            )

        search.fit(X, y)
        results = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": pd.DataFrame(search.cv_results_),
        }
        return search.best_estimator_, results

    @staticmethod
    def cross_validate_model(
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = 5,
        scoring: Union[str, List[str]] = "accuracy",
        task_type: str = "classification",
    ) -> Dict:
        """Validación cruzada con una o varias métricas."""
        if task_type == "classification":
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

        if isinstance(scoring, str):
            scoring = [scoring]

        results = {}
        for metric in scoring:
            scores = cross_val_score(
                model, X, y, cv=cv_splitter, scoring=metric, n_jobs=-1
            )
            results[metric] = {
                "scores": scores,
                "mean": float(scores.mean()),
                "std": float(scores.std()),
            }
        return results


class ModelPersistence:
    """Guardado y carga de modelos con versionado."""

    @staticmethod
    def save_model(
        model: BaseEstimator,
        model_name: str,
        include_timestamp: bool = True,
    ) -> str:
        """Guarda el modelo en disco."""
        save_dir = "modelos"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        filename = (
            f"{model_name}_{timestamp}.joblib" if timestamp else f"{model_name}.joblib"
        )
        path = os.path.join(save_dir, filename)
        joblib.dump(model, path)
        print(f"Modelo guardado en: {path}")
        return path

    @staticmethod
    def load_model(path: str) -> BaseEstimator:
        """Carga un modelo guardado."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el modelo en: {path}")
        return joblib.load(path)
