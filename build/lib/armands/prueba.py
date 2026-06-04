"""
Ejemplo de uso de la biblioteca ArmanDS (armands).
"""

import matplotlib

matplotlib.use("Agg")  # Sin ventanas bloqueantes en scripts

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from armands import (
    DataCleaner,
    DataVisualizer,
    DuplicateHandler,
    FeatureEngineer,
    ModelEvaluator,
    ModelOptimizer,
    ModelPersistence,
    NullAnalyzer,
)


def main():
    print("\n=== Ejemplo de uso de ArmanDS ===\n")

    data = {
        "edad": [25, 30, None, 40, 45, 28, 32, 38, 42, 47],
        "salario": [30000, 45000, 50000, None, 70000, 35000, 48000, 55000, 65000, 75000],
        "experiencia": [1, 5, 7, 10, None, 2, 6, 8, 12, 17],
        "satisfaccion": [7, None, 6, 8, 9, 7, 8, 7, None, 9],
        "departamento": ["IT", "HR", "IT", "Finance", "IT", None, "Finance", "IT", "HR", "Finance"],
    }
    df = pd.DataFrame(data)

    # EDA: nulos
    print("=== Análisis de Valores Nulos ===")
    null_info, recomendaciones = NullAnalyzer.null_analysis(df, show_plot=False)
    print(null_info)
    print("\nRecomendaciones:")
    for col, rec in recomendaciones.items():
        print(f"  {col}: {rec}")

    print("\n=== Patrones de Nulos ===")
    NullAnalyzer.analizar_patrones_nulos(df)

    # EDA: duplicados (ejemplo)
    df_dup = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    print("\n=== Duplicados ===")
    stats = DuplicateHandler.get_duplicate_stats(df_dup)
    print(stats)
    df = DuplicateHandler.remove_duplicates(df_dup)

    # Visualizaciones (guardadas en memoria con Agg; usar interactive=False)
    print("\n=== Visualizaciones ===")
    numeric_cols = ["edad", "salario", "experiencia", "satisfaccion"]
    DataVisualizer.plot_correlation_matrix(df[numeric_cols], interactive=False)
    DataVisualizer.plot_distribution(df, columns=numeric_cols, interactive=False)
    DataVisualizer.plot_categorical_analysis(
        df.dropna(subset=["departamento"]),
        cat_column="departamento",
        value_column="salario",
        interactive=False,
    )

    # Preprocesamiento
    print("\n=== Preprocesamiento ===")
    df_clean = DataCleaner.remove_outliers(
        df, columns=["salario", "experiencia"], method="zscore"
    )
    df_clean = DataCleaner.handle_missing_values(
        df_clean,
        strategy={"edad": "mean", "salario": "mean", "experiencia": "mean", "satisfaccion": "mean"},
    )
    df_clean = DataCleaner.handle_infinite_values(df_clean)

    df_encoded = FeatureEngineer.encode_categorical(
        df_clean, columns=["departamento"], method="onehot"
    )

    # ML
    print("\n=== Machine Learning ===")
    target = "salario"
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        task_type="regression",
        model_name="Random Forest",
    )
    print(f"R² en test: {metrics['r2']:.4f}")

    cv_results = ModelOptimizer.cross_validate_model(
        RandomForestRegressor(random_state=42),
        X,
        y,
        scoring=["neg_mean_squared_error", "r2"],
        task_type="regression",
    )
    print("\nValidación cruzada:")
    for metric, values in cv_results.items():
        print(f"  {metric}: media={values['mean']:.4f}, std={values['std']:.4f}")

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    }
    best_model, results = ModelOptimizer.optimize_hyperparameters(
        RandomForestRegressor(random_state=42),
        X,
        y,
        param_grid=param_grid,
        method="grid",
        scoring="neg_mean_squared_error",
    )
    print("\nMejores parámetros:", results["best_params"])

    model_path = ModelPersistence.save_model(best_model, "random_forest", include_timestamp=True)
    loaded = ModelPersistence.load_model(model_path)
    print(f"Modelo recargado: {type(loaded).__name__}")

    print("\nAnálisis completado. Reportes en resultados/ y modelos/")


if __name__ == "__main__":
    main()
