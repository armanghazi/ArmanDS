# ArmanDS

<div align="center">

Python library for exploratory data analysis (EDA), visualization, preprocessing, and machine learning.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Package](https://img.shields.io/badge/pypi-armands-0.2.0-orange.svg)](https://pypi.org/project/armands/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

ArmanDS consolidates the most common tasks in a data project lifecycle into a single installable package: **`armands`**. It covers automatic CSV loading, null and duplicate analysis, static and interactive (Plotly) charts, data cleaning and feature engineering, and model evaluation and optimization with automatic reports.

---

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Using in VS Code or another project](#using-in-vs-code-or-another-project)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Features](#features)
- [Usage by module](#usage-by-module)
- [Generated outputs](#generated-outputs)
- [Tests](#tests)
- [Local development](#local-development)
- [License](#license)

---

## Requirements

- Python 3.8 or higher
- pip

**Main dependencies** (installed automatically): `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `category-encoders`, `joblib`.

---

## Installation

### From PyPI

Once published:

```bash
pip install armands
```

### From GitHub

Without cloning manually:

```bash
pip install "git+https://github.com/armanghazi/ArmanDS.git"
```

### From a local clone

```bash
git clone https://github.com/armanghazi/ArmanDS.git
cd ArmanDS
pip install .
```

Verify the installation:

```bash
python -c "from armands import DataLoader, ModelEvaluator; print('OK')"
```

> **Note:** `requirements.txt` lists the same dependencies as `pyproject.toml` and serves as a reference. There is no need to run `pip install -r requirements.txt` if you already installed the package with `pip install .` or `pip install armands`.

---

## Using in VS Code or another project

1. Create a virtual environment in your data project (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate    # Windows
   # source .venv/bin/activate   # Linux / macOS
   ```

2. Install the library into that environment (`pip install armands`, `pip install .` from the clone, or the GitHub URL).

3. In VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `.venv` where you installed `armands`.

4. Place your CSVs in a folder (e.g. `data/`) and use the library in a script:

   ```python
   import matplotlib
   matplotlib.use("Agg")  # optional: suppresses blocking windows in scripts

   from armands import DataLoader, NullAnalyzer, DataCleaner, DataVisualizer

   df = DataLoader.load_csv_from_directory("data/")
   null_info, tips = NullAnalyzer.null_analysis(df, show_plot=False)
   DataVisualizer.plot_correlation_matrix(df, interactive=False, show=False)
   df = DataCleaner.handle_missing_values(df, strategy="most_frequent")
   ```

5. Run with **Run Python File** or `python my_analysis.py`.

---

## Project structure

The source lives in the repository root; setuptools publishes it as the **`armands`** package (`package-dir` in `pyproject.toml`).

```
ArmanDS/
├── __init__.py                   # Public API (import armands)
├── eda/                          # → armands.eda
│   └── eda.py                    # DataLoader, NullAnalyzer, DuplicateHandler
├── ml/                           # → armands.ml
│   └── model.py                  # ModelEvaluator, ModelOptimizer, ModelPersistence
├── preprocessing/                # → armands.preprocessing
│   └── preprocessing.py          # DataCleaner, FeatureEngineer
├── visualizations/               # → armands.visualizations
│   ├── correlation.py            # plot_correlation_matrix, plot_categorical_analysis
│   ├── plots.py                  # plot_distribution, plot_time_series
│   └── _plot.py                  # internal helper
├── tests/
├── pyproject.toml                # metadata and build config (PEP 517)
├── requirements.txt
└── README.md
```

After cloning:

```bash
pip install .
python -m pytest tests/ -v
```

---

## Quick start

```python
from armands import (
    DataLoader,
    NullAnalyzer,
    DuplicateHandler,
    DataVisualizer,
    DataCleaner,
    FeatureEngineer,
    ModelEvaluator,
    ModelOptimizer,
    ModelPersistence,
)
```

---

## Features

### Exploratory data analysis (EDA)

| Class | Methods |
|-------|---------|
| **DataLoader** | `load_csv_from_directory()`, `load_multiple_csv()` |
| **NullAnalyzer** | `null_analysis()`, `analyze_null_patterns()` |
| **DuplicateHandler** | `find_duplicates()`, `remove_duplicates()`, `get_duplicate_stats()` |

- Automatically loads the first CSV in a directory and prints a summary.
- Batch loading with glob patterns (`sales_*.csv`).
- Null analysis with recommendations and an optional bar chart.
- Detects correlations between missing-value patterns across columns.
- Duplicate statistics, detection, and removal.

### Visualization

| Class | Methods |
|-------|---------|
| **DataVisualizer** | `plot_correlation_matrix()`, `plot_distribution()`, `plot_categorical_analysis()`, `plot_time_series()` |

- Correlation matrices (Matplotlib or interactive Plotly).
- Histograms and boxplots per numeric column.
- Categorical variable analysis (counts and means per category).
- Time series with frequency aggregation (`D`, `W`, `M`, etc.).
- `show=False` on static charts for headless scripts (compatible with `Agg` backend).

### Preprocessing

| Class | Methods |
|-------|---------|
| **DataCleaner** | `remove_outliers()`, `handle_missing_values()`, `handle_infinite_values()` |
| **FeatureEngineer** | `create_date_features()`, `create_interaction_features()`, `encode_categorical()`, `scale_features()` |

- Outlier removal by **z-score** or **IQR**.
- **Numeric and categorical** imputation (global strategy or per-column dict; KNN for numeric only).
- One-hot, label, target, WOE, or binary encoding (`target_column` required for target/WOE).
- Standard, min-max, or robust scaling.
- Interaction features with safe division (no divide-by-zero errors).

### Machine learning

| Class | Methods |
|-------|---------|
| **ModelEvaluator** | `evaluate_model()` |
| **ModelOptimizer** | `optimize_hyperparameters()`, `cross_validate_model()` |
| **ModelPersistence** | `save_model()`, `load_model()` |

- Evaluation for **classification** and **regression** with metrics and plots.
- ROC curve for binary classification, including string labels (e.g. `yes`/`no`).
- Automatic text reports saved to `resultados/`.
- Grid Search and Random Search for hyperparameter tuning.
- Cross-validation with automatic `n_splits` adjustment for small datasets.
- Model save/load in `.joblib` format with optional timestamp.

---

## Usage by module

### EDA

```python
from armands import DataLoader, NullAnalyzer, DuplicateHandler

df = DataLoader.load_csv_from_directory("data/")
dfs = DataLoader.load_multiple_csv("data/", pattern="sales_*.csv")

null_info, recommendations = NullAnalyzer.null_analysis(df, show_plot=True)
NullAnalyzer.analyze_null_patterns(df)

stats = DuplicateHandler.get_duplicate_stats(df)
dupes = DuplicateHandler.find_duplicates(df)
df_clean = DuplicateHandler.remove_duplicates(df)
```

### Visualization

```python
from armands import DataVisualizer

# Interactive (notebook / browser)
DataVisualizer.plot_correlation_matrix(df, interactive=True)

# Headless script
DataVisualizer.plot_correlation_matrix(df, interactive=False, show=False)
DataVisualizer.plot_distribution(df, columns=["age", "salary"], show=False)
DataVisualizer.plot_categorical_analysis(
    df, cat_column="department", value_column="salary", show=False
)
DataVisualizer.plot_time_series(
    df, date_column="date", value_columns="sales", freq="M", show=False
)
```

### Preprocessing

```python
from armands import DataCleaner, FeatureEngineer

df = DataCleaner.remove_outliers(df, columns=["salary"], method="zscore")

# Global: numeric + categorical
df = DataCleaner.handle_missing_values(df, strategy="most_frequent")

# Per-column
df = DataCleaner.handle_missing_values(
    df,
    strategy={"age": "mean", "department": "most_frequent"},
)
df = DataCleaner.handle_infinite_values(df)

df = FeatureEngineer.create_date_features(df, "date")
df = FeatureEngineer.create_interaction_features(
    df, columns=[("price", "quantity")], operation="multiply"
)
df = FeatureEngineer.encode_categorical(df, columns=["department"], method="onehot")
df, scaler = FeatureEngineer.scale_features(df, method="standard")
```

### Machine learning

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from armands import ModelEvaluator, ModelOptimizer, ModelPersistence

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor().fit(X_train, y_train)

evaluator = ModelEvaluator()
metrics = evaluator.evaluate_model(
    model, X_train, X_test, y_train, y_test,
    task_type="regression",
    model_name="Random Forest",
)

cv = ModelOptimizer.cross_validate_model(
    RandomForestRegressor(), X, y,
    cv=5,
    scoring=["neg_mean_squared_error", "r2"],
    task_type="regression",
)

param_grid = {"n_estimators": [50, 100], "max_depth": [None, 10]}
best_model, results = ModelOptimizer.optimize_hyperparameters(
    RandomForestRegressor(), X, y,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
)

path = ModelPersistence.save_model(best_model, "random_forest")
loaded = ModelPersistence.load_model(path)
```

---

## Generated outputs

| Folder | Contents |
|--------|----------|
| `resultados/` | Text reports, confusion matrices, ROC curves, regression plots |
| `modelos/` | Models saved in `.joblib` format with timestamp |

These folders are created in the current working directory when running evaluations or saving models. They are not included in the installed package and are listed in `.gitignore`.

---

## Tests

```bash
pip install .
python -m pytest tests/ -v
```

Covers categorical imputation, safe interaction features (divide-by-zero), ROC curve with string labels, and cross-validation on small datasets.

---

## Local development

```bash
git clone https://github.com/armanghazi/ArmanDS.git
cd ArmanDS
python -m venv .venv
.venv\Scripts\activate
pip install .
python -m pytest tests/ -v
```

To publish to PyPI (once credentials are configured):

```bash
pip install build twine
python -m build
twine upload dist/*
```

## 👤 Author

**Arman Ghaziaskari Naeini** _GIS & Remote Sensing Specialist | Spatial Data Scientist | GeoAI Enthusiast_ Bilbao, Spain

- Portfolio : [armanghazi.github.io/portfolio/projects](https://armanghazi.github.io/portfolio/projects)
- GitHub : [@armanghazi](https://github.com/armanghazi)
- LinkedIn : [arman-ghaziaskari](https://www.linkedin.com/in/arman-ghaziaskari/)
---

## License

This project is licensed under the [MIT License](LICENSE).
