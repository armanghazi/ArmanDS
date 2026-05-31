# ArmanDS

Biblioteca de utilidades para ciencia de datos que incluye herramientas para análisis exploratorio, visualización, preprocesamiento y machine learning.

## Requisitos previos

- Python 3.9 o superior
- pip (gestor de paquetes de Python)

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/armanghazi/ArmanDS.git
cd ArmanDS
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso rápido

Para probar la funcionalidad básica, ejecuta el script de ejemplo:

```bash
python prueba.py
```

Este script demostrará:
- Análisis de valores nulos
- Generación de visualizaciones
- Preprocesamiento de datos
- Entrenamiento y evaluación de un modelo de machine learning

Los resultados y visualizaciones se guardarán en el directorio `resultados/`.

<div align="center">

![ArmanDS Logo](img/logo.png)

Una biblioteca de ciencia de datos diseñada para simplificar el análisis exploratorio de datos y machine learning.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## 📋 Tabla de Contenidos
- [Instalación](#-instalación)
- [Características](#-características)
- [Uso](#-uso)
  - [Análisis Exploratorio de Datos](#análisis-exploratorio-de-datos)
  - [Visualización](#visualización)
  - [Preprocesamiento](#preprocesamiento)
  - [Machine Learning](#machine-learning)
- [Ejemplos](#-ejemplos)
- [Documentación](#-documentación)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

## ✨ Características

### Análisis Exploratorio de Datos (EDA)
- **DataLoader**: Carga y análisis automático de datos
  - `cargar_csv_de_directorio()`: Carga automática de archivos CSV
  - `cargar_multiple_csv()`: Carga múltiples archivos CSV con patrones

- **NullAnalyzer**: Análisis de valores nulos
  - `null_analysis()`: Análisis detallado de valores nulos
  - `analizar_patrones_nulos()`: Detección de patrones en valores nulos

- **DuplicateHandler**: Gestión de duplicados
  - `find_duplicates()`: Encuentra filas duplicadas
  - `remove_duplicates()`: Elimina duplicados
  - `get_duplicate_stats()`: Estadísticas de duplicación

### Visualización
- **DataVisualizer**: Visualizaciones avanzadas
  - `plot_correlation_matrix()`: Matrices de correlación interactivas
  - `plot_distribution()`: Distribuciones y boxplots
  - `plot_categorical_analysis()`: Análisis de variables categóricas
  - `plot_time_series()`: Visualización de series temporales

### Preprocesamiento
- **DataCleaner**: Limpieza de datos
  - `remove_outliers()`: Eliminación de valores atípicos
  - `handle_missing_values()`: Manejo de valores faltantes
  - `handle_infinite_values()`: Tratamiento de valores infinitos

- **FeatureEngineer**: Ingeniería de características
  - `create_date_features()`: Extracción de características temporales
  - `create_interaction_features()`: Creación de interacciones
  - `encode_categorical()`: Codificación de variables categóricas
  - `scale_features()`: Escalado de características

### Machine Learning
- **ModelEvaluator**: Evaluación de modelos
  - `evaluate_model()`: Evaluación completa con métricas y visualizaciones
  - Soporte para clasificación y regresión
  - Generación automática de reportes

- **ModelOptimizer**: Optimización de modelos
  - `optimize_hyperparameters()`: Optimización de hiperparámetros
  - `cross_validate_model()`: Validación cruzada con múltiples métricas

- **ModelPersistence**: Gestión de modelos
  - `save_model()`: Guardado de modelos con versionado
  - `load_model()`: Carga de modelos guardados

## 🎯 Uso

### Análisis Exploratorio de Datos
```python
from datautilityhub.eda import DataLoader, NullAnalyzer
Cargar datos
df = DataLoader.cargar_csv_de_directorio("ruta/datos")
Analizar valores nulos
null_info, recomendaciones = NullAnalyzer.null_analysis(df)
```
### Visualización
```python:readme.md
from datautilityhub.visualization import DataVisualizer
Crear matriz de correlación interactiva
DataVisualizer.plot_correlation_matrix(df, interactive=True)
Análisis de variables categóricas
DataVisualizer.plot_categorical_analysis(df, 'categoria', 'valor')
```
### Preprocesamiento

from datautilityhub.preprocessing import DataCleaner, FeatureEngineer
- **Limpiar datos**
- `df_clean = DataCleaner.remove_outliers(df, columns=['columna'])`
- `df_clean = DataCleaner.handle_missing_values(df_clean, strategy='mean')`
- **Crear características**
- `df_features = FeatureEngineer.create_date_features(df, 'fecha')`

### Machine Learning

from datautilityhub.ml import ModelEvaluator, ModelOptimizer, ModelPersistence
- **Evaluar modelo**
`evaluator = ModelEvaluator()`
`metrics = evaluator.evaluate_model(model, X_train, X_test, y_train, y_test)`
- **Optimizar hiperparámetros**
`best_model, results = ModelOptimizer.optimize_hyperparameters(model, X, y, param_grid)`

## 📚 Documentación
Para documentación detallada, visita nuestra [wiki](https://github.com/armanghazi/ArmanDS/wiki).

## 🤝 Contribuir
Las contribuciones son bienvenidas! Por favor, lee nuestras [guías de contribución](CONTRIBUTING.md).

## 📄 Licencia
Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

