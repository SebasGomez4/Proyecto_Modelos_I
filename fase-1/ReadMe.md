# Predicción del precio de venta de viviendas – Fase 1

Proyecto Integrador v1 · 2026 II · Modelos y Simulación de Sistemas I

## Integrantes

Samuel Echeverri Ortiz (@Eche0813)
Miguel Angel Foronda (@Foronda713)
Sebastian Gómez Quintero (@SebasGomez4)

## Descripción del problema

Estimar el precio de venta de una vivienda a partir de sus características (dimensiones, calidad de materiales y acabados, año de construcción, barrio, tipo de vivienda y condiciones de la venta). Es un problema de aprendizaje supervisado de **regresión**; la variable objetivo es `SalePrice` (USD).

## Fuente del conjunto de datos

Competencia de Kaggle [`aecincode_houseprices`](https://www.kaggle.com/competitions/aecincode_houseprices), basada en el *Ames Housing Dataset* (De Cock, 2011) y relacionada con [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/).

- `data/train.csv`: 1.758 viviendas, 79 predictoras + `SalePrice` + identificadores (`Order`, `PID`, `Id`).
- `data/test.csv`: 1.172 viviendas sin precio; se usan como "datos nuevos" para probar el modelo guardado.

## Objetivo del modelo

Dada la ficha de una vivienda, predecir su precio de venta en dólares. En esta fase el énfasis está en un proceso reproducible, documentado y libre de fuga de información, que servirá de base para las siguientes fases (scripts, API REST y monitoreo).

## Algoritmo

`HistGradientBoostingRegressor` (Gradient Boosting) dentro de un `Pipeline` con el preprocesamiento anterior, envuelto en `TransformedTargetRegressor` (entrena con `log1p(precio)` y devuelve dólares). Modelo base: `DummyRegressor` que predice siempre la mediana.

## Métrica

**MAE** (error medio absoluto, en USD) como métrica principal por ser interpretable y poco dominada por las viviendas extremas; se reportan también RMSE, R² y RMSLE (la métrica de la competencia). Validación: 5 folds sobre entrenamiento; evaluación final única sobre el 20 % de prueba.

## Principales resultados

| | MAE (USD) | RMSE (USD) | R² | RMSLE |
|---|---|---|---|---|
| Baseline (mediana) – test | 53.977 | 80.588 | −0,047 | 0,419 |
| **Gradient Boosting – test** | **15.177** | **22.886** | **0,916** | **0,139** |
| Gradient Boosting – validación cruzada (train) | 15.594 ± 1.177 | 26.131 | 0,884 | 0,137 |

El modelo reduce el MAE en ≈ 72 % respecto al baseline y el error de test es coherente con el de validación cruzada (sin señales de sobreajuste). El error absoluto es mayor en las viviendas más caras; el relativo, en las más baratas. Las variables más influyentes son la calidad general y el área habitable.

## Cómo ejecutar el notebook

1. Clonar el repositorio y entrar a `fase-1/`.
2. Instalar dependencias: `pip install -r requirements.txt`.
3. Abrir `notebook.ipynb` y ejecutar todas las celdas (*Run All*); o desde consola: `jupyter nbconvert --to notebook --execute notebook.ipynb`.

Al ejecutarse se regeneran `modelo.joblib` y `metricas.json`. Uso del modelo guardado:

```python
import joblib, pandas as pd
modelo = joblib.load("modelo.joblib")
nuevos = pd.read_csv("data/test.csv")
predicciones = modelo.predict(nuevos.drop(columns=["Order", "PID", "Id"]))   # USD
```

El archivo `modelo.joblib` se generó con scikit-learn 1.8.0; con otra versión puede no cargar, en cuyo caso basta re-ejecutar el notebook.

## Estructura de la carpeta

```
fase-1/
├── notebook.ipynb      # proceso completo, ejecutable de principio a fin
├── modelo.joblib       # modelo entrenado (pipeline completo)
├── README.md
├── requirements.txt
├── metricas.json       # métricas y metadatos de la ejecución entregada
└── data/
    ├── train.csv
    └── test.csv
```

## Preparación de los datos y prevención de fuga de información

**Valores faltantes.** 26 de las 79 predictoras tienen vacíos. En `train.csv`, `Mas Vnr Area` (0,68 %) es la variable dentro del rango 0,1 %–2 %; las de sótano (`Bsmt Qual`, `Bsmt Cond`, `BsmtFin Type 1/2`) están en 2,39 %. La mayoría de vacíos **no son datos perdidos sino "la vivienda no tiene esa característica"** (sótano, garaje, chimenea, piscina), lo que se verificó cruzando con variables numéricas de referencia. Tratamiento:

- Categóricas con vacío "no aplica": categoría `"None"` (el vacío es información y se asocia a precios distintos).
- Numéricas (`Lot Frontage`, `Mas Vnr Area`, `Garage Yr Blt`, ...): mediana + indicador de ausencia.
- Categóricas ordenadas: entero ordenado; nominales: one-hot.
- Objetivo entrenado sobre `log(1 + precio)`; sin escalamiento (modelo de árboles).
- No se descartó ninguna variable predictora (solo los identificadores). Las variables muy correlacionadas (|r| > 0,8) se conservaron con justificación en el notebook. Los valores atípicos no se eliminan.

**Separación.** 80 % entrenamiento / 20 % prueba, aleatoria simple, `random_state=42`. Cada fila es una vivienda distinta, sin grupos naturales.

**Fuga de información.** El objetivo no se usa como predictora; la partición se hace antes de explorar y el test se evalúa una sola vez; imputación y codificación se ajustan solo con entrenamiento (dentro de un `Pipeline`, reajustado en cada fold de la validación cruzada); se verificó con `assert` y se midió el aporte de las variables de la venta (`Sale Type`, `Sale Condition`, `Mo Sold`, `Yr Sold`).

## Limitaciones y trabajo futuro

Datos de una sola ciudad y época (Ames, 2006–2010); un único split 80/20; hiperparámetros sin ajuste fino; valores atípicos sin tratar. Próximos pasos: búsqueda de hiperparámetros, análisis de atípicos, nuevas variables (edad, área total) y, si se usara para tasar antes de vender, retirar las variables de la venta.
