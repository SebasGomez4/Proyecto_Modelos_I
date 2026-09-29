# Predicción del precio de venta de viviendas – Fase 1

Proyecto Integrador v1 · 2026 II · Modelos y Simulación de Sistemas I

## Integrantes

- Samuel Echeverri Ortiz (@Eche0813) 
- Miguel Angel Foronda (@Foronda713)
- Sebastian Gómez Quintero (@SebasGomez4)

## Descripción del problema

Estimar el precio de venta de una vivienda a partir de sus características (dimensiones, calidad de materiales y acabados, año de construcción, barrio, tipo de vivienda y condiciones de la venta). Es un problema de aprendizaje supervisado de **regresión**; la variable objetivo es `SalePrice` (USD).

## Fuente del conjunto de datos

Competencia de Kaggle [`aecincode_houseprices`](https://www.kaggle.com/competitions/aecincode_houseprices), basada en el *Ames Housing Dataset* (De Cock, 2011) y relacionada con [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/).

- `data/train.csv`: 1.758 viviendas, 79 predictoras + `SalePrice` + identificadores (`Order`, `PID`, `Id`).
- `data/test.csv`: 1.172 viviendas sin precio; se usan como "datos nuevos" para probar el modelo guardado.

## Objetivo del modelo

Dada la ficha de una vivienda, predecir su precio de venta en dólares. En esta fase el énfasis está en un proceso reproducible, documentado y libre de fuga de información, que servirá de base para las siguientes fases (scripts, API REST y monitoreo).

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
