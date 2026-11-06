# Predicción del precio de viviendas – Fase 2: scripts, versiones del modelo y Docker

Proyecto Integrador v1 · 2026 II · Modelos y Simulación de Sistemas I · Universidad de Antioquia

## Integrantes

-Samuel Echeverri Ortiz (@Eche0813)

-Miguel Angel Foronda (@Foronda713)

-Sebastian Gómez Quintero (@SebasGomez4)

## Problema y relación con la Fase 1

Estimar el precio de venta (`SalePrice`, USD) de una vivienda a partir de sus características: es un problema de **regresión** sobre el *Ames Housing Dataset* (Kaggle `aecincode_houseprices`). La Fase 1 (`../fase-1/`, se conserva sin cambios) desarrolló el modelo en un notebook. Esta fase lo convierte en una aplicación reutilizable:

- `train.py`: entrena y guarda una **versión** identificable del modelo (modelo + preprocesamiento + métricas + metadatos).
- `predict.py`: carga una versión guardada y predice sobre datos nuevos, sin reentrenar ni reajustar nada.
- `Dockerfile`: ejecuta ambos scripts en un contenedor, con datos y resultados en carpetas montadas.
- `tests/`: 17 pruebas con pytest.

Se conservan el problema, los datos y el enfoque de la Fase 1: Gradient Boosting (`HistGradientBoostingRegressor`) entrenado sobre `log(1 + precio)`, partición aleatoria 80/20 con semilla 42 y modelo base que predice la mediana.

## Correcciones respecto a la Fase 1

| Retroalimentación | Qué se hizo |
|---|---|
| Separar ausencia estructural de información desconocida | `AusenciaEstructural` (`src/preprocessing.py`): un vacío es `"None"` solo si la variable de referencia vale 0 (p. ej. `Bsmt Qual` vacío y `Total Bsmt SF == 0`); en otro caso es `"Desconocido"`. En `train.csv`: `Bsmt Qual` 41 "None" + 1 "Desconocido"; `Garage Finish` 98 + 1; `Bsmt Exposure` 41 + 3; `Mas Vnr Type` 1.037 + 18. El vacío extra es una fila donde la propia variable de referencia también falta. La regla usa solo valores de la misma fila, así que no aprende nada de los datos (no hay fuga). `Alley`, `Fence` y `Misc Feature` no tienen variable de referencia: su vacío se trata como "None" (limitación). |
| Precisar el alcance de generalización | Se declara en `metadata.json` y aquí: estimación para viviendas del mismo mercado (Ames, 2006-2010) y **barrios conocidos**, con partición aleatoria. No se evaluó en barrios nuevos ni en períodos futuros. |
| Disponibilidad de las variables de la transacción | Se conservan `Sale Type`, `Sale Condition`, `Mo Sold` y `Yr Sold`, pero solo son válidas si se conocen al predecir; para una tasación previa a la venta habría que retirarlas. Queda registrado en `limitaciones` de cada `metadata.json`. |
| Moderar conclusiones | Que CV y prueba sean cercanos es consistente con un modelo estable, pero no demuestra por sí solo ausencia de sobreajuste o fuga. El RMSLE es un error en escala logarítmica, no un porcentaje medio de error. |
| Fijar versiones del entorno | `requirements.txt` con versiones exactas; Python 3.12 en local y en Docker (`python:3.12-slim`). |
| Pruebas de recarga, faltantes y categorías nuevas | `tests/test_aplicacion.py` (ver sección de pruebas). |
| Revisión de los PR de cada funcionalidad | Cada rama `feature/*` se integró a `develop` mediante un Pull Request revisado por otro integrante. |

**Efecto en el modelo.** Con la misma partición (80/20, semilla 42), el MAE en prueba pasó de 15.177 USD (notebook de la Fase 1) a 14.769 USD (`v1`). La diferencia reúne varios cambios del preprocesamiento (reglas de ausencia, categoría "Desconocido", indicadores en las variables ordinales) y no se atribuye a uno solo; mejorar la métrica no era objetivo de esta fase.

## Estructura

```
repositorio/
├── fase-1/                    # entrega anterior (sin cambios)
└── fase-2/
    ├── train.py               # entrena y guarda una versión
    ├── predict.py             # predice con una versión guardada
    ├── requirements.txt       # dependencias con versiones exactas (incluye pytest)
    ├── Dockerfile
    ├── .dockerignore
    ├── pytest.ini
    ├── src/                   # lógica reutilizable (config, data, preprocessing, modeling,
    │                          #   training, registry, metadata, inference, errors)
    ├── tests/test_aplicacion.py
    ├── data/
    │   ├── train.csv              # datos etiquetados
    │   └── ejemplo_prediccion.csv # 5 filas sin SalePrice, con el esquema real
    ├── models/
    │   ├── v1/                # modelo.joblib, metrics.json, metadata.json
    │   └── v2/
    ├── evidencia/docker.md    # salida real de la ejecución en Docker
    └── outputs/               # resultados generados (no se versiona)
```