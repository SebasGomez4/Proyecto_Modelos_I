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


## Entorno

Se usó **Python 3.12** (el mismo de la imagen Docker). Desde la carpeta `fase-2`:

Bash:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea el script de activación: `Set-ExecutionPolicy -Scope Process Bypass`.

## Datos

- **Fuente:** competencia de Kaggle [`aecincode_houseprices`](https://www.kaggle.com/competitions/aecincode_houseprices), basada en el *Ames Housing Dataset* (De Cock, 2011).
- **Archivos requeridos y ubicación:** `data/train.csv` (1.758 viviendas, 83 columnas: 79 predictoras, `SalePrice` y los identificadores `Order`, `PID`, `Id`) y `data/ejemplo_prediccion.csv`. Ambos ya están en el repositorio: no hace falta descargar nada ni usar credenciales de Kaggle.
- **Ejemplo de predicción:** 5 filas de `test.csv` (sin `SalePrice`) con el esquema real.

### Esquema del CSV de predicción

Las **79 columnas predictoras** son obligatorias (el orden en el CSV no importa). `Order`, `PID` e `Id` son opcionales: si vienen, se copian a la salida para relacionar cada predicción con su fila. Una celda vacía es válida en cualquier columna; lo que no se admite es que falte la columna entera. Tipos: 36 numéricas y 43 categóricas (texto). `MS SubClass` es numérica pero codifica categorías. El esquema exacto queda en `models/<versión>/metadata.json` (`esquema_entrada`).

<details>
<summary>Lista de columnas</summary>

**Numéricas (36):** MS SubClass, Lot Frontage, Lot Area, Overall Qual, Overall Cond, Year Built, Year Remod/Add, Mas Vnr Area, BsmtFin SF 1, BsmtFin SF 2, Bsmt Unf SF, Total Bsmt SF, 1st Flr SF, 2nd Flr SF, Low Qual Fin SF, Gr Liv Area, Bsmt Full Bath, Bsmt Half Bath, Full Bath, Half Bath, Bedroom AbvGr, Kitchen AbvGr, TotRms AbvGrd, Fireplaces, Garage Yr Blt, Garage Cars, Garage Area, Wood Deck SF, Open Porch SF, Enclosed Porch, 3Ssn Porch, Screen Porch, Pool Area, Misc Val, Mo Sold, Yr Sold.

**Categóricas (43):** MS Zoning, Street, Alley, Lot Shape, Land Contour, Utilities, Lot Config, Land Slope, Neighborhood, Condition 1, Condition 2, Bldg Type, House Style, Roof Style, Roof Matl, Exterior 1st, Exterior 2nd, Mas Vnr Type, Exter Qual, Exter Cond, Foundation, Bsmt Qual, Bsmt Cond, Bsmt Exposure, BsmtFin Type 1, BsmtFin Type 2, Heating, Heating QC, Central Air, Electrical, Kitchen Qual, Functional, Fireplace Qu, Garage Type, Garage Finish, Garage Qual, Garage Cond, Paved Drive, Pool QC, Fence, Misc Feature, Sale Type, Sale Condition.

</details>

## Entrenamiento (`train.py`)

Comando completo (en una sola línea, desde `fase-2`):

```bash
python train.py --data data/train.csv --target SalePrice --output-dir models --version v3 --seed 42
```

| Argumento | Valor por defecto | Descripción |
|---|---|---|
| `--data` | (obligatorio) | CSV con datos etiquetados |
| `--target` | `SalePrice` | variable objetivo |
| `--output-dir` | `models` | carpeta donde se guardan las versiones |
| `--version` | (obligatorio) | identificador único de la versión (letras, números, `.`, `-`, `_`) |
| `--seed` | `42` | semilla de la partición, la validación cruzada y el modelo |
| `--test-size` | `0.2` | fracción de datos para prueba |
| `--cv-folds` | `5` | folds de validación cruzada sobre entrenamiento (`0` la omite) |
| `--max-iter` | `300` | iteraciones (árboles) de Gradient Boosting |
| `--learning-rate` | `0.05` | tasa de aprendizaje |
| `--id-columns` | `Order PID Id` | columnas identificadoras que se excluyen |

Proceso: valida el identificador (falla antes de entrenar si ya existe) → lee y valida los datos → **separa train/test** → ajusta preprocesamiento y modelo solo con entrenamiento (en la validación cruzada se reajustan dentro de cada fold) → evalúa modelo y modelo base con la misma partición → guarda la versión.

Las versiones entregadas se generaron así:

```bash
python train.py --data data/train.csv --target SalePrice --output-dir models --version v1 --seed 42
python train.py --data data/train.csv --target SalePrice --output-dir models --version v2 --seed 42 --max-iter 500 --learning-rate 0.03
```

**Versionamiento.** Cada ejecución crea `models/<versión>/` con `modelo.joblib` (pipeline completo: preprocesamiento + modelo), `metrics.json` (modelo y modelo base) y `metadata.json` (identificador, fecha, algoritmo, parámetros, semilla, esquema de entrada, datos con su SHA-256, partición, métrica, versiones de Python y librerías, y el hash del commit). **Una versión existente nunca se sobrescribe**: un identificador repetido termina con error. Las carpetas `models/v3/` y `models/v-docker/` están en `.gitignore` porque son las que usan los comandos de ejemplo de este README; solo `v1` y `v2` se versionan en Git.

## Predicción (`predict.py`)

```bash
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

`--model` acepta el archivo `modelo.joblib` o la carpeta de la versión (`models/v2`). Se elige la versión explícitamente en cada ejecución. La salida es un CSV con `fila` (posición en la entrada), los identificadores presentes (`Order`, `PID`, `Id`) y `prediccion` (USD). Genera una predicción por fila y conserva el orden.

**Reglas documentadas:**

- Faltan columnas obligatorias → error con la lista de columnas faltantes.
- Columna numérica con texto no convertible → error que indica columna y fila del CSV.
- Columnas adicionales → se ignoran, con un aviso por `stderr`.
- Celdas vacías → válidas: las trata el preprocesamiento guardado con la versión.
- Categorías no vistas en el entrenamiento → no fallan: se codifican con las reglas guardadas (one-hot en ceros; en las variables ordinales se tratan como desconocidas).
- Ante cualquier error: mensaje comprensible, código de salida `1` y **no se crea el archivo de salida**.

## Ayuda de los scripts

```bash
python train.py --help
python predict.py --help
```

## Ubicación de los resultados

| Qué | Dónde |
|---|---|
| Versiones del modelo | `models/<versión>/modelo.joblib` |
| Métricas (modelo y modelo base) | `models/<versión>/metrics.json` |
| Metadatos | `models/<versión>/metadata.json` |
| Predicciones | `outputs/` (se crea al ejecutar; no se versiona) |