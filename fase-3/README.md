# API REST de precios de viviendas – Fase 3

Proyecto Integrador v1 · 2026 II · Modelos y Simulación de Sistemas I · Universidad de Antioquia

## Integrantes

-Samuel Echeverri Ortiz (@Eche0813)
-Miguel Angel Foronda (@Foronda713)
-Sebastian Gómez Quintero (@SebasGomez4)

## Descripción y relación con las fases anteriores

API REST (FastAPI) que consulta el modelo de precios de viviendas de la Fase 2 y entrena nuevas versiones mediante solicitudes HTTP. **Reutiliza** el código de `../fase-2/src` (lectura de versiones, esquema de entrada, preprocesamiento guardado, predicción y entrenamiento) sin copiarlo: las rutas solo traducen JSON a la misma función que usa `predict.py` y a la misma que usa `train.py`. Se conservan `fase-1/` y `fase-2/`.

## Estructura

```
fase-3/
├── api.py                  # aplicación FastAPI: /health, /modelo, /predict, /train
├── ajustes.py              # configuración por variables de entorno
├── servicio_modelo.py      # acceso al modelo configurado (503 si no está disponible)
├── entrada.py              # validación del registro JSON contra el esquema guardado
├── entrenamiento.py        # lógica de /train (parámetros, identificador, bloqueo)
├── rutas.py                # hace importable fase-2/src
├── cliente_prueba.py       # cliente de prueba (solo biblioteca estándar)
├── ejemplos/               # registro_valido.json y su generador desde el CSV de la fase 2
├── tests/test_api.py       # pruebas con pytest
├── requirements.txt        # dependencias con versiones exactas
├── Dockerfile
├── Dockerfile.dockerignore
└── evidencia/              # salida real de la ejecución en Docker
```

## Arranque local

Python 3.12. Desde `fase-3`:

Bash:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Por defecto usa los datos y modelos de `../fase-2` (`data/train.csv` y `models/`). La documentación interactiva queda en `http://localhost:8000/docs`.

## Rutas

| Ruta | Qué hace | Respuestas |
|---|---|---|
| `GET /health` | indica que la API está en ejecución y si el modelo configurado está disponible | 200 siempre |
| `GET /modelo` | metadatos de la versión configurada | 200; 503 si no está disponible |
| `POST /predict` | predice el precio con la versión configurada | 200; 400 JSON inválido; 422 campo ausente o tipo incompatible; 503 sin modelo |
| `POST /train` | entrena y guarda una versión nueva (no cambia la versión activa) | 201; 409 identificador repetido o entrenamiento en curso; 422 parámetros inválidos; 500 datos no disponibles |

### `POST /predict`

Recibe **un registro JSON** con las 79 variables predictoras, con los mismos nombres del CSV de la Fase 2: los números como números, las categorías como texto y los datos faltantes como `null`. No necesita el precio. Las claves adicionales (`Id`, `PID`, `SalePrice`…) se ignoran. Un `null` no equivale a una clave ausente: lo trata el preprocesamiento guardado con la versión (mediana, `"None"` o `"Desconocido"`, según la regla de la Fase 2).

`ejemplos/registro_valido.json` es la fila 0 de `../fase-2/data/ejemplo_prediccion.csv` convertida a JSON con `python ejemplos/generar_registro.py` (79 claves; tiene `null` en `Lot Frontage`, `Mas Vnr Type`, `Pool QC` y `Misc Feature`). Fragmento (las claves restantes están en el archivo):

```json
{
  "MS SubClass": 70,
  "MS Zoning": "RM",
  "Lot Frontage": null,
  "Lot Area": 5100,
  "Street": "Pave",
  "Alley": "Grvl",
  "...": "(79 claves en total)"
}
```

Solicitud (Bash; en PowerShell, `curl.exe` y `--data-binary "@ejemplos/registro_valido.json"`):

```bash
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @ejemplos/registro_valido.json
```

Respuesta (el valor exacto depende del modelo y del entorno):

```json
{"prediccion": 168104.09, "version": "v1"}
```

Con un campo ausente (`python cliente_prueba.py --sin-campo "Lot Area"`) responde **422**:

```json
{
  "detail": "Entrada inválida: falta 1 campo obligatorio: Lot Area.",
  "errores": [{"campo": "Lot Area", "problema": "falta", "mensaje": "Falta el campo obligatorio 'Lot Area'."}]
}
```

### `POST /train`

El cuerpo puede estar vacío (usa la configuración: semilla 42, 300 iteraciones, tasa 0,05, 5 folds de validación cruzada) o incluir cualquiera de estas claves: `version`, `seed`, `max_iter`, `learning_rate`, `cv_folds`. Una clave desconocida da 422. Si no se indica `version`, se genera `api-AAAAMMDDTHHMMSSZ`. Guarda `modelo.joblib`, `metrics.json` y `metadata.json` en la carpeta de modelos, **sin tocar las versiones anteriores**, y devuelve **201** (con validación cruzada tarda alrededor de 20 a 30 segundos; `"cv_folds": 0` la omite):

```bash
curl -s -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '{"max_iter": 500, "learning_rate": 0.03}'
```

```json
{
  "version": "api-20261105T153012Z",
  "estado": "guardada",
  "ubicacion": "models/api-20261105T153012Z",
  "archivos": ["modelo.joblib", "metrics.json", "metadata.json"],
  "parametros": {"seed": 42, "max_iter": 500, "learning_rate": 0.03, "cv_folds": 5},
  "metricas_prueba": {"MAE": "...", "RMSE": "...", "R2": "...", "RMSLE": "..."},
  "metricas_modelo_base_prueba": {"MAE": "...", "RMSE": "...", "R2": "...", "RMSLE": "..."},
  "version_activa": "v1",
  "nota": "/predict sigue usando 'v1'. Para usar 'api-20261105T153012Z' inicie la API con MODEL_VERSION=api-20261105T153012Z."
}
```

(Valores abreviados con `...`.) Solo corre un entrenamiento a la vez: otro `/train` simultáneo recibe 409.

## Errores

Todos los errores son JSON con la clave `detail` que explica el problema.

| Código | Cuándo |
|---|---|
| 400 | el cuerpo no es JSON |
| 422 | falta una variable, un tipo no coincide, el cuerpo no es un objeto o los parámetros de `/train` son inválidos |
| 409 | `/train` con un identificador existente (nunca se sobrescribe) o con otro entrenamiento en curso |
| 503 | la versión configurada no está disponible; la respuesta lista las versiones existentes y explica cómo elegir otra |
| 500 | los datos configurados para `/train` no existen o no son válidos |

## Cómo seleccionar una versión disponible

`/predict` usa la versión de la variable de entorno **`MODEL_VERSION`** (por defecto `v1`). Para ver cuáles existen: `GET /health` (`versiones_disponibles`). Para cambiar o volver a una anterior, se reinicia la API con otra versión. Bash:

```bash
MODEL_VERSION=v2 uvicorn api:app --port 8000
```

PowerShell:

```powershell
$env:MODEL_VERSION = "v2"; uvicorn api:app --port 8000
```

Si la versión configurada no existe, `/health` sigue respondiendo 200 (con `"disponible": false`) y `/predict` y `/modelo` responden 503 con la lista de versiones disponibles. Una versión creada con `/train` no se activa sola: hay que reiniciar con su `MODEL_VERSION`.

## Variables de entorno

| Variable | Por defecto | Significado |
|---|---|---|
| `MODEL_VERSION` | `v1` | versión que usa `/predict` |
| `MODELS_DIR` | `../fase-2/models` (o `./models` en el contenedor) | carpeta con las versiones |
| `DATA_PATH` | `../fase-2/data/train.csv` (o `./data/train.csv`) | datos para `/train` |
| `TARGET` | `SalePrice` | variable objetivo |
| `SEED`, `MAX_ITER`, `LEARNING_RATE`, `CV_FOLDS` | `42`, `300`, `0.05`, `5` | valores por defecto de `/train` |
| `GIT_COMMIT` | (vacío) | hash del commit para los metadatos de las versiones entrenadas dentro de Docker |

## Cliente de prueba

`cliente_prueba.py` (solo biblioteca estándar) consulta `/health`, envía `ejemplos/registro_valido.json` a `/predict` y muestra la respuesta. Termina con código distinto de cero si alguna respuesta no es la esperada:

```bash
python cliente_prueba.py                                     # /health y /predict con un registro real
python cliente_prueba.py --sin-campo "Lot Area"              # además, el rechazo 422
python cliente_prueba.py --entrenar --cv-folds 0             # además, /train (versión api-...)
```