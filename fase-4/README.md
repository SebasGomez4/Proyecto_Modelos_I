# API de precios de viviendas con monitoreo – Fase 4

Proyecto Integrador v1 · 2026 II · Modelos y Simulación de Sistemas I · Universidad de Antioquia

## Integrantes

-Samuel Echeverri Ortiz (@Eche0813)
- Miguel Ángel Foronda (@Foronda713)
- Sebastian Gómez Quintero (@SebasGomez4)
## Relación con la Fase 3

Se conserva `../fase-3/`. Esta carpeta parte de la misma API (`/health`, `/modelo`, `/predict`, `/train`, que reutiliza `../fase-2/src`) y agrega lo necesario para comprobar cómo se usa:

- **Registro de solicitudes** (`registro.py`): cada llamada a `POST /predict` agrega una fila a un archivo CSV, **también las rechazadas por validación**.
- **Script de monitoreo** (`monitoreo.py`): lee el registro y muestra el total de solicitudes y el porcentaje de fallos.
- **Reporte, versiones y política de reentrenamiento** (este README).

## Estructura

```
fase-4/
├── api.py                  # API con el registro de solicitudes (middleware)
├── registro.py             # escribe el CSV del registro
├── monitoreo.py            # lee el registro y calcula los indicadores
├── ajustes.py, entrada.py, entrenamiento.py, servicio_modelo.py, rutas.py   # igual que en la fase 3 (más LOG_PATH)
├── cliente_prueba.py       # cliente para enviar solicitudes de ensayo
├── muestra_registro.csv    # registro del ensayo que respalda el reporte
├── tests/                  # pruebas con pytest
├── Dockerfile, Dockerfile.dockerignore
└── evidencia/ensayo-docker.md
```

## Cómo arrancar la API con el registro habilitado

El registro está **siempre habilitado**; la versión del modelo se identifica con `MODEL_VERSION`. Desde `fase-4`, con el entorno de la fase 3 (`pip install -r requirements.txt`). Bash:

```bash
MODEL_VERSION=v1 uvicorn api:app --host 0.0.0.0 --port 8000
```

PowerShell:

```powershell
$env:MODEL_VERSION = "v1"; uvicorn api:app --host 0.0.0.0 --port 8000
```

Con Docker (terminal usada: **<terminal>**), desde la raíz del repositorio. Bash:

```bash
docker build -f fase-4/Dockerfile -t modelos-api-monitoreo:1.0 .
mkdir -p fase-4/logs
docker run --rm -p 8000:8000 -e MODEL_VERSION=v1 -v "$PWD/fase-2/data:/app/data:ro" -v "$PWD/fase-2/models:/app/models" -v "$PWD/fase-4/logs:/app/logs" modelos-api-monitoreo:1.0
```

PowerShell:

```powershell
docker build -f fase-4/Dockerfile -t modelos-api-monitoreo:1.0 .
mkdir fase-4/logs -Force
docker run --rm -p 8000:8000 -e MODEL_VERSION=v1 -v "${PWD}/fase-2/data:/app/data:ro" -v "${PWD}/fase-2/models:/app/models" -v "${PWD}/fase-4/logs:/app/logs" modelos-api-monitoreo:1.0
```

**Ubicación del registro persistente:** `fase-4/logs/predicciones.csv` en el equipo anfitrión (dentro del contenedor, `/app/logs/predicciones.csv`, montado desde `fase-4/logs`). Se conserva aunque el contenedor se elimine y se **agrega** en cada ejecución: no se borra. Su ruta se cambia con la variable `LOG_PATH`.

## Formato del registro

Un CSV con encabezado, creado con la primera llamada, y una fila por llamada a `POST /predict`:

| Columna | Contenido |
|---|---|
| `fecha_hora` | hora UTC, `AAAA-MM-DDTHH:MM:SS` |
| `estado_http` | código HTTP devuelto |
| `version` | versión del modelo que validó o predijo; vacía si no se llegó a usar un modelo (JSON inválido, versión no disponible o método incorrecto) |
| `prediccion` | precio estimado en USD; vacío en las respuestas de error |

Una entrada rechazada por validación (422) queda registrada con la versión cuyo esquema la validó y la predicción vacía. Cada intento genera una sola fila. Si el archivo no se puede escribir, la API avisa por `stderr` y sigue respondiendo.

## Monitoreo

```bash
python monitoreo.py                                   # lee logs/predicciones.csv
python monitoreo.py --registro logs/predicciones.csv --detalle
```

Muestra el **total de solicitudes** y el **porcentaje de fallos** (respuestas HTTP 4xx o 5xx sobre el total, con una cifra decimal y coma). Un registro vacío da total 0 y «sin solicitudes». `--detalle` agrega el conteo por código y por versión. Un registro inexistente o con filas corruptas termina con un error claro y código 1. Estos indicadores describen el **funcionamiento del servicio**; medir la calidad predictiva requeriría conocer los precios reales de las viviendas.

## Reporte

Con la API corriendo en Docker con la versión `v1` y **sobre un registro vacío**, se enviaron tres solicitudes a `POST /predict` con `cliente_prueba.py`: dos con el registro válido `ejemplos/registro_valido.json` y una con ese mismo registro **sin la variable `Lot Area`**. Las filas guardadas (`muestra_registro.csv`) y los indicadores que calcula `monitoreo.py` son:

<!-- REPORTE-MONITOREO:INICIO -->

| # | Fecha y hora (UTC) | Código HTTP | Versión | Predicción (USD) |
|---|---|---|---|---|
| 1 | 2026-11-21T19:38:25 | 200 | v1 | 168104.09 |
| 2 | 2026-11-21T19:42:25 | 200 | v1 | 168104.09 |
| 3 | 2026-11-21T19:42:27 | 422 | v1 | — |

- **Total de solicitudes:** 3
- **Solicitudes fallidas (HTTP 4xx o 5xx):** 1
- **Porcentaje de fallos:** 33,3 %

<!-- REPORTE-MONITOREO:FIN -->

**Interpretación.** La API procesó dos registros con `v1` y rechazó el tercero (HTTP 422) porque faltaba `Lot Area`. La tasa de fallos fue 1/3 (33,3 %). El rechazo era el resultado esperado de la prueba de validación. La salida del ensayo está en [`evidencia/ensayo-docker.md`](evidencia/ensayo-docker.md).