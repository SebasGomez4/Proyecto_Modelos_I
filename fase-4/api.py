"""API REST del modelo de precios de viviendas con registro de solicitudes (Fase 4).

Arranque local (desde fase-4):
    uvicorn api:app --host 0.0.0.0 --port 8000

Rutas:
    GET /health   la API está en ejecución (siempre 200)
    GET /modelo   datos de la versión configurada (503 si no está disponible)
    POST /predict predice con la versión configurada (400 JSON inválido, 422 entrada inválida, 503 sin modelo)
    POST /train   entrena y guarda una versión nueva (201; 409 si el identificador existe o hay otro entrenamiento)

Cada llamada a /predict (también las rechazadas) agrega una fila a LOG_PATH (ver registro.py).
La versión que usa la API se elige con la variable de entorno MODEL_VERSION (ver ajustes.py).
"""
import json

import rutas  # noqa: F401  (hace importable fase-2/src)

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ajustes import Ajustes
import entrada
from entrada import EntradaInvalidaDetallada, JsonInvalido
from entrenamiento import EntrenamientoEnCurso, ServicioEntrenamiento
from registro import RegistroSolicitudes
from servicio_modelo import ServicioModelo
from src import inference
from src.errors import (DatosInvalidos, EntradaInvalida, ErrorAplicacion, ErrorDePrediccion, ModeloIlegible,
                        ModeloNoEncontrado, VersionExistente, VersionInvalida)

# error de la aplicación -> código HTTP (el orden importa: se usa el primero que coincida)
ESTADO_POR_ERROR = [
    (ModeloNoEncontrado, 503),
    (ModeloIlegible, 503),
    (JsonInvalido, 400),            # el cuerpo ni siquiera es JSON
    (EntradaInvalida, 422),         # falta una variable o llega un tipo incompatible
    (VersionInvalida, 422),         # /train: identificador con caracteres no permitidos
    (VersionExistente, 409),        # /train: no se sobrescribe una versión existente
    (EntrenamientoEnCurso, 409),    # /train: ya hay un entrenamiento en ejecución
    (DatosInvalidos, 500),          # /train: los datos configurados no existen o no son válidos
    (ErrorDePrediccion, 500),
]


def codigo_http(error):
    for tipo, codigo in ESTADO_POR_ERROR:
        if isinstance(error, tipo):
            return codigo
    return 500


def predecir_registro(servicio, registro, traza=None):
    """Mismo camino que predict.py (Fase 2): esquema guardado -> preprocesamiento guardado -> modelo.

    `traza` recoge lo que el registro de solicitudes necesita saber aunque la llamada falle después:
    la versión que se usó y, si hubo éxito, la predicción.
    """
    traza = {} if traza is None else traza
    modelo, meta = servicio.obtener()                                  # 503 si la versión no está disponible
    traza["version"] = meta["version"]                                 # ya se usó el esquema de esta versión
    df = entrada.validar_registro(registro, meta["esquema_entrada"])   # 422 si falta o no coincide el tipo
    X, _ = inference.preparar_entrada(df, meta["esquema_entrada"])     # orden y tipos del entrenamiento
    pred = inference.predecir(modelo, X)
    resultado = {"prediccion": round(float(pred[0]), 2), "version": meta["version"]}
    traza["prediccion"] = resultado["prediccion"]
    return resultado


def create_app(ajustes=None):
    """Fábrica de la aplicación: las pruebas pasan Ajustes propios (modelo pequeño, carpeta temporal)."""
    ajustes = ajustes or Ajustes.desde_entorno()
    servicio = ServicioModelo(ajustes)
    entrenador = ServicioEntrenamiento(ajustes, servicio)
    registro_solicitudes = RegistroSolicitudes(ajustes.log_path)

    app = FastAPI(title="API de precios de viviendas", version="4.0",
                  description="Consulta el modelo guardado (Fase 2), entrena nuevas versiones y registra el uso.")
    app.state.ajustes = ajustes
    app.state.servicio = servicio
    app.state.entrenador = entrenador
    app.state.registro = registro_solicitudes

    @app.middleware("http")
    async def registrar_solicitudes_predict(request: Request, call_next):
        """Una fila por cada llamada a /predict, con el código HTTP que finalmente se devolvió."""
        if request.url.path != "/predict":
            return await call_next(request)
        request.state.traza = {}                                       # la ruta la completa (versión, predicción)
        codigo = 500                                                   # si algo escapa sin respuesta, es un 500
        try:
            respuesta = await call_next(request)
            codigo = respuesta.status_code
            return respuesta
        finally:
            traza = request.state.traza
            registro_solicitudes.registrar(codigo, traza.get("version"), traza.get("prediccion"))

    @app.exception_handler(ErrorAplicacion)
    async def errores_de_la_aplicacion(request: Request, error: ErrorAplicacion):
        codigo = codigo_http(error)
        cuerpo = {"detail": str(error)}
        if isinstance(error, EntradaInvalidaDetallada):
            cuerpo["errores"] = error.errores                      # un elemento por campo con problema
        if codigo == 503:
            cuerpo.update(
                version_configurada=servicio.version,
                versiones_disponibles=servicio.versiones_disponibles(),
                ayuda="Seleccione una versión disponible con la variable de entorno MODEL_VERSION "
                      "y reinicie la API, o entrene una nueva con POST /train.")
        return JSONResponse(cuerpo, status_code=codigo)

    @app.get("/health")
    def health():
        return {"estado": "ok", "mensaje": "API en ejecución",
                "modelo": {"version": servicio.version, "disponible": servicio.disponible()},
                "versiones_disponibles": servicio.versiones_disponibles()}

    @app.get("/modelo")
    def modelo():
        _, meta = servicio.obtener()
        return {"version": meta["version"], "fecha_creacion": meta["fecha_creacion"],
                "algoritmo": meta["algoritmo"], "variable_objetivo": meta["variable_objetivo"],
                "n_variables_entrada": len(meta["esquema_entrada"]), "semilla": meta["semilla"],
                "commit": meta["commit"]}

    @app.post("/predict")
    async def predict(request: Request):
        try:
            registro = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise JsonInvalido("El cuerpo de la solicitud no es un JSON válido.") from None
        return await run_in_threadpool(predecir_registro, servicio, registro, request.state.traza)

    @app.post("/train", status_code=201)
    async def train(request: Request):
        crudo = await request.body()
        try:
            cuerpo = json.loads(crudo) if crudo.strip() else {}              # cuerpo vacío = configuración por defecto
        except (ValueError, UnicodeDecodeError):
            raise JsonInvalido("El cuerpo de /train no es un JSON válido (envíe un cuerpo vacío o un objeto JSON).") from None
        return await run_in_threadpool(entrenador.entrenar, cuerpo)

    return app


app = create_app()