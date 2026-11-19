"""API REST del modelo de precios de viviendas (Fase 3).

Arranque local (desde fase-3):
    uvicorn api:app --host 0.0.0.0 --port 8000

Rutas (se van completando por Pull Requests):
    GET /health   la API está en ejecución (siempre 200)
    GET /modelo   datos de la versión configurada (503 si no está disponible)

La versión que usa la API se elige con la variable de entorno MODEL_VERSION (ver ajustes.py).
"""
import rutas  # noqa: F401  (hace importable fase-2/src)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ajustes import Ajustes
from servicio_modelo import ServicioModelo
from src.errors import ErrorAplicacion, ModeloIlegible, ModeloNoEncontrado

# error de la aplicación -> código HTTP (los siguientes PR agregan entrada, entrenamiento, etc.)
ESTADO_POR_ERROR = [
    (ModeloNoEncontrado, 503),
    (ModeloIlegible, 503),
]


def codigo_http(error):
    for tipo, codigo in ESTADO_POR_ERROR:
        if isinstance(error, tipo):
            return codigo
    return 500


def create_app(ajustes=None):
    """Fábrica de la aplicación: las pruebas pasan Ajustes propios (modelo pequeño, carpeta temporal)."""
    ajustes = ajustes or Ajustes.desde_entorno()
    servicio = ServicioModelo(ajustes)

    app = FastAPI(title="API de precios de viviendas", version="3.0",
                  description="Consulta el modelo guardado (Fase 2) y entrena nuevas versiones.")
    app.state.ajustes = ajustes
    app.state.servicio = servicio

    @app.exception_handler(ErrorAplicacion)
    async def errores_de_la_aplicacion(request: Request, error: ErrorAplicacion):
        codigo = codigo_http(error)
        cuerpo = {"detail": str(error)}
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

    return app


app = create_app()