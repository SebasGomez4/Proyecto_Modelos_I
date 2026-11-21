"""Lógica de POST /train: valida los parámetros, genera un identificador libre y reutiliza
src.training.entrenar (la misma función que usa train.py en la Fase 2).

Cuerpo de la solicitud (opcional): un objeto JSON con cualquiera de estas claves; lo no indicado
toma el valor de la configuración (ajustes.py: SEED, MAX_ITER, LEARNING_RATE, CV_FOLDS).

    version        texto   identificador de la nueva versión (si se omite, se genera api-AAAAMMDDTHHMMSSZ)
    seed           entero  semilla de la partición, la validación cruzada y el modelo
    max_iter       entero  iteraciones de Gradient Boosting (1 a 2000)
    learning_rate  número  tasa de aprendizaje (mayor que 0 y hasta 1)
    cv_folds       entero  folds de validación cruzada: 0 la omite, o entre 2 y 10

Una versión nueva nunca sobrescribe una existente y no cambia la versión activa de /predict.
"""
import threading
from datetime import datetime, timezone
from pathlib import Path

import rutas  # noqa: F401
from src import training
from src.errors import EntradaInvalida, ErrorAplicacion


def _entero(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _numero(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# clave -> (comprobación de tipo, descripción del tipo, comprobación de rango, descripción del rango)
REGLAS = {
    "version": (lambda v: isinstance(v, str), "texto", lambda v: True, ""),
    "seed": (_entero, "entero", lambda v: True, ""),
    "max_iter": (_entero, "entero", lambda v: 1 <= v <= 2000, "entre 1 y 2000"),
    "learning_rate": (_numero, "número", lambda v: 0 < v <= 1, "mayor que 0 y hasta 1"),
    "cv_folds": (_entero, "entero", lambda v: v == 0 or 2 <= v <= 10, "0 (sin validación cruzada) o entre 2 y 10"),
}


def leer_parametros(cuerpo, ajustes):
    """Combina el cuerpo (opcional) con la configuración. Lanza EntradaInvalida con todos los problemas."""
    cuerpo = {} if cuerpo is None else cuerpo
    if not isinstance(cuerpo, dict):
        raise EntradaInvalida("El cuerpo de /train debe estar vacío o ser un objeto JSON "
                              f"con claves entre: {', '.join(REGLAS)}.")
    problemas = [f"clave desconocida '{k}' (permitidas: {', '.join(REGLAS)})" for k in cuerpo if k not in REGLAS]
    for clave, (es_tipo, tipo, en_rango, rango) in REGLAS.items():
        if clave not in cuerpo or cuerpo[clave] is None:
            continue
        valor = cuerpo[clave]
        if not es_tipo(valor):
            problemas.append(f"'{clave}' debe ser {tipo}")
        elif not en_rango(valor):
            problemas.append(f"'{clave}' debe estar {rango}")
    if problemas:
        raise EntradaInvalida("Parámetros de /train inválidos: " + "; ".join(problemas) + ".")

    def valor(clave, por_defecto):
        return por_defecto if cuerpo.get(clave) is None else cuerpo[clave]

    return {"version": cuerpo.get("version"),
            "seed": valor("seed", ajustes.seed),
            "max_iter": valor("max_iter", ajustes.max_iter),
            "learning_rate": float(valor("learning_rate", ajustes.learning_rate)),
            "cv_folds": valor("cv_folds", ajustes.cv_folds)}


def generar_identificador(models_dir, ahora=None):
    """api-AAAAMMDDTHHMMSSZ; si ya existe (dos entrenamientos en el mismo segundo), agrega -2, -3..."""
    ahora = ahora or datetime.now(timezone.utc)
    base = f"api-{ahora:%Y%m%dT%H%M%SZ}"
    candidato, n = base, 1
    while (Path(models_dir) / candidato).exists():
        n += 1
        candidato = f"{base}-{n}"
    return candidato


class EntrenamientoEnCurso(ErrorAplicacion):
    """Ya hay un entrenamiento en ejecución (409): se entrena de a uno para no competir por los archivos."""


class ServicioEntrenamiento:
    def __init__(self, ajustes, servicio_modelo):
        self.ajustes = ajustes
        self.servicio_modelo = servicio_modelo
        self._lock = threading.Lock()

    def entrenar(self, cuerpo):
        """Entrena y guarda una versión nueva. Devuelve el resumen que responde POST /train."""
        parametros = leer_parametros(cuerpo, self.ajustes)
        if not self._lock.acquire(blocking=False):
            raise EntrenamientoEnCurso("Ya hay un entrenamiento en curso; espere a que termine y reintente.")
        try:
            models_dir = self.ajustes.models_dir
            version = parametros["version"] or generar_identificador(models_dir)
            r = training.entrenar(self.ajustes.data_path, self.ajustes.target, models_dir, version,
                                  seed=parametros["seed"], cv_folds=parametros["cv_folds"],
                                  max_iter=parametros["max_iter"], learning_rate=parametros["learning_rate"])
        finally:
            self._lock.release()

        m = r["metricas"]
        activa = self.servicio_modelo.version
        return {
            "version": version,
            "estado": "guardada",
            "ubicacion": f"{models_dir.name}/{version}",
            "archivos": ["modelo.joblib", "metrics.json", "metadata.json"],
            "parametros": {k: parametros[k] for k in ("seed", "max_iter", "learning_rate", "cv_folds")},
            "metricas_prueba": {k: round(v, 4) for k, v in m["modelo"]["test"].items()},
            "metricas_modelo_base_prueba": {k: round(v, 4) for k, v in m["baseline"]["test"].items()},
            "version_activa": activa,
            "nota": f"/predict sigue usando '{activa}'. Para usar '{version}' inicie la API con "
                    f"MODEL_VERSION={version}.",
        }