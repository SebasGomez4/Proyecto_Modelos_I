"""Construcción de los metadatos de una versión (metadata.json).

train.py llama a crear_metadata(...) y guarda el resultado con registry.guardar_version.
predict.py usa el esquema guardado ("esquema_entrada") para validar los datos nuevos.
"""
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn


def esquema_entrada(X):
    """Lista ordenada de {'nombre', 'tipo'} con tipo 'numerica' o 'categorica'."""
    return [
        {"nombre": c, "tipo": "numerica" if pd.api.types.is_numeric_dtype(X[c]) else "categorica"}
        for c in X.columns
    ]


def obtener_commit():
    """Hash del commit del código. Dentro de Docker no hay .git: usar la variable GIT_COMMIT."""
    desde_entorno = os.environ.get("GIT_COMMIT", "").strip()
    if desde_entorno:
        return desde_entorno
    raiz = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True,
                                text=True, check=True, timeout=5).stdout.strip()
        cambios = subprocess.run(["git", "status", "--porcelain"], cwd=raiz, capture_output=True,
                                 text=True, check=True, timeout=5).stdout.strip()
        return commit + ("+cambios-sin-commit" if cambios else "")
    except (OSError, subprocess.SubprocessError):
        return "desconocido"


def entorno():
    return {"python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "dependencias": "requirements.txt"}


def crear_metadata(version, *, objetivo, algoritmo, parametros, semilla, esquema, datos,
                   separacion, metrica, tipo_problema="regresión", identificadores=()):
    """Reúne los metadatos mínimos exigidos para una versión.

    datos:       {"fuente": ..., "archivo": ..., "n_observaciones": ...}
    separacion:  {"estrategia": ..., "test_size": ..., "n_train": ..., "n_test": ...}
    esquema:     resultado de esquema_entrada(X) (orden y tipos de las variables de entrada)
    """
    return {
        "version": version,
        "fecha_creacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo_problema": tipo_problema,
        "variable_objetivo": objetivo,
        "algoritmo": algoritmo,
        "parametros": parametros,
        "semilla": semilla,
        "esquema_entrada": esquema,
        "columnas_identificador": list(identificadores),
        "datos": datos,
        "separacion": separacion,
        "metrica_principal": metrica,
        "entorno": entorno(),
        "commit": obtener_commit(),
    }