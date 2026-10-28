"""Versionamiento de modelos en carpetas.

Estructura de una versión (una carpeta por ejecución de entrenamiento):

    models/
        v1/
            modelo.joblib    # pipeline completo: preprocesamiento + modelo
            metrics.json     # métricas del modelo y del modelo base
            metadata.json    # información de la ejecución (ver src/metadata.py)

Reglas:
- Una versión existente NUNCA se sobrescribe: se rechaza el identificador repetido.
- Si falla el guardado, no queda una versión a medias.
- Este módulo no entrena ni lee datos: solo guarda y carga artefactos.
"""
import json
import re
import shutil
from pathlib import Path

import joblib
import numpy as np

from src.errors import ModeloIlegible, ModeloNoEncontrado, VersionExistente, VersionInvalida

ARCHIVO_MODELO = "modelo.joblib"
ARCHIVO_METRICAS = "metrics.json"
ARCHIVO_METADATA = "metadata.json"

_PATRON_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validar_version(version):
    """Acepta letras, números, punto, guion y guion bajo (evita rutas como '../x')."""
    if not isinstance(version, str) or not _PATRON_VERSION.match(version):
        raise VersionInvalida(
            f"Identificador de versión inválido: {version!r}. "
            "Use letras, números, '.', '-' o '_' (por ejemplo v1, v2, v-docker)."
        )
    return version


def _a_json(obj):
    """Convierte tipos de numpy para poder escribirlos en JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"No se puede serializar {type(obj).__name__} a JSON")


def _escribir_json(ruta, contenido):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(contenido, f, indent=2, ensure_ascii=False, default=_a_json)


def guardar_version(output_dir, version, modelo, metricas, metadata):
    """Crea output_dir/<version>/ con modelo, métricas y metadatos. Devuelve la carpeta."""
    validar_version(version)
    destino = Path(output_dir) / version
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        destino.mkdir()  # falla si ya existe: así nunca se sobrescribe una versión
    except FileExistsError:
        raise VersionExistente(
            f"La versión '{version}' ya existe en {destino}. "
            "Elija otro identificador: las versiones guardadas no se sobrescriben."
        ) from None
    try:
        joblib.dump(modelo, destino / ARCHIVO_MODELO)
        _escribir_json(destino / ARCHIVO_METRICAS, metricas)
        _escribir_json(destino / ARCHIVO_METADATA, metadata)
    except Exception:
        shutil.rmtree(destino, ignore_errors=True)  # no dejar una versión a medias
        raise
    return destino


def resolver_ruta_modelo(ruta):
    """Acepta la ruta del archivo modelo.joblib o la carpeta de la versión."""
    ruta = Path(ruta)
    return ruta / ARCHIVO_MODELO if ruta.is_dir() else ruta


def cargar_version(ruta):
    """Carga (modelo, metadata) de una versión guardada. No reentrena ni reajusta nada."""
    archivo = resolver_ruta_modelo(ruta)
    if not archivo.is_file():
        raise ModeloNoEncontrado(f"No existe el modelo: {archivo}")
    archivo_meta = archivo.parent / ARCHIVO_METADATA
    if not archivo_meta.is_file():
        raise ModeloNoEncontrado(f"Falta {ARCHIVO_METADATA} junto al modelo ({archivo_meta}).")
    try:
        modelo = joblib.load(archivo)
    except Exception as e:
        raise ModeloIlegible(f"No se pudo leer el modelo {archivo}: {e}") from e
    try:
        with open(archivo_meta, encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ModeloIlegible(f"No se pudo leer {archivo_meta}: {e}") from e
    return modelo, metadata


def listar_versiones(output_dir):
    """Nombres de las versiones existentes (carpetas que contienen modelo.joblib)."""
    base = Path(output_dir)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / ARCHIVO_MODELO).is_file())