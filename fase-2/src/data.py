"""Lectura y validación de datos de entrenamiento, separación train/test y huella del archivo."""
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import MIN_FILAS
from src.errors import DatosInvalidos


def leer_datos(ruta, target, id_columns):
    """Lee el CSV etiquetado y valida que exista la variable objetivo y haya predictoras.

    Devuelve (DataFrame completo, lista de columnas predictoras). Las predictoras excluyen
    el objetivo y los identificadores.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise DatosInvalidos(f"No existe el archivo de datos: {ruta}")
    try:
        df = pd.read_csv(ruta)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as e:
        raise DatosInvalidos(f"No se pudo leer {ruta} como CSV: {e}") from e

    if target not in df.columns:
        raise DatosInvalidos(
            f"La variable objetivo '{target}' no está en {ruta}. "
            f"Columnas disponibles: {', '.join(map(str, df.columns[:15]))}..."
        )
    if len(df) < MIN_FILAS:
        raise DatosInvalidos(f"Se necesitan al menos {MIN_FILAS} filas; el archivo tiene {len(df)}.")
    objetivo = pd.to_numeric(df[target], errors="coerce")
    if objetivo.isna().any():
        raise DatosInvalidos(
            f"La variable objetivo '{target}' tiene {int(objetivo.isna().sum())} valores vacíos "
            "o no numéricos; no se descartan filas silenciosamente."
        )
    if (objetivo < 0).any():
        raise DatosInvalidos(f"La variable objetivo '{target}' tiene valores negativos.")

    predictoras = [c for c in df.columns if c != target and c not in id_columns]
    if not predictoras:
        raise DatosInvalidos("No quedan variables predictoras tras excluir objetivo e identificadores.")
    return df, predictoras


def separar(df, predictoras, target, test_size, seed):
    """Partición aleatoria train/test. Debe hacerse ANTES de ajustar cualquier transformación."""
    if not 0 < test_size < 1:
        raise DatosInvalidos(f"--test-size debe estar entre 0 y 1 (recibido {test_size}).")
    return train_test_split(df[predictoras], df[target], test_size=test_size, random_state=seed)


def huella_archivo(ruta):
    """SHA-256 del archivo de datos: identifica exactamente qué datos se usaron."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()