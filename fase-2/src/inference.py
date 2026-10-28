"""Lectura, validación y predicción sobre datos nuevos (sin reentrenar ni reajustar nada).

Comportamiento documentado:
- Columnas obligatorias: todas las del esquema guardado; si falta alguna, se rechaza el archivo.
- Orden: las columnas se reordenan según el esquema; el orden del CSV no importa.
- Columnas adicionales: se ignoran (se avisa por stderr).
- Celdas nulas: son válidas; las trata el preprocesamiento guardado con el modelo.
- Categorías no vistas en el entrenamiento: no fallan; el preprocesamiento guardado las
  ignora (one-hot en ceros) o las codifica como desconocidas (ordinales).
- Tipos: una columna numérica con texto no convertible se rechaza indicando la fila.
- Se genera exactamente una predicción por fila, en el mismo orden; no se descartan filas.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.errors import EntradaInvalida, ErrorDePrediccion


def leer_csv_entrada(ruta):
    ruta = Path(ruta)
    if not ruta.is_file():
        raise EntradaInvalida(f"No existe el archivo de entrada: {ruta}")
    try:
        df = pd.read_csv(ruta)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as e:
        raise EntradaInvalida(f"No se pudo leer {ruta} como CSV: {e}") from e
    if df.empty:
        raise EntradaInvalida(f"El archivo {ruta} no tiene filas para predecir.")
    return df


def preparar_entrada(df, esquema):
    """Valida columnas y tipos y devuelve (X ordenado según el esquema, columnas ignoradas)."""
    nombres = [c["nombre"] for c in esquema]
    faltan = [n for n in nombres if n not in df.columns]
    if faltan:
        raise EntradaInvalida(
            f"Faltan {len(faltan)} columnas obligatorias en la entrada: {', '.join(faltan)}"
        )
    extra = [c for c in df.columns if c not in nombres]

    X = df[nombres].copy()
    problemas = []
    for col in esquema:
        nombre = col["nombre"]
        if col["tipo"] == "numerica":
            convertida = pd.to_numeric(X[nombre], errors="coerce")
            malos = X[nombre].notna() & convertida.isna()
            if malos.any():
                fila = int(np.flatnonzero(malos.to_numpy())[0]) + 2  # +2: encabezado y base 1
                problemas.append(
                    f"'{nombre}' debe ser numérica, pero la fila {fila} del CSV "
                    f"contiene {X.loc[malos, nombre].iloc[0]!r}"
                )
            X[nombre] = convertida
        else:
            X[nombre] = X[nombre].astype(object)  # una columna categórica vacía llega como float
    if problemas:
        raise EntradaInvalida("Tipos de datos incompatibles: " + "; ".join(problemas))
    return X, extra


def predecir(modelo, X):
    try:
        pred = np.asarray(modelo.predict(X), dtype=float).ravel()
    except Exception as e:
        raise ErrorDePrediccion(f"El modelo no pudo procesar los datos de entrada: {e}") from e
    if len(pred) != len(X) or not np.isfinite(pred).all():
        raise ErrorDePrediccion("El modelo no generó una predicción válida para todas las filas.")
    return pred


def escribir_salida(df_original, pred, identificadores, ruta):
    """Escribe fila, identificadores presentes y prediccion. Es el último paso: si algo falló
    antes, no se crea ningún archivo de salida."""
    ruta = Path(ruta)
    if ruta.is_dir():
        raise EntradaInvalida(f"La ruta de salida es una carpeta, no un archivo: {ruta}")
    salida = pd.DataFrame({"fila": np.arange(1, len(df_original) + 1)})
    for col in identificadores:
        if col in df_original.columns:
            salida[col] = df_original[col].to_numpy()
    salida["prediccion"] = np.round(pred, 2)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    try:
        salida.to_csv(temporal, index=False)
        os.replace(temporal, ruta)  # reemplazo atómico: nunca queda un archivo a medias
    finally:
        if temporal.exists():
            temporal.unlink()
    return ruta