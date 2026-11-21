"""Validación del registro JSON de POST /predict contra el esquema guardado con la versión.

Reglas documentadas:
- El cuerpo es UN objeto JSON con una clave por variable predictora (los nombres de las columnas del CSV).
- Todas las variables del esquema son obligatorias; faltar una es un error (422) que la nombra.
- Un valor null es válido en cualquier variable: el preprocesamiento guardado en la versión aplica
  la regla de la Fase 2 (mediana, "None" o "Desconocido"). Un null no equivale a una clave ausente.
- Tipos: las variables numéricas se envían como número y las categóricas como texto.
- Claves adicionales (p. ej. Order, PID, Id o la variable objetivo) se ignoran: la predicción no
  necesita el precio.
"""
import math

import numpy as np
import pandas as pd

import rutas  # noqa: F401
from src.errors import EntradaInvalida, ErrorAplicacion


class JsonInvalido(ErrorAplicacion):
    """El cuerpo de la solicitud no es JSON (400)."""


class EntradaInvalidaDetallada(EntradaInvalida):
    """Registro que no cumple el esquema; errores detalla cada campo (422)."""

    def __init__(self, mensaje, errores):
        super().__init__(mensaje)
        self.errores = errores


def _es_numero(valor):
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def _describir(valor):
    if isinstance(valor, bool):
        return f"un booleano ({valor})"
    if isinstance(valor, str):
        return f"texto {valor!r}"
    if isinstance(valor, list):
        return "una lista"
    if isinstance(valor, dict):
        return "un objeto"
    return f"un valor de tipo {type(valor).__name__}"


def validar_registro(registro, esquema):
    """Valida el registro y devuelve un DataFrame de una fila con las columnas en el orden del esquema."""
    if not isinstance(registro, dict):
        raise EntradaInvalida("El cuerpo debe ser un objeto JSON con una clave por variable predictora "
                              "(un solo registro, no una lista).")
    errores, faltan, tipos = [], [], []
    for col in esquema:
        nombre = col["nombre"]
        if nombre not in registro:
            faltan.append(nombre)
            errores.append({"campo": nombre, "problema": "falta",
                            "mensaje": f"Falta el campo obligatorio '{nombre}'."})
            continue
        valor = registro[nombre]
        if valor is None:
            continue                                                   # nulo permitido: regla de la Fase 2
        problema = None
        if col["tipo"] == "numerica":
            if not _es_numero(valor) or (isinstance(valor, float) and math.isinf(valor)):
                problema = f"'{nombre}' debe ser numérico (recibido {_describir(valor)})"
        elif not isinstance(valor, str):
            problema = f"'{nombre}' debe ser texto (recibido {_describir(valor)})"
        if problema:
            tipos.append(problema)
            errores.append({"campo": nombre, "problema": "tipo", "mensaje": problema + "."})

    if errores:
        partes = []
        if faltan:
            partes.append((f"falta 1 campo obligatorio: {faltan[0]}" if len(faltan) == 1 else
                           f"faltan {len(faltan)} campos obligatorios: {', '.join(faltan)}"))
        if tipos:
            partes.append("tipos incompatibles: " + "; ".join(tipos))
        raise EntradaInvalidaDetallada("Entrada inválida: " + " | ".join(partes) + ".", errores)

    fila = {}
    for col in esquema:
        valor = registro[col["nombre"]]
        if valor is None or (col["tipo"] == "numerica" and isinstance(valor, float) and math.isnan(valor)):
            valor = np.nan
        fila[col["nombre"]] = valor
    return pd.DataFrame([fila], columns=[c["nombre"] for c in esquema])