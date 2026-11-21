"""Monitoreo de operación (Fase 4): lee el registro de /predict y muestra dos indicadores.

Uso (desde fase-4):
    python monitoreo.py                                   # lee logs/predicciones.csv
    python monitoreo.py --registro logs/predicciones.csv --detalle

Indicadores:
- Total de solicitudes: filas de datos del registro (una por llamada a /predict, rechazadas incluidas).
- Porcentaje de fallos: solicitudes con código HTTP 4xx o 5xx sobre el total, con una cifra decimal
  y coma (p. ej. 33,3 %). Si el registro está vacío: total 0 y «sin solicitudes».

Describe cómo FUNCIONA el servicio, no qué tan acertadas son las predicciones: medir la calidad
predictiva requeriría conocer los precios reales de las viviendas.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from registro import CAMPOS

RUTA_POR_DEFECTO = "logs/predicciones.csv"


class ErrorRegistro(Exception):
    """El registro no existe o su contenido no tiene el formato esperado."""


def leer_registro(ruta):
    """Filas de datos del registro como diccionarios (estado_http ya convertido a entero).

    Un archivo vacío o con solo el encabezado devuelve []. Una fila con un código que no es un
    número no se ignora en silencio: se informa con su número de línea.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ErrorRegistro(f"No existe el registro {ruta}. El archivo se crea con la primera llamada a "
                            "/predict; si ya hubo llamadas, revise la ruta (--registro).")
    filas = []
    try:
        with open(ruta, encoding="utf-8-sig", newline="") as f:
            lector = csv.DictReader(f)
            if lector.fieldnames is None:                       # archivo de 0 bytes
                return []
            faltan = [c for c in CAMPOS if c not in lector.fieldnames]
            if faltan:
                raise ErrorRegistro(f"El encabezado de {ruta} no tiene las columnas {', '.join(faltan)}; "
                                    f"se esperaba: {','.join(CAMPOS)}.")
            for fila in lector:
                texto = (fila["estado_http"] or "").strip()
                try:
                    fila["estado_http"] = int(texto)
                except ValueError:
                    raise ErrorRegistro(f"Línea {lector.line_num} del registro: estado_http inválido "
                                        f"({texto!r}); se esperaba un código HTTP numérico.") from None
                filas.append(fila)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        raise ErrorRegistro(f"No se pudo leer {ruta}: {e}") from e
    return filas