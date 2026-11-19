"""Convierte una fila del CSV de ejemplo de la Fase 2 en el JSON que espera POST /predict.

Ejemplo (desde fase-3):
    python ejemplos/generar_registro.py --fila 0 --salida ejemplos/registro_valido.json

Reglas de conversión: cada columna predictora pasa a ser una clave JSON con su nombre original;
los números se envían como números, las categorías como texto y las celdas vacías como null.
Se omiten los identificadores (Order, PID, Id): el modelo no los usa.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CSV_POR_DEFECTO = RAIZ.parent / "fase-2" / "data" / "ejemplo_prediccion.csv"
IDENTIFICADORES = ["Order", "PID", "Id"]


def fila_a_registro(fila):
    """Serie de pandas -> diccionario con tipos JSON nativos (NaN -> None)."""
    registro = {}
    for nombre, valor in fila.items():
        if nombre in IDENTIFICADORES:
            continue
        if pd.isna(valor):
            registro[nombre] = None
        elif hasattr(valor, "item"):                 # tipos de numpy -> int / float / str de Python
            registro[nombre] = valor.item()
        else:
            registro[nombre] = valor
    return registro


def main(argv=None):
    p = argparse.ArgumentParser(description="Convierte una fila del CSV de ejemplo en un registro JSON.")
    p.add_argument("--csv", default=str(CSV_POR_DEFECTO), help="CSV de ejemplo de la fase 2")
    p.add_argument("--fila", type=int, default=0, help="posición de la fila (desde 0)")
    p.add_argument("--salida", help="archivo JSON de salida (si se omite, imprime por pantalla)")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    if not 0 <= args.fila < len(df):
        print(f"Error: --fila debe estar entre 0 y {len(df) - 1}.", file=sys.stderr)
        return 1
    texto = json.dumps(fila_a_registro(df.iloc[args.fila]), indent=2, ensure_ascii=False)
    if args.salida:
        Path(args.salida).write_text(texto + "\n", encoding="utf-8")
        print(f"Registro guardado en {args.salida}")
    else:
        print(texto)
    return 0


if __name__ == "__main__":
    sys.exit(main())