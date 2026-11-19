"""Muestra (o actualiza en el README) la tabla de versiones generada desde los artefactos.

Ejemplos (en una sola línea, desde la carpeta fase-2):
    python resumen_versiones.py --versiones v1 v2
    python resumen_versiones.py --actualizar-readme README.md
"""
import argparse
import sys

from src import registry, resumen
from src.errors import ErrorAplicacion


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Resume las versiones guardadas del modelo.")
    p.add_argument("--models-dir", default="models", help="carpeta con las versiones")
    p.add_argument("--versiones", nargs="+", help="versiones a resumir (por defecto, las del README o todas)")
    p.add_argument("--actualizar-readme", metavar="README", help="reescribe el bloque marcado de este README")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.actualizar_readme:
            print(resumen.actualizar_readme(args.actualizar_readme, args.models_dir, args.versiones))
            print(f"\nREADME actualizado: {args.actualizar_readme}", file=sys.stderr)
        else:
            versiones = args.versiones or registry.listar_versiones(args.models_dir)
            print(resumen.generar_bloque(args.models_dir, versiones))
    except ErrorAplicacion as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())