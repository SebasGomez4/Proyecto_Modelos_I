"""Entrena una versión del modelo (esqueleto: la lógica llega en feature/train-script)."""
import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entrena una versión del modelo de precios de viviendas.")
    p.add_argument("--data", required=True, help="CSV con datos etiquetados")
    p.add_argument("--target", default="SalePrice", help="nombre de la variable objetivo")
    p.add_argument("--output-dir", default="models", help="carpeta donde se guardan las versiones")
    p.add_argument("--version", required=True, help="identificador único de la versión, p. ej. v1")
    p.add_argument("--seed", type=int, default=42, help="semilla aleatoria")
    p.add_argument("--test-size", type=float, default=0.2, help="fracción para prueba")
    return p.parse_args(argv)


def main(argv=None):
    parse_args(argv)
    raise SystemExit("train.py: entrenamiento aún no implementado")


if __name__ == "__main__":
    main()