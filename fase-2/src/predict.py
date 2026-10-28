"""Genera predicciones con una versión guardada (esqueleto: la lógica llega en feature/predict-script)."""
import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Predice precios con una versión guardada del modelo.")
    p.add_argument("--model", required=True, help="ruta a modelos/<versión>/modelo.joblib")
    p.add_argument("--input", required=True, help="CSV con las variables de entrada")
    p.add_argument("--output", required=True, help="CSV de salida con la columna prediccion")
    return p.parse_args(argv)


def main(argv=None):
    parse_args(argv)
    raise SystemExit("predict.py: predicción aún no implementada")


if __name__ == "__main__":
    main()