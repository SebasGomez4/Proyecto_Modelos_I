"""Genera predicciones con una versión guardada del modelo.

Ejemplo (en una sola línea):
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv

No entrena ni reajusta transformaciones: solo carga el pipeline guardado y lo aplica.
Ante un error previsible muestra un mensaje claro y termina con código 1 sin crear la salida.
"""
import argparse
import sys

import sklearn

from src import inference, registry
from src.errors import ErrorAplicacion


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Predice el precio de viviendas con una versión guardada del modelo."
    )
    p.add_argument("--model", required=True,
                   help="ruta a models/<versión>/modelo.joblib (o a la carpeta de la versión)")
    p.add_argument("--input", required=True, help="CSV con las variables de entrada (sin objetivo)")
    p.add_argument("--output", required=True, help="CSV de salida con la columna 'prediccion'")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        modelo, metadata = registry.cargar_version(args.model)
        if "esquema_entrada" not in metadata:
            raise ErrorAplicacion("metadata.json no contiene 'esquema_entrada'; versión incompleta.")

        version_sklearn = metadata.get("entorno", {}).get("scikit-learn")
        if version_sklearn and version_sklearn != sklearn.__version__:
            print(f"Aviso: el modelo se entrenó con scikit-learn {version_sklearn} y aquí hay "
                  f"{sklearn.__version__}.", file=sys.stderr)

        df = inference.leer_csv_entrada(args.input)
        X, extra = inference.preparar_entrada(df, metadata["esquema_entrada"])
        ids = metadata.get("columnas_identificador", [])
        extra = [c for c in extra if c not in ids]  # los identificadores se usan en la salida
        if extra:
            print(f"Aviso: se ignoran {len(extra)} columnas que el modelo no usa: {', '.join(extra)}",
                  file=sys.stderr)
        pred = inference.predecir(modelo, X)
        ruta = inference.escribir_salida(df, pred, ids, args.output)
    except ErrorAplicacion as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # cualquier otra falla: mensaje comprensible y código distinto de cero
        print(f"Error inesperado ({type(e)._name_}): {e}", file=sys.stderr)
        return 1

    print(f"Versión usada:   {metadata.get('version', '?')}")
    print(f"Filas predichas: {len(pred)}")
    print(f"Salida:          {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())