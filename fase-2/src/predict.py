"""Genera predicciones con una versión guardada del modelo.

Ejemplo (en una sola línea):
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv

No entrena ni reajusta transformaciones: solo carga el pipeline guardado y lo aplica.
Ante un error previsible muestra un mensaje claro y termina con código 1 sin crear la salida.
"""
import argparse
import sys
from pathlib import Path

# Garantizar que la raíz del proyecto esté en el PATH para importar el paquete 'src',
# tanto si el script se ejecuta desde src/ como si se copia a la raíz (p. ej. en Docker).
_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parent if (_AQUI.parent / "src" / "__init__.py").is_file() else _AQUI
sys.path.insert(0, str(_RAIZ))

import sklearn

try:
    from src import inference, registry
    from src.errors import ErrorAplicacion, EntradaInvalida
except ImportError as e:
    print(f"Error de importación: {e}", file=sys.stderr)
    sys.exit(1)


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
    try:
        args = parse_args(argv)
        model_path = Path(args.model)

        if not model_path.exists():
            print(f"Error: No existe el modelo en {model_path}", file=sys.stderr)
            sys.exit(1)

        modelo, metadata = registry.cargar_version(args.model)
        if "esquema_entrada" not in metadata:
            print("Error: metadata.json no contiene 'esquema_entrada'; versión incompleta.", file=sys.stderr)
            sys.exit(1)

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

        print(f"Versión usada:   {metadata.get('version', '?')}")
        print(f"Filas predichas: {len(pred)}")
        print(f"Salida:          {ruta}")
        return 0

    except (EntradaInvalida, ErrorAplicacion, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except SystemExit as e:
        if e.code != 0:
            sys.exit(e.code)
        return 0
    except Exception as e:
        print(f"Error inesperado: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()