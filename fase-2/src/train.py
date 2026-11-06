"""Entrena y guarda una versión del modelo de precios de viviendas.

Ejemplo (en una sola línea, desde la carpeta fase-2):
    python train.py --data data/train.csv --target SalePrice --output-dir models --version v1 --seed 42

Cada ejecución crea models/<versión>/ con modelo.joblib, metrics.json y metadata.json.
Una versión existente no se sobrescribe: si el identificador se repite, el script termina con error.
"""
import argparse
import sys
from pathlib import Path

# Garantizar que la raíz del proyecto esté en el PATH para importar el paquete 'src',
# tanto si el script se ejecuta desde src/ como si se copia a la raíz (p. ej. en Docker).
_AQUI = Path(__file__).resolve().parent
_RAIZ = _AQUI.parent if (_AQUI.parent / "src" / "__init__.py").is_file() else _AQUI
sys.path.insert(0, str(_RAIZ))

try:
    from src import config
    from src.errors import ErrorAplicacion, VersionExistente
    from src.training import entrenar
except ImportError as e:
    print(f"Error de importación: {e}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Entrena una versión del modelo de precios de viviendas.")
    p.add_argument("--data", required=True, help="CSV con datos etiquetados")
    p.add_argument("--target", default=config.TARGET_DEFECTO, help="nombre de la variable objetivo")
    p.add_argument("--output-dir", default="models", help="carpeta donde se guardan las versiones")
    p.add_argument("--version", required=True, help="identificador único de la versión (p. ej. v1)")
    p.add_argument("--seed", type=int, default=42, help="semilla aleatoria (partición, CV y modelo)")
    p.add_argument("--test-size", type=float, default=0.2, help="fracción de datos para prueba")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="folds de validación cruzada sobre entrenamiento (0 para omitirla)")
    p.add_argument("--max-iter", type=int, default=300, help="iteraciones (árboles) de Gradient Boosting")
    p.add_argument("--learning-rate", type=float, default=0.05, help="tasa de aprendizaje")
    p.add_argument("--id-columns", nargs="*", default=config.ID_COLUMNS_DEFECTO,
                   help="columnas identificadoras que se excluyen de las predictoras")
    return p.parse_args(argv)


def main(argv=None):
    try:
        args = parse_args(argv)
        version_dir = Path(args.output_dir) / args.version

        if version_dir.exists():
            print(f"Error: ya existe la versión '{args.version}' en {args.output_dir}.", file=sys.stderr)
            sys.exit(1)

        r = entrenar(args.data, args.target, args.output_dir, args.version, seed=args.seed,
                     test_size=args.test_size, cv_folds=args.cv_folds, max_iter=args.max_iter,
                     learning_rate=args.learning_rate, id_columns=args.id_columns)

        m = r["metricas"]
        print(f"Versión:     {args.version}")
        print(f"Datos:       {r['n_train']} train / {r['n_test']} test")
        print(f"Baseline:    MAE = {m['baseline']['test']['MAE']:,.0f}  R2 = {m['baseline']['test']['R2']:.3f}")
        print(f"Modelo:      MAE = {m['modelo']['test']['MAE']:,.0f}  R2 = {m['modelo']['test']['R2']:.3f}  "
              f"RMSLE = {m['modelo']['test']['RMSLE']:.3f}")
        print(f"Reducción del MAE frente al baseline (test): {m['reduccion_mae_vs_baseline_test']:.1%}")
        print(f"Archivos en: {r['carpeta']}  (modelo.joblib, metrics.json, metadata.json)")
        return 0

    except (VersionExistente, ErrorAplicacion) as e:
        msg = str(e)
        if "ya existe" not in msg:
            print(f"Error: ya existe la versión ({msg})", file=sys.stderr)
        else:
            print(f"Error: {msg}", file=sys.stderr)
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