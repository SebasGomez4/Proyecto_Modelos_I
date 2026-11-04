"""Entrena y guarda una versión del modelo de precios de viviendas.

Ejemplo (en una sola línea, desde la carpeta fase-2):
    python train.py --data data/train.csv --target SalePrice --output-dir models --version v1 --seed 42

Cada ejecución crea models/<versión>/ con modelo.joblib, metrics.json y metadata.json.
Una versión existente no se sobrescribe: si el identificador se repite, el script termina con error.
"""
import argparse
import sys

from src import config
from src.errors import ErrorAplicacion
from src.training import entrenar


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
    args = parse_args(argv)
    try:
        r = entrenar(args.data, args.target, args.output_dir, args.version, seed=args.seed,
                     test_size=args.test_size, cv_folds=args.cv_folds, max_iter=args.max_iter,
                     learning_rate=args.learning_rate, id_columns=args.id_columns)
    except ErrorAplicacion as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error inesperado ({type(e)._name_}): {e}", file=sys.stderr)
        return 1

    m = r["metricas"]
    print(f"Versión:     {args.version}")
    print(f"Datos:       {r['n_train']} train / {r['n_test']} test")
    print(f"Baseline:    MAE = {m['baseline']['test']['MAE']:,.0f}  R2 = {m['baseline']['test']['R2']:.3f}")
    print(f"Modelo:      MAE = {m['modelo']['test']['MAE']:,.0f}  R2 = {m['modelo']['test']['R2']:.3f}  "
          f"RMSLE = {m['modelo']['test']['RMSLE']:.3f}")
    print(f"Reducción del MAE frente al baseline (test): {m['reduccion_mae_vs_baseline_test']:.1%}")
    print(f"Archivos en: {r['carpeta']}  (modelo.joblib, metrics.json, metadata.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())