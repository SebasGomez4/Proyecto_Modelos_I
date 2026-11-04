"""Flujo completo de entrenamiento (lo usan train.py y las pruebas)."""
from pathlib import Path

from src import config, data, metadata, registry
from src.errors import VersionExistente
from src.modeling import (construir_baseline, construir_modelo, evaluar, validacion_cruzada)
from src.preprocessing import CATEGORICAS_CONOCIDAS


def entrenar(data_path, target, output_dir, version, seed=42, test_size=0.2, cv_folds=5,
             max_iter=300, learning_rate=0.05, id_columns=None):
    """Entrena modelo y modelo base con la misma partición y guarda una versión nueva.

    Orden (evita fuga de información): validar versión -> leer datos -> SEPARAR -> ajustar
    todo (preprocesamiento + modelo) solo con entrenamiento -> evaluar en prueba -> guardar.
    Devuelve un diccionario con las rutas y las métricas.
    """
    id_columns = list(config.ID_COLUMNS_DEFECTO if id_columns is None else id_columns)

    # falla antes de gastar tiempo entrenando si el identificador es inválido o ya existe
    registry.validar_version(version)
    destino = Path(output_dir) / version
    if destino.exists():
        raise VersionExistente(
            f"La versión '{version}' ya existe en {destino}. "
            "Elija otro identificador: las versiones guardadas no se sobrescriben."
        )

    df, predictoras = data.leer_datos(data_path, target, id_columns)
    X_train, X_test, y_train, y_test = data.separar(df, predictoras, target, test_size, seed)

    # --- modelo base (misma partición y métricas) ---
    baseline = construir_baseline().fit(X_train, y_train)
    base_test = evaluar(y_test, baseline.predict(X_test))

    # --- modelo ---
    cv_modelo = cv_base = None
    if cv_folds >= 2:
        cv_modelo = validacion_cruzada(construir_modelo(X_train, seed, max_iter, learning_rate),
                                       X_train, y_train, cv_folds, seed)
        cv_base = validacion_cruzada(construir_baseline(), X_train, y_train, cv_folds, seed)
    modelo = construir_modelo(X_train, seed, max_iter, learning_rate).fit(X_train, y_train)
    modelo_test = evaluar(y_test, modelo.predict(X_test))

    metricas = {
        "metrica_principal": config.METRICA_PRINCIPAL,
        "modelo": {"algoritmo": config.ALGORITMO, "test": modelo_test, "cv_train": cv_modelo},
        "baseline": {"algoritmo": "mediana de entrenamiento", "test": base_test, "cv_train": cv_base},
        "reduccion_mae_vs_baseline_test": 1 - modelo_test["MAE"] / base_test["MAE"],
    }
    meta = metadata.crear_metadata(
        version, objetivo=target, algoritmo=config.ALGORITMO,
        parametros={"max_iter": max_iter, "learning_rate": learning_rate, "cv_folds": cv_folds,
                    "objetivo_transformado": "log1p"},
        semilla=seed,
        esquema=metadata.esquema_entrada(X_train, categoricas=CATEGORICAS_CONOCIDAS),
        datos={"fuente": config.FUENTE_DATOS, "archivo": Path(data_path).name,
               "sha256": data.huella_archivo(data_path), "n_observaciones": int(len(df)),
               "limitaciones": config.LIMITACIONES},
        separacion={"estrategia": "aleatoria simple (train_test_split)", "test_size": test_size,
                    "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                    "alcance": config.ALCANCE},
        metrica=config.METRICA_PRINCIPAL, identificadores=id_columns)

    carpeta = registry.guardar_version(output_dir, version, modelo, metricas, meta)
    return {"carpeta": carpeta, "metricas": metricas, "metadata": meta,
            "n_train": len(X_train), "n_test": len(X_test)}