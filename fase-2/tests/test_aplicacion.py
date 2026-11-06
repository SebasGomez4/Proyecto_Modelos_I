"""Pruebas de la aplicación (ejecutar con: python -m pytest, desde la carpeta fase-2).

Usan una muestra de 300 filas de data/train.csv y directorios temporales (tmp_path):
no descargan nada y no modifican las versiones de models/.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data, inference, registry
from src.config import ID_COLUMNS_DEFECTO
from src.errors import DatosInvalidos, EntradaInvalida, VersionExistente
from src.modeling import construir_modelo
from src.preprocessing import AusenciaEstructural, definir_grupos
from src.training import entrenar
from src import predict as predict_cli

RAIZ = Path(__file__).resolve().parents[1]
TARGET = "SalePrice"
PARAMS_RAPIDOS = dict(seed=1, cv_folds=2, max_iter=50)


# ----------------------------------------------------------------------------- utilidades
def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def correr(script, *args):
    """Ejecuta train.py / predict.py como lo haría un usuario y devuelve el proceso."""
    return subprocess.run([sys.executable, str(RAIZ / script), *map(str, args)],
                          capture_output=True, text=True, cwd=RAIZ)


@pytest.fixture(scope="session")
def muestra():
    df = pd.read_csv(RAIZ / "data" / "train.csv")
    return df.sample(300, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="session")
def csv_muestra(muestra, tmp_path_factory):
    ruta = tmp_path_factory.mktemp("datos") / "muestra.csv"
    muestra.to_csv(ruta, index=False)
    return ruta


@pytest.fixture(scope="session")
def version_entrenada(csv_muestra, tmp_path_factory):
    """Una versión entrenada una sola vez y compartida (solo lectura) por varias pruebas."""
    salida = tmp_path_factory.mktemp("models")
    resultado = entrenar(csv_muestra, TARGET, salida, "t1", **PARAMS_RAPIDOS)
    return {"salida": salida, "carpeta": resultado["carpeta"], "resultado": resultado}


def entradas(muestra, n=20):
    """Filas sin la variable objetivo (como llegarían datos nuevos)."""
    return muestra.drop(columns=[TARGET]).head(n).copy()


# ----------------------------------------------------- 1. Entrenamiento y almacenamiento
def test_entrenamiento_genera_artefacto_metricas_y_metadatos(version_entrenada):
    carpeta = version_entrenada["carpeta"]
    assert sorted(p.name for p in carpeta.iterdir()) == ["metadata.json", "metrics.json", "modelo.joblib"]

    modelo, meta = registry.cargar_version(carpeta)           # el artefacto es cargable
    assert hasattr(modelo, "predict")

    metricas = json.loads((carpeta / "metrics.json").read_text(encoding="utf-8"))
    for k in ("MAE", "RMSE", "R2", "RMSLE"):                 # métricas del modelo Y del modelo base
        assert metricas["modelo"]["test"][k] == pytest.approx(metricas["modelo"]["test"][k])
        assert k in metricas["baseline"]["test"]

    # metadatos mínimos exigidos
    assert meta["version"] == "t1" and meta["variable_objetivo"] == TARGET
    assert meta["semilla"] == 1 and meta["tipo_problema"] == "regresión"
    assert len(meta["esquema_entrada"]) == 79
    assert meta["datos"]["n_observaciones"] == 300 and len(meta["datos"]["sha256"]) == 64
    assert meta["separacion"]["n_train"] + meta["separacion"]["n_test"] == 300
    assert {"python", "scikit-learn"} <= set(meta["entorno"]) and "commit" in meta
    assert "fecha_creacion" in meta and "alcance" in meta["separacion"]


def test_el_modelo_supera_al_modelo_base_en_la_misma_particion(version_entrenada):
    m = json.loads((version_entrenada["carpeta"] / "metrics.json").read_text(encoding="utf-8"))
    assert m["modelo"]["test"]["MAE"] < m["baseline"]["test"]["MAE"]
    assert 0 < m["reduccion_mae_vs_baseline_test"] < 1


def test_la_imputacion_se_ajusta_solo_con_entrenamiento(version_entrenada, csv_muestra):
    """Anti-fuga: la mediana aprendida de Lot Frontage es la del conjunto de ENTRENAMIENTO."""
    modelo, _ = registry.cargar_version(version_entrenada["carpeta"])
    df, predictoras = data.leer_datos(csv_muestra, TARGET, ID_COLUMNS_DEFECTO)
    X_train, X_test, _, _ = data.separar(df, predictoras, TARGET, 0.2, seed=1)

    num_cols = definir_grupos(X_train)[0]
    imputador = (modelo.regressor_.named_steps["preprocesamiento"]
                 .named_steps["columnas"].named_transformers_["num"])
    aprendida = imputador.statistics_[num_cols.index("Lot Frontage")]
    assert aprendida == pytest.approx(X_train["Lot Frontage"].median())
    assert not set(X_train.index) & set(X_test.index)