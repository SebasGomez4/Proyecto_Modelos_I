"""Pruebas de la API (ejecutar con: python -m pytest, desde la carpeta fase-3).

Usan un modelo pequeño entrenado al inicio con una muestra de 300 filas de fase-2/data/train.csv y
carpetas temporales: no descargan nada ni tocan fase-2/models. Las tres pruebas que exige la
especificación son:
  1. test_health_responde_200_y_estado
  2. test_predict_con_entrada_valida_devuelve_prediccion_y_version
  3. test_predict_rechaza_un_campo_obligatorio_ausente
"""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import rutas  # noqa: F401  (hace importable fase-2/src)
from ajustes import Ajustes
from api import create_app
from src import inference, registry, training

RAIZ = Path(__file__).resolve().parents[1]
FASE2 = RAIZ.parent / "fase-2"
REGISTRO = json.loads((RAIZ / "ejemplos" / "registro_valido.json").read_text(encoding="utf-8"))


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def base(tmp_path_factory):
    """Modelo pequeño v1 (entrenado una sola vez) y la muestra de datos con la que se entrenó."""
    carpeta = tmp_path_factory.mktemp("api")
    csv = carpeta / "muestra.csv"
    pd.read_csv(FASE2 / "data" / "train.csv").sample(300, random_state=0).to_csv(csv, index=False)
    training.entrenar(csv, "SalePrice", carpeta / "models", "v1", seed=1, cv_folds=0, max_iter=30)
    return {"models": carpeta / "models", "csv": csv}


@pytest.fixture
def entorno(base, tmp_path):
    """Una copia de la carpeta de modelos por prueba (así /train no contamina a las demás)."""
    models = tmp_path / "models"
    shutil.copytree(base["models"], models)

    def cliente(version="v1", **extra):
        ajustes = Ajustes.desde_entorno({"MODELS_DIR": str(models), "DATA_PATH": str(base["csv"]),
                                         "MODEL_VERSION": version, "CV_FOLDS": "0", "MAX_ITER": "20", **extra})
        return TestClient(create_app(ajustes))

    return {"models": models, "cliente": cliente}


@pytest.fixture
def registro():
    """Un registro válido de la fase 2 (fila 0 del CSV de ejemplo) convertido a JSON."""
    return copy.deepcopy(REGISTRO)


# ------------------------------------------------------------------ 1. /health (exigida)
def test_health_responde_200_y_estado(entorno):
    r = entorno["cliente"]().get("/health")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["estado"] == "ok" and "en ejecución" in cuerpo["mensaje"]
    assert cuerpo["modelo"] == {"version": "v1", "disponible": True}
    assert cuerpo["versiones_disponibles"] == ["v1"]