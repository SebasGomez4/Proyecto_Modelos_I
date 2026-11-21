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
                                         "MODEL_VERSION": version, "CV_FOLDS": "0", "MAX_ITER": "20",
                                         "LOG_PATH": str(tmp_path / "logs" / "predicciones.csv"), **extra})
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



# --------------------------------------------------------------- 2. /predict con entrada válida
def test_predict_con_entrada_valida_devuelve_prediccion_y_version(entorno, registro):
    r = entorno["cliente"]().post("/predict", json=registro)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"prediccion", "version"}
    assert cuerpo["version"] == "v1"
    assert isinstance(cuerpo["prediccion"], float) and 20_000 < cuerpo["prediccion"] < 800_000


def test_el_json_y_el_csv_de_la_fase_2_dan_la_misma_prediccion(entorno, registro):
    """El registro JSON es la fila 0 del CSV de ejemplo: ambos caminos deben coincidir."""
    modelo, meta = registry.cargar_version(entorno["models"] / "v1")
    fila = inference.leer_csv_entrada(FASE2 / "data" / "ejemplo_prediccion.csv").head(1)
    X, _ = inference.preparar_entrada(fila, meta["esquema_entrada"])
    esperado = round(float(inference.predecir(modelo, X)[0]), 2)

    assert entorno["cliente"]().post("/predict", json=registro).json()["prediccion"] == esperado


def test_predict_no_necesita_la_variable_objetivo_y_ignora_claves_adicionales(entorno, registro):
    cliente = entorno["cliente"]()
    sin_extras = cliente.post("/predict", json=registro).json()["prediccion"]
    assert "SalePrice" not in registro                                  # el registro ya viene sin precio
    con_extras = {**registro, "Id": 983, "PID": 1, "SalePrice": 999_999_999, "columna_inventada": "x"}
    assert cliente.post("/predict", json=con_extras).json()["prediccion"] == sin_extras


@pytest.mark.parametrize("campo", ["Lot Frontage", "Lot Area", "Neighborhood", "Electrical", "Bsmt Qual"])
def test_predict_acepta_nulos_y_los_trata_con_la_regla_de_la_fase_2(entorno, registro, campo):
    registro[campo] = None                                              # null != campo ausente
    r = entorno["cliente"]().post("/predict", json=registro)
    assert r.status_code == 200 and r.json()["version"] == "v1"


def test_predict_acepta_una_categoria_no_vista_en_el_entrenamiento(entorno, registro):
    registro["Neighborhood"] = "BarrioInexistente"
    assert entorno["cliente"]().post("/predict", json=registro).status_code == 200




# ----------------------------------------------------------------- 3. /predict rechaza entradas
@pytest.mark.parametrize("campo", ["Lot Area", "Neighborhood", "Year Built"])
def test_predict_rechaza_un_campo_obligatorio_ausente(entorno, registro, campo):
    del registro[campo]
    r = entorno["cliente"]().post("/predict", json=registro)
    assert r.status_code == 422
    cuerpo = r.json()
    assert campo in cuerpo["detail"]                                    # el mensaje nombra la variable
    assert cuerpo["errores"] == [{"campo": campo, "problema": "falta",
                                  "mensaje": f"Falta el campo obligatorio '{campo}'."}]
    assert "prediccion" not in cuerpo                                   # no se devuelve ningún resultado


def test_predict_nombra_todos_los_campos_ausentes(entorno, registro):
    for campo in ("Lot Area", "Neighborhood"):
        del registro[campo]
    detalle = entorno["cliente"]().post("/predict", json=registro).json()["detail"]
    assert "Lot Area" in detalle and "Neighborhood" in detalle


@pytest.mark.parametrize("campo, valor", [("Lot Area", "grande"), ("Neighborhood", 5), ("Year Built", True)])
def test_predict_rechaza_un_tipo_incompatible(entorno, registro, campo, valor):
    registro[campo] = valor
    r = entorno["cliente"]().post("/predict", json=registro)
    assert r.status_code == 422
    assert campo in r.json()["detail"] and r.json()["errores"][0]["problema"] == "tipo"


@pytest.mark.parametrize("cuerpo", ["esto no es json", ""])
def test_predict_con_cuerpo_que_no_es_json_responde_400(entorno, cuerpo):
    r = entorno["cliente"]().post("/predict", content=cuerpo, headers={"content-type": "application/json"})
    assert r.status_code == 400 and "JSON" in r.json()["detail"]


def test_predict_con_una_lista_en_vez_de_un_objeto_responde_422(entorno, registro):
    r = entorno["cliente"]().post("/predict", json=[registro])
    assert r.status_code == 422 and "objeto JSON" in r.json()["detail"]


def test_modelo_no_disponible_responde_503_con_ayuda(entorno, registro):
    cliente = entorno["cliente"](version="v9")                         # versión que no existe
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 503
    cuerpo = r.json()
    assert "v9" in cuerpo["detail"] and cuerpo["versiones_disponibles"] == ["v1"]
    assert "MODEL_VERSION" in cuerpo["ayuda"]                          # explica cómo elegir una versión disponible
    assert cliente.get("/health").json()["modelo"] == {"version": "v9", "disponible": False}   # sigue "en ejecución"



# ---------------------------------------------------------------------------------- 4. /train
def test_train_guarda_una_version_nueva_y_conserva_la_anterior(entorno, registro):
    models = entorno["models"]
    sha_v1 = sha256(models / "v1" / "modelo.joblib")
    cliente = entorno["cliente"]()

    r = cliente.post("/train", json={"version": "v-api", "max_iter": 25, "cv_folds": 0})
    assert r.status_code == 201
    cuerpo = r.json()
    assert cuerpo["version"] == "v-api" and cuerpo["estado"] == "guardada"
    assert cuerpo["parametros"]["max_iter"] == 25 and "MAE" in cuerpo["metricas_prueba"]

    assert sorted(p.name for p in (models / "v-api").iterdir()) == ["metadata.json", "metrics.json", "modelo.joblib"]
    meta = json.loads((models / "v-api" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["version"] == "v-api" and meta["parametros"]["max_iter"] == 25
    assert sha256(models / "v1" / "modelo.joblib") == sha_v1            # la versión anterior no cambió
    assert registry.listar_versiones(models) == ["v-api", "v1"]
    assert "v-api" in cliente.get("/health").json()["versiones_disponibles"]

    # /predict sigue con v1 hasta cambiar la configuración; con MODEL_VERSION=v-api usa la nueva
    assert cliente.post("/predict", json=registro).json()["version"] == "v1"
    nuevo = entorno["cliente"](version="v-api").post("/predict", json=registro)
    assert nuevo.status_code == 200 and nuevo.json()["version"] == "v-api"


def test_train_sin_cuerpo_genera_un_identificador_disponible(entorno):
    cliente = entorno["cliente"]()
    r1, r2 = cliente.post("/train"), cliente.post("/train")
    assert r1.status_code == r2.status_code == 201
    assert r1.json()["version"].startswith("api-") and r1.json()["version"] != r2.json()["version"]
    assert registry.listar_versiones(entorno["models"]) == sorted(["v1", r1.json()["version"], r2.json()["version"]])


def test_train_con_identificador_repetido_responde_409_y_no_sobrescribe(entorno):
    cliente = entorno["cliente"]()
    antes = {p.name: sha256(p) for p in (entorno["models"] / "v1").iterdir()}
    r = cliente.post("/train", json={"version": "v1"})
    assert r.status_code == 409 and "ya existe" in r.json()["detail"]
    assert {p.name: sha256(p) for p in (entorno["models"] / "v1").iterdir()} == antes


@pytest.mark.parametrize("cuerpo", [{"max_iter": "mucho"}, {"max_iter": 0}, {"learning_rate": -1},
                                    {"cv_folds": 1}, {"parametro_desconocido": 1}, {"version": "../fuera"}])
def test_train_con_parametros_invalidos_responde_422_y_no_crea_versiones(entorno, cuerpo):
    r = entorno["cliente"]().post("/train", json=cuerpo)
    assert r.status_code == 422 and r.json()["detail"]
    assert registry.listar_versiones(entorno["models"]) == ["v1"]


def test_train_con_json_invalido_responde_400(entorno):
    r = entorno["cliente"]().post("/train", content="no es json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_train_sin_datos_responde_500_con_mensaje_claro(entorno, tmp_path):
    r = entorno["cliente"](DATA_PATH=str(tmp_path / "no_existe.csv")).post("/train")
    assert r.status_code == 500 and "No existe el archivo de datos" in r.json()["detail"]
    assert registry.listar_versiones(entorno["models"]) == ["v1"]