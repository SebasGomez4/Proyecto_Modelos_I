"""Pruebas del registro de solicitudes de /predict (Fase 4): una fila por intento, rechazos incluidos."""
import copy
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import rutas  # noqa: F401  (hace importable fase-2/src)
from ajustes import Ajustes
from api import create_app
from registro import CAMPOS, FORMATO_FECHA, RegistroSolicitudes
from src import training

RAIZ = Path(__file__).resolve().parents[1]
FASE2 = RAIZ.parent / "fase-2"
REGISTRO = json.loads((RAIZ / "ejemplos" / "registro_valido.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def base(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("monitoreo")
    csv_datos = carpeta / "muestra.csv"
    pd.read_csv(FASE2 / "data" / "train.csv").sample(300, random_state=0).to_csv(csv_datos, index=False)
    training.entrenar(csv_datos, "SalePrice", carpeta / "models", "v1", seed=1, cv_folds=0, max_iter=30)
    return {"models": carpeta / "models"}


@pytest.fixture
def entorno(base, tmp_path):
    models = tmp_path / "models"
    shutil.copytree(base["models"], models)
    log = tmp_path / "logs" / "predicciones.csv"

    def cliente(version="v1", log_path=log):
        return TestClient(create_app(Ajustes.desde_entorno(
            {"MODELS_DIR": str(models), "MODEL_VERSION": version, "LOG_PATH": str(log_path)})))

    return {"cliente": cliente, "log": log}


def filas(ruta):
    with open(ruta, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def registro():
    return copy.deepcopy(REGISTRO)


def test_dos_solicitudes_validas_y_una_invalida_generan_tres_filas(entorno, registro):
    """El ensayo de la especificación: sobre un registro vacío, 2 válidas + 1 con una variable ausente."""
    cliente = entorno["cliente"]()
    assert not entorno["log"].exists()                                  # el registro empieza vacío
    r1 = cliente.post("/predict", json=registro)
    r2 = cliente.post("/predict", json=registro)
    incompleto = {k: v for k, v in registro.items() if k != "Lot Area"}
    r3 = cliente.post("/predict", json=incompleto)

    assert entorno["log"].read_text(encoding="utf-8").splitlines()[0] == "fecha_hora,estado_http,version,prediccion"
    f = filas(entorno["log"])
    assert len(f) == 3                                                  # exactamente una fila por intento
    assert [x["estado_http"] for x in f] == ["200", "200", "422"]
    assert [x["version"] for x in f] == ["v1", "v1", "v1"]
    assert f[0]["prediccion"] == str(r1.json()["prediccion"]) and f[1]["prediccion"] == str(r2.json()["prediccion"])
    assert f[2]["prediccion"] == "" and r3.status_code == 422           # en un error, la predicción queda vacía


def test_se_registran_todos_los_rechazos_con_su_codigo(entorno, registro):
    cliente = entorno["cliente"]()
    cliente.post("/predict", content="no es json", headers={"content-type": "application/json"})   # 400
    cliente.post("/predict", json={**registro, "Lot Area": "grande"})                              # 422 (tipo)
    cliente.get("/predict")                                                                        # 405
    entorno["cliente"](version="v9").post("/predict", json=registro)                               # 503
    f = filas(entorno["log"])
    assert [x["estado_http"] for x in f] == ["400", "422", "405", "503"]
    # la versión solo aparece si se llegó a usar un modelo: 422 validó con v1; 400, 405 y 503 no usaron ninguno
    assert [x["version"] for x in f] == ["", "v1", "", ""]
    assert all(x["prediccion"] == "" for x in f)


def test_el_registro_se_conserva_entre_ejecuciones(entorno, registro):
    entorno["cliente"]().post("/predict", json=registro)
    nueva_ejecucion = entorno["cliente"]()                              # otra instancia de la app, mismo archivo
    nueva_ejecucion.post("/predict", json=registro)
    lineas = entorno["log"].read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 3 and lineas.count("fecha_hora,estado_http,version,prediccion") == 1   # un solo encabezado
    assert len(filas(entorno["log"])) == 2


def test_solo_se_registra_predict(entorno, registro):
    cliente = entorno["cliente"]()
    cliente.get("/health")
    cliente.get("/modelo")
    assert not entorno["log"].exists()
    cliente.post("/predict", json=registro)
    assert len(filas(entorno["log"])) == 1


def test_fecha_hora_tiene_el_formato_documentado(entorno, registro):
    entorno["cliente"]().post("/predict", json=registro)
    fecha = filas(entorno["log"])[0]["fecha_hora"]
    assert datetime.strptime(fecha, FORMATO_FECHA)


def test_un_fallo_al_escribir_el_registro_no_rompe_la_api(entorno, registro, tmp_path, capsys):
    carpeta_no_archivo = tmp_path / "es_una_carpeta"
    carpeta_no_archivo.mkdir()                                          # abrirla para agregar produce un error de escritura
    r = entorno["cliente"](log_path=carpeta_no_archivo).post("/predict", json=registro)
    assert r.status_code == 200 and "prediccion" in r.json()
    assert "no se pudo escribir el registro" in capsys.readouterr().err


def test_escrituras_simultaneas_no_mezclan_ni_pierden_filas(tmp_path):
    reg = RegistroSolicitudes(tmp_path / "log.csv")
    with ThreadPoolExecutor(max_workers=16) as pool:
        resultados = list(pool.map(lambda i: reg.registrar(200 if i % 2 else 422, "v1", float(i)), range(200)))
    assert all(resultados)
    f = filas(tmp_path / "log.csv")
    assert len(f) == 200 and set(f[0]) == set(CAMPOS)
    assert sorted(float(x["prediccion"]) for x in f) == [float(i) for i in range(200)]