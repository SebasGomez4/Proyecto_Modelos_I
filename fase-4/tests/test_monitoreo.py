"""Pruebas del script de monitoreo: total de solicitudes y porcentaje de fallos (HTTP 4xx o 5xx)."""
import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import monitoreo
import rutas  # noqa: F401  (hace importable fase-2/src)
from ajustes import Ajustes
from api import create_app
from registro import CAMPOS, RegistroSolicitudes
from src import training

RAIZ = Path(__file__).resolve().parents[1]
FASE2 = RAIZ.parent / "fase-2"
REGISTRO = json.loads((RAIZ / "ejemplos" / "registro_valido.json").read_text(encoding="utf-8"))
ENCABEZADO = ",".join(CAMPOS)


def escribir(ruta, *lineas):
    Path(ruta).write_text("\n".join([ENCABEZADO, *lineas]) + "\n", encoding="utf-8")
    return ruta


def ejecutar(capsys, *argumentos):
    codigo = monitoreo.main([str(a) for a in argumentos])
    salida = capsys.readouterr()
    return codigo, salida.out, salida.err


# ------------------------------------------------------------------ ejemplo de la especificación
def test_dos_validas_y_una_invalida_dan_3_solicitudes_y_33_coma_3_por_ciento(tmp_path, capsys):
    log = escribir(tmp_path / "log.csv",
                   "2026-11-06T10:00:00,200,v1,18.7", "2026-11-06T10:01:00,200,v1,19.2", "2026-11-06T10:02:00,422,,")
    codigo, salida, _ = ejecutar(capsys, "--registro", log)
    assert codigo == 0
    assert "Total de solicitudes: 3" in salida
    assert "Solicitudes fallidas (HTTP 4xx o 5xx): 1" in salida
    assert "Porcentaje de fallos: 33,3 %" in salida


def test_registro_vacio_muestra_total_0_y_sin_solicitudes(tmp_path, capsys):
    con_encabezado = escribir(tmp_path / "solo_encabezado.csv")
    sin_bytes = tmp_path / "vacio.csv"
    sin_bytes.write_text("", encoding="utf-8")
    for log in (con_encabezado, sin_bytes):
        codigo, salida, _ = ejecutar(capsys, "--registro", log)
        assert codigo == 0
        assert "Total de solicitudes: 0" in salida and "Porcentaje de fallos: sin solicitudes" in salida


# ------------------------------------------------------------------------------ qué cuenta como fallida
@pytest.mark.parametrize("codigo, fallida", [(200, False), (201, False), (399, False), (400, True), (404, True),
                                             (422, True), (499, True), (500, True), (503, True), (599, True),
                                             (600, False)])
def test_solo_los_codigos_4xx_y_5xx_son_fallidos(codigo, fallida):
    assert monitoreo.es_fallida(codigo) is fallida


@pytest.mark.parametrize("filas, esperado", [
    ([200] * 7 + [500], "12,5 %"), ([200, 200], "0,0 %"), ([422, 503], "100,0 %"), ([200, 200, 422], "33,3 %"),
    ([200] + [400] * 2, "66,7 %")])
def test_formato_del_porcentaje_con_coma_y_una_cifra_decimal(filas, esperado):
    m = monitoreo.calcular_metricas([{"estado_http": c, "version": "v1"} for c in filas])
    assert monitoreo.formatear_porcentaje(m["porcentaje_fallos"]) == esperado


def test_el_script_no_modifica_el_registro(tmp_path, capsys):
    log = escribir(tmp_path / "log.csv", "2026-11-06T10:00:00,200,v1,18.7", "2026-11-06T10:02:00,422,,")
    antes = log.read_bytes()
    ejecutar(capsys, "--registro", log, "--detalle")
    assert log.read_bytes() == antes


def test_detalle_cuenta_por_codigo_y_por_version(tmp_path, capsys):
    log = escribir(tmp_path / "log.csv", "2026-11-06T10:00:00,200,v1,1", "2026-11-06T10:01:00,200,v2,2",
                   "2026-11-06T10:02:00,422,v1,", "2026-11-06T10:03:00,503,,")
    _, salida, _ = ejecutar(capsys, "--registro", log, "--detalle")
    assert "200: 2, 422: 1, 503: 1" in salida
    assert "(sin versión): 1, v1: 2, v2: 1" in salida


# ---------------------------------------------------------------------------------------- errores
def test_registro_inexistente_da_error_claro_y_codigo_distinto_de_cero(tmp_path, capsys):
    codigo, salida, error = ejecutar(capsys, "--registro", tmp_path / "no_existe.csv")
    assert codigo == 1 and "No existe el registro" in error and "Traceback" not in error and salida == ""


def test_fila_con_codigo_no_numerico_se_informa_con_su_linea(tmp_path, capsys):
    log = escribir(tmp_path / "log.csv", "2026-11-06T10:00:00,200,v1,1", "2026-11-06T10:01:00,abc,v1,2")
    codigo, _, error = ejecutar(capsys, "--registro", log)
    assert codigo == 1 and "Línea 3" in error and "'abc'" in error


def test_encabezado_incorrecto_da_error_claro(tmp_path, capsys):
    log = tmp_path / "log.csv"
    log.write_text("hora,codigo\n2026-11-06T10:00:00,200\n", encoding="utf-8")
    codigo, _, error = ejecutar(capsys, "--registro", log)
    assert codigo == 1 and "estado_http" in error


# ------------------------------------------------ integración: registro real de la API -> indicadores
@pytest.fixture(scope="session")
def modelo_pequeno(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("monitoreo_script")
    datos = carpeta / "muestra.csv"
    pd.read_csv(FASE2 / "data" / "train.csv").sample(300, random_state=0).to_csv(datos, index=False)
    training.entrenar(datos, "SalePrice", carpeta / "models", "v1", seed=1, cv_folds=0, max_iter=30)
    return carpeta / "models"


def test_el_ensayo_completo_api_registro_y_script_da_3_solicitudes_y_33_coma_3(modelo_pequeno, tmp_path, capsys):
    models, log = tmp_path / "models", tmp_path / "logs" / "predicciones.csv"
    shutil.copytree(modelo_pequeno, models)
    cliente = TestClient(create_app(Ajustes.desde_entorno(
        {"MODELS_DIR": str(models), "MODEL_VERSION": "v1", "LOG_PATH": str(log)})))
    registro = copy.deepcopy(REGISTRO)

    assert cliente.post("/predict", json=registro).status_code == 200
    assert cliente.post("/predict", json=registro).status_code == 200
    assert cliente.post("/predict", json={k: v for k, v in registro.items() if k != "Lot Area"}).status_code == 422

    codigo, salida, _ = ejecutar(capsys, "--registro", log)
    assert codigo == 0
    assert "Total de solicitudes: 3" in salida and "Porcentaje de fallos: 33,3 %" in salida
    assert [f["estado_http"] for f in monitoreo.leer_registro(log)] == [200, 200, 422]   # coincide con las filas


def test_el_script_lee_lo_que_escribe_el_modulo_del_registro(tmp_path):
    reg = RegistroSolicitudes(tmp_path / "log.csv")
    reg.registrar(200, "v1", 171698.12)
    reg.registrar(503)
    m = monitoreo.calcular_metricas(monitoreo.leer_registro(tmp_path / "log.csv"))
    assert (m["total"], m["fallidas"], m["porcentaje_fallos"]) == (2, 1, 50.0)