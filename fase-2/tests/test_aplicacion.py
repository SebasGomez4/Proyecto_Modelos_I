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

    

# ------------------------------------------------------------- 2. Carga y predicción
def test_modelo_guardado_conserva_las_predicciones_del_original(muestra, tmp_path):
    X = muestra.drop(columns=[TARGET] + ID_COLUMNS_DEFECTO)
    original = construir_modelo(X, semilla=3, max_iter=20).fit(X, muestra[TARGET])
    antes = original.predict(X)

    registry.guardar_version(tmp_path, "vt", original, {"MAE": 0}, {"version": "vt"})
    cargado, _ = registry.cargar_version(tmp_path / "vt" / "modelo.joblib")
    despues = cargado.predict(X)

    assert len(despues) == len(X)
    np.testing.assert_allclose(despues, antes, rtol=1e-9)


def test_predict_da_una_prediccion_por_fila_en_el_orden_original(version_entrenada, muestra, tmp_path):
    X = entradas(muestra, 25)
    X = X[list(reversed(X.columns))]                          # el orden de columnas del CSV no importa
    entrada, salida = tmp_path / "in.csv", tmp_path / "out.csv"
    X.to_csv(entrada, index=False)

    assert predict_cli.main(["--model", str(version_entrenada["carpeta"] / "modelo.joblib"),
                             "--input", str(entrada), "--output", str(salida)]) == 0
    res = pd.read_csv(salida)
    assert len(res) == 25 and list(res["fila"]) == list(range(1, 26))
    assert res["Id"].tolist() == X["Id"].tolist()             # misma correspondencia con la entrada

    modelo, meta = registry.cargar_version(version_entrenada["carpeta"])
    esperado = modelo.predict(X[[c["nombre"] for c in meta["esquema_entrada"]]])
    np.testing.assert_allclose(res["prediccion"], esperado, atol=0.01)


def test_categoria_nueva_y_columna_adicional_no_rompen_ni_cambian_la_prediccion(version_entrenada, muestra):
    modelo, meta = registry.cargar_version(version_entrenada["carpeta"])
    base = entradas(muestra, 10)
    raro = base.copy()
    raro["Neighborhood"] = "BarrioInexistente"                # categoría no vista en entrenamiento
    raro["columna_extra"] = 123                               # columna que el modelo no usa

    X, extra = inference.preparar_entrada(raro, meta["esquema_entrada"])
    assert "columna_extra" in extra and "columna_extra" not in X.columns
    pred = inference.predecir(modelo, X)
    assert len(pred) == 10 and np.isfinite(pred).all()

    X_base, _ = inference.preparar_entrada(base, meta["esquema_entrada"])
    X_sin_extra, _ = inference.preparar_entrada(base.assign(columna_extra=1), meta["esquema_entrada"])
    np.testing.assert_allclose(inference.predecir(modelo, X_sin_extra), inference.predecir(modelo, X_base))


# ----------------------------------------------------------------- 3. Valores faltantes
def test_ausencia_estructural_se_distingue_de_dato_desconocido():
    X = pd.DataFrame({
        "Bsmt Qual":     [np.nan, np.nan, np.nan, "TA"],
        "Total Bsmt SF": [0.0,    800.0,  np.nan, 900.0],
        "Alley":         [np.nan, "Pave", np.nan, np.nan],
    })
    t = AusenciaEstructural().fit(X).transform(X)
    assert t["Bsmt Qual"].tolist() == ["None",         # sin sótano: ausencia real
                                       "Desconocido",  # hay sótano pero falta el dato
                                       "Desconocido",  # la referencia tampoco existe
                                       "TA"]
    assert t["Alley"].tolist() == ["None", "Pave", "None", "None"]   # sin referencia: "None"


def test_entrada_con_nulos_se_procesa_segun_la_estrategia_documentada(version_entrenada, muestra):
    modelo, meta = registry.cargar_version(version_entrenada["carpeta"])
    X = entradas(muestra, 12)
    X.loc[0:5, "Lot Frontage"] = np.nan                       # numérica: mediana + indicador
    X.loc[3:8, "Mas Vnr Area"] = np.nan
    X.loc[6:11, "Bsmt Qual"] = np.nan                         # categórica: None / Desconocido
    X.loc[9:11, "Total Bsmt SF"] = np.nan                     # referencia nula: no es columna ausente

    Xp, _ = inference.preparar_entrada(X, meta["esquema_entrada"])
    assert Xp["Lot Frontage"].iloc[0:6].isna().all()          # los nulos llegan al modelo, no se descartan filas
    pred = inference.predecir(modelo, Xp)
    assert len(pred) == 12 and np.isfinite(pred).all()