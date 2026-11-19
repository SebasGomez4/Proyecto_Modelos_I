"""Pruebas del resumen de versiones: la tabla debe coincidir exactamente con los artefactos."""
import hashlib
import json
from pathlib import Path

import pytest

from src import resumen
from src.errors import ErrorAplicacion

RAIZ = Path(__file__).resolve().parents[1]


def crear_version(base, nombre, *, contenido=b"modelo-A", semilla=42, max_iter=300, lr=0.05,
                  mae=14769.261, rmse=22461.491, r2=0.919, rmsle=0.136, n_train=1406, n_test=352):
    """Escribe una versión mínima (artefacto falso + metadatos y métricas con las claves reales)."""
    carpeta = base / nombre
    carpeta.mkdir(parents=True)
    (carpeta / "modelo.joblib").write_bytes(contenido)
    metadata = {
        "version": nombre, "fecha_creacion": "2026-11-05T15:30:12+00:00",
        "commit": "3aace47f9c1d2e0b7a6f5e4d3c2b1a0987654321", "semilla": semilla,
        "parametros": {"max_iter": max_iter, "learning_rate": lr},
        "entorno": {"python": "3.12.3", "scikit-learn": "1.8.0"},
        "separacion": {"test_size": 0.2, "n_train": n_train, "n_test": n_test},
        "datos": {"sha256": "a" * 64},
    }
    metricas = {
        "modelo": {"test": {"MAE": mae, "RMSE": rmse, "R2": r2, "RMSLE": rmsle},
                   "cv_train": {"MAE": mae + 800, "MAE_std": 1200.0}},
        "baseline": {"test": {"MAE": 53976.918}},
    }
    (carpeta / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (carpeta / "metrics.json").write_text(json.dumps(metricas), encoding="utf-8")


def tabla(base, versiones):
    return resumen.generar_markdown(resumen.cargar_versiones(base, versiones))


def fila(md, version, contiene):
    """La fila de `version` que contiene `contiene` (hay una de trazabilidad y otra de métricas)."""
    return next(l for l in md.splitlines() if l.startswith(f"| `{version}` |") and contiene in l)


def test_la_tabla_refleja_exactamente_los_artefactos(tmp_path):
    crear_version(tmp_path, "v1")
    crear_version(tmp_path, "v2", contenido=b"modelo-B", max_iter=500, lr=0.03, mae=14937.585, rmse=22722.797)
    md = tabla(tmp_path, ["v1", "v2"])

    traza_v2 = fila(md, "v2", "500")
    assert "| 500 | 0,03 |" in traza_v2                                    # parámetros reales de v2
    assert hashlib.sha256(b"modelo-B").hexdigest()[:12] in traza_v2        # SHA del archivo publicado
    assert "3aace47" in traza_v2 and "3.12.3" in traza_v2 and "1.8.0" in traza_v2
    assert "14.938" in fila(md, "v2", "14.938") and "14.769" not in fila(md, "v2", "14.938")
    assert "14.769" in fila(md, "v1", "14.769")
    assert "72,6 %" in fila(md, "v1", "14.769")                            # reducción frente al modelo base


def test_detecta_artefactos_identicos(tmp_path):
    crear_version(tmp_path, "v1")
    crear_version(tmp_path, "v2")                                          # mismos bytes y configuración
    md = tabla(tmp_path, ["v1", "v2"])
    assert "Artefactos idénticos" in md and "`v1`, `v2`" in md and "no representan una mejora" in md

    crear_version(tmp_path, "v3", contenido=b"otro", max_iter=500)
    assert "Artefactos idénticos" not in tabla(tmp_path, ["v1", "v3"])


def test_advierte_si_la_particion_cambia(tmp_path):
    crear_version(tmp_path, "v1")
    crear_version(tmp_path, "v2", contenido=b"B", semilla=7)               # otra semilla = otro conjunto de prueba
    assert "no son directamente comparables" in tabla(tmp_path, ["v1", "v2"])

    crear_version(tmp_path, "v3", contenido=b"C", max_iter=500)            # solo cambian hiperparámetros
    md = tabla(tmp_path, ["v1", "v3"])
    assert "no son directamente comparables" not in md and "misma partición" in md


def test_version_inexistente_da_error_claro(tmp_path):
    crear_version(tmp_path, "v1")
    with pytest.raises(ErrorAplicacion, match="v9"):
        resumen.cargar_versiones(tmp_path, ["v1", "v9"])


def test_actualizar_readme_reemplaza_solo_el_bloque_y_es_idempotente(tmp_path):
    models = tmp_path / "models"
    crear_version(models, "v1")
    crear_version(models, "v2", contenido=b"B", max_iter=500, lr=0.03, mae=14937.585)
    readme = tmp_path / "README.md"
    readme.write_text("# Título\n\nantes\n\n<!-- RESUMEN-VERSIONES:INICIO versiones=v1,v2 -->\n"
                      "TABLA VIEJA INCORRECTA\n<!-- RESUMEN-VERSIONES:FIN -->\n\ndespués\n", encoding="utf-8")

    resumen.actualizar_readme(readme, models)
    texto = readme.read_text(encoding="utf-8")
    assert texto.startswith("# Título\n\nantes\n\n") and texto.endswith("\n\ndespués\n")
    assert "TABLA VIEJA" not in texto and "14.938" in texto

    resumen.actualizar_readme(readme, models)
    assert readme.read_text(encoding="utf-8") == texto                     # repetir no cambia nada


def test_readme_sin_bloque_marcado_da_error_claro(tmp_path):
    crear_version(tmp_path / "models", "v1")
    readme = tmp_path / "README.md"
    readme.write_text("# sin marcadores\n", encoding="utf-8")
    with pytest.raises(ErrorAplicacion, match="RESUMEN-VERSIONES"):
        resumen.actualizar_readme(readme, tmp_path / "models")