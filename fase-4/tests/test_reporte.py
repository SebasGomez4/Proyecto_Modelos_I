"""Pruebas que atan el README de la fase 4 a los artefactos: registro de muestra y versiones del modelo.

Si alguien cambia a mano una cifra del README (o el registro de muestra) sin regenerar el bloque,
estas pruebas fallan: lo documentado debe corresponder exactamente a lo ejecutado.
"""
import re
from pathlib import Path

import monitoreo
import rutas  # noqa: F401  (hace importable fase-2/src)
from src import resumen

RAIZ = Path(__file__).resolve().parents[1]
FASE2 = RAIZ.parent / "fase-2"
README = (RAIZ / "README.md").read_text(encoding="utf-8")
MUESTRA = RAIZ / "muestra_registro.csv"


def test_el_reporte_del_readme_coincide_con_la_muestra_del_registro():
    bloque = monitoreo.PATRON_REPORTE.search(README)
    assert bloque, "El README debe tener el bloque REPORTE-MONITOREO"
    esperado = monitoreo.generar_reporte(monitoreo.leer_registro(MUESTRA))
    assert bloque.group(0) == esperado, ("El reporte del README no coincide con muestra_registro.csv. "
                                         "Ejecute: python monitoreo.py --registro muestra_registro.csv "
                                         "--actualizar-readme README.md")


def test_la_muestra_es_el_ensayo_de_la_especificacion_dos_validas_y_una_invalida():
    filas = monitoreo.leer_registro(MUESTRA)
    assert sorted(f["estado_http"] for f in filas) == [200, 200, 422]
    assert [f["prediccion"] != "" for f in filas if f["estado_http"] == 200] == [True, True]
    assert all(f["prediccion"] == "" for f in filas if f["estado_http"] != 200)   # en un error no hay predicción
    metricas = monitoreo.calcular_metricas(filas)
    assert (metricas["total"], metricas["fallidas"]) == (3, 1)


def test_las_versiones_del_readme_coinciden_con_los_artefactos_guardados():
    versiones, bloque = resumen.bloque_en_readme(README)
    assert bloque is not None, "El README debe tener el bloque RESUMEN-VERSIONES"
    assert len(versiones) == 2 and len(set(versiones)) == 2                    # dos versiones distintas
    for v in versiones:
        assert (FASE2 / "models" / v / "modelo.joblib").is_file(), f"no existe la versión {v} en fase-2/models"
    assert bloque == resumen.generar_bloque(FASE2 / "models", versiones), (
        "La tabla de versiones del README no coincide con los artefactos. Ejecute desde fase-2: "
        "python resumen_versiones.py --actualizar-readme ../fase-4/README.md")


def test_la_version_activa_declarada_es_la_que_uso_el_ensayo():
    declarada = re.search(r"Versión activa de la API:\*\* `([^`]+)`", README)
    assert declarada, "El README debe declarar «**Versión activa de la API:** `vN`»"
    versiones, _ = resumen.bloque_en_readme(README)
    assert declarada.group(1) in versiones                                      # la activa es una de las dos
    usadas = {f["version"] for f in monitoreo.leer_registro(MUESTRA) if f["estado_http"] == 200}
    assert usadas == {declarada.group(1)}                                      # y fue la que predijo en el ensayo