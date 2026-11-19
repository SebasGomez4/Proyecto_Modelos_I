"""Resumen de versiones guardadas, generado desde metrics.json y metadata.json.

Evita que el README documente cifras distintas a las del artefacto publicado: la tabla del
README se genera con este módulo y una prueba comprueba que ambas coinciden.

Bloque en el README (lo reescribe `python resumen_versiones.py --actualizar-readme README.md`):

    <!-- RESUMEN-VERSIONES:INICIO versiones=v1,v2 -->
    ...contenido generado...
    <!-- RESUMEN-VERSIONES:FIN -->
"""
import hashlib
import json
import re
from pathlib import Path

from src.errors import ErrorAplicacion
from src.registry import ARCHIVO_METADATA, ARCHIVO_METRICAS, ARCHIVO_MODELO

PATRON_BLOQUE = re.compile(
    r"<!-- RESUMEN-VERSIONES:INICIO versiones=([^ ]+) -->.*?<!-- RESUMEN-VERSIONES:FIN -->", re.S)


def _usd(x):
    return f"{x:,.0f}".replace(",", ".")


def _num(x, decimales=3):
    return f"{x:.{decimales}f}".replace(".", ",")


def sha256_archivo(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def cargar_versiones(models_dir, versiones):
    """Lee metadatos, métricas y SHA-256 del modelo de cada versión."""
    registros = []
    for v in versiones:
        carpeta = Path(models_dir) / v
        try:
            meta = json.loads((carpeta / ARCHIVO_METADATA).read_text(encoding="utf-8"))
            metricas = json.loads((carpeta / ARCHIVO_METRICAS).read_text(encoding="utf-8"))
            sha = sha256_archivo(carpeta / ARCHIVO_MODELO)
        except (OSError, json.JSONDecodeError) as e:
            raise ErrorAplicacion(f"No se pudo leer la versión '{v}' en {carpeta}: {e}") from e
        registros.append({"version": v, "meta": meta, "metricas": metricas, "sha": sha})
    return registros


def _commit_corto(commit):
    base = commit.split("+")[0]
    return (base[:7] if base != "desconocido" else base) + ("+cambios" if "+" in commit else "")


def _particion(meta):
    s = meta["separacion"]
    return (meta["semilla"], s["test_size"], s["n_train"], s["n_test"], meta["datos"]["sha256"])


def generar_markdown(registros):
    """Dos tablas (trazabilidad y métricas) y notas automáticas sobre duplicados y particiones."""
    if not registros:
        raise ErrorAplicacion("No hay versiones para resumir.")
    lineas = ["**Trazabilidad de cada versión**", "",
              "| Versión | Creada (UTC) | Commit del código | Python | scikit-learn | Semilla "
              "| Iteraciones | Tasa de aprendizaje | SHA-256 del modelo |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in registros:
        m, p = r["meta"], r["meta"]["parametros"]
        lineas.append(
            f"| `{r['version']}` | {m['fecha_creacion'].replace('T', ' ')[:19]} | {_commit_corto(m['commit'])} "
            f"| {m['entorno']['python']} | {m['entorno']['scikit-learn']} | {m['semilla']} "
            f"| {p['max_iter']} | {_num(p['learning_rate'], 2)} | `{r['sha'][:12]}` |")

    lineas += ["", "**Métricas de cada versión** (MAE y RMSE en USD)", "",
               "| Versión | Partición (semilla; train/test) | MAE prueba | RMSE prueba | R² prueba "
               "| RMSLE prueba | MAE validación cruzada (± desv.) | MAE modelo base (prueba) "
               "| Reducción del MAE vs base |",
               "|---|---|---|---|---|---|---|---|---|"]
    for r in registros:
        t = r["metricas"]["modelo"]["test"]
        base_mae = r["metricas"]["baseline"]["test"]["MAE"]
        cv = r["metricas"]["modelo"].get("cv_train")
        cv_txt = f"{_usd(cv['MAE'])} ± {_usd(cv['MAE_std'])}" if cv else "—"
        s = r["meta"]["separacion"]
        lineas.append(
            f"| `{r['version']}` | {r['meta']['semilla']}; {s['n_train']}/{s['n_test']} | {_usd(t['MAE'])} "
            f"| {_usd(t['RMSE'])} | {_num(t['R2'])} | {_num(t['RMSLE'])} | {cv_txt} "
            f"| {_usd(base_mae)} | {_num((1 - t['MAE'] / base_mae) * 100, 1)} % |")

    lineas.append("")
    por_sha = {}
    for r in registros:
        por_sha.setdefault(r["sha"], []).append(r["version"])
    for grupo in por_sha.values():
        if len(grupo) > 1:
            lineas.append(f"- **Artefactos idénticos:** {', '.join(f'`{v}`' for v in grupo)} comparten el mismo "
                          "SHA-256: mismos datos, semilla y parámetros, por lo que sus métricas son idénticas "
                          "y no representan una mejora.")
    if len({_particion(r["meta"]) for r in registros}) > 1:
        lineas.append("- **Atención:** las versiones no usan la misma partición o los mismos datos; "
                      "sus métricas no son directamente comparables.")
    else:
        sem, _, ntr, nte, _ = _particion(registros[0]["meta"])
        lineas.append(f"- Todas las versiones usan los mismos datos y la misma partición (semilla {sem}, "
                      f"{ntr}/{nte}): sus métricas son comparables.")
    return "\n".join(lineas)


def generar_bloque(models_dir, versiones):
    """Bloque completo (con marcadores) listo para pegar o comparar con el README."""
    cuerpo = generar_markdown(cargar_versiones(models_dir, versiones))
    return (f"<!-- RESUMEN-VERSIONES:INICIO versiones={','.join(versiones)} -->\n{cuerpo}\n"
            "<!-- RESUMEN-VERSIONES:FIN -->")


def bloque_en_readme(texto):
    """Devuelve (versiones, bloque_actual) o (None, None) si el README no tiene los marcadores."""
    m = PATRON_BLOQUE.search(texto)
    return (m.group(1).split(","), m.group(0)) if m else (None, None)


def actualizar_readme(ruta_readme, models_dir, versiones=None):
    """Reescribe solo el bloque marcado del README con las cifras reales de los artefactos."""
    ruta = Path(ruta_readme)
    texto = ruta.read_text(encoding="utf-8")
    en_readme, actual = bloque_en_readme(texto)
    if actual is None:
        raise ErrorAplicacion(f"{ruta} no tiene el bloque <!-- RESUMEN-VERSIONES:INICIO versiones=... -->.")
    nuevo = generar_bloque(models_dir, versiones or en_readme)
    ruta.write_text(texto.replace(actual, nuevo), encoding="utf-8")
    return nuevo