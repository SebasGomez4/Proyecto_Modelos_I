"""Monitoreo de operación (Fase 4): lee el registro de /predict y muestra dos indicadores.

Uso (desde fase-4):
    python monitoreo.py                                   # lee logs/predicciones.csv
    python monitoreo.py --registro logs/predicciones.csv --detalle

Indicadores:
- Total de solicitudes: filas de datos del registro (una por llamada a /predict, rechazadas incluidas).
- Porcentaje de fallos: solicitudes con código HTTP 4xx o 5xx sobre el total, con una cifra decimal
  y coma (p. ej. 33,3 %). Si el registro está vacío: total 0 y «sin solicitudes».

Describe cómo FUNCIONA el servicio, no qué tan acertadas son las predicciones: medir la calidad
predictiva requeriría conocer los precios reales de las viviendas.
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

from registro import CAMPOS

RUTA_POR_DEFECTO = "logs/predicciones.csv"
INICIO_REPORTE = "<!-- REPORTE-MONITOREO:INICIO -->"
FIN_REPORTE = "<!-- REPORTE-MONITOREO:FIN -->"
PATRON_REPORTE = re.compile(re.escape(INICIO_REPORTE) + ".*?" + re.escape(FIN_REPORTE), re.S)


class ErrorRegistro(Exception):
    """El registro no existe o su contenido no tiene el formato esperado."""


def leer_registro(ruta):
    """Filas de datos del registro como diccionarios (estado_http ya convertido a entero).

    Un archivo vacío o con solo el encabezado devuelve []. Una fila con un código que no es un
    número no se ignora en silencio: se informa con su número de línea.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ErrorRegistro(f"No existe el registro {ruta}. El archivo se crea con la primera llamada a "
                            "/predict; si ya hubo llamadas, revise la ruta (--registro).")
    filas = []
    try:
        with open(ruta, encoding="utf-8-sig", newline="") as f:
            lector = csv.DictReader(f)
            if lector.fieldnames is None:                       # archivo de 0 bytes
                return []
            faltan = [c for c in CAMPOS if c not in lector.fieldnames]
            if faltan:
                raise ErrorRegistro(f"El encabezado de {ruta} no tiene las columnas {', '.join(faltan)}; "
                                    f"se esperaba: {','.join(CAMPOS)}.")
            for fila in lector:
                texto = (fila["estado_http"] or "").strip()
                try:
                    fila["estado_http"] = int(texto)
                except ValueError:
                    raise ErrorRegistro(f"Línea {lector.line_num} del registro: estado_http inválido "
                                        f"({texto!r}); se esperaba un código HTTP numérico.") from None
                filas.append(fila)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        raise ErrorRegistro(f"No se pudo leer {ruta}: {e}") from e
    return filas


def es_fallida(codigo):
    """Fallida = respuesta HTTP 4xx o 5xx."""
    return 400 <= codigo <= 599


def calcular_metricas(filas):
    """Total de solicitudes, cuántas fallaron y el porcentaje (None si no hay solicitudes)."""
    total = len(filas)
    fallidas = sum(es_fallida(f["estado_http"]) for f in filas)
    return {"total": total, "fallidas": fallidas,
            "porcentaje_fallos": None if total == 0 else fallidas / total * 100}


def formatear_porcentaje(porcentaje):
    """33,3 % (una cifra decimal y coma) o «sin solicitudes» si el registro está vacío."""
    if porcentaje is None:
        return "sin solicitudes"
    return f"{porcentaje:.1f}".replace(".", ",") + " %"


def detalle(filas):
    """Solicitudes por código HTTP y por versión del modelo (vacío = no se usó ningún modelo)."""
    return (Counter(f["estado_http"] for f in filas),
            Counter(f["version"] or "(sin versión)" for f in filas))


def generar_reporte(filas):
    """Bloque Markdown del reporte (tabla de solicitudes e indicadores), generado SOLO desde el registro."""
    m = calcular_metricas(filas)
    lineas = [INICIO_REPORTE, "",
              "| # | Fecha y hora (UTC) | Código HTTP | Versión | Predicción (USD) |", "|---|---|---|---|---|"]
    for i, f in enumerate(filas, start=1):
        lineas.append(f"| {i} | {f['fecha_hora']} | {f['estado_http']} | {f['version'] or '—'} "
                      f"| {f['prediccion'] or '—'} |")
    lineas += ["", f"- **Total de solicitudes:** {m['total']}",
               f"- **Solicitudes fallidas (HTTP 4xx o 5xx):** {m['fallidas']}",
               f"- **Porcentaje de fallos:** {formatear_porcentaje(m['porcentaje_fallos'])}", "", FIN_REPORTE]
    return "\n".join(lineas)


def actualizar_readme(ruta_readme, filas):
    """Reescribe solo el bloque marcado del README con las cifras reales del registro."""
    ruta = Path(ruta_readme)
    texto = ruta.read_text(encoding="utf-8")
    if not PATRON_REPORTE.search(texto):
        raise ErrorRegistro(f"{ruta} no tiene el bloque {INICIO_REPORTE} ... {FIN_REPORTE}.")
    nuevo = generar_reporte(filas)
    ruta.write_text(PATRON_REPORTE.sub(lambda _: nuevo, texto), encoding="utf-8")
    return nuevo


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Muestra el total de solicitudes y el porcentaje de fallos "
                                            "del registro de /predict.")
    p.add_argument("--registro", default=RUTA_POR_DEFECTO, help="CSV del registro (por defecto logs/predicciones.csv)")
    p.add_argument("--detalle", action="store_true", help="agrega el conteo por código HTTP y por versión")
    p.add_argument("--markdown", action="store_true", help="imprime el bloque Markdown del reporte para el README")
    p.add_argument("--actualizar-readme", metavar="README",
                   help="reescribe en ese README el bloque marcado con REPORTE-MONITOREO")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        filas = leer_registro(args.registro)
    except ErrorRegistro as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.markdown or args.actualizar_readme:
        try:
            if args.actualizar_readme:
                actualizar_readme(args.actualizar_readme, filas)
                print(f"README actualizado: {args.actualizar_readme}", file=sys.stderr)
            else:
                print(generar_reporte(filas))
        except ErrorRegistro as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0

    m = calcular_metricas(filas)
    print(f"Registro: {args.registro}")
    print(f"Total de solicitudes: {m['total']}")
    print(f"Solicitudes fallidas (HTTP 4xx o 5xx): {m['fallidas']}")
    print(f"Porcentaje de fallos: {formatear_porcentaje(m['porcentaje_fallos'])}")
    if args.detalle and filas:
        por_codigo, por_version = detalle(filas)
        print("\nPor código HTTP:  " + ", ".join(f"{c}: {n}" for c, n in sorted(por_codigo.items())))
        print("Por versión:      " + ", ".join(f"{v}: {n}" for v, n in sorted(por_version.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())