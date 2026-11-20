"""Cliente de prueba de la API: consulta /health y envía un registro real a /predict.

Solo usa la biblioteca estándar de Python (no hay que instalar nada). Con la API en marcha
(uvicorn local o contenedor Docker), desde la carpeta fase-3:

    python cliente_prueba.py
    python cliente_prueba.py --url http://localhost:8000 --registro ejemplos/registro_valido.json
    python cliente_prueba.py --sin-campo "Lot Area"          # muestra el rechazo por campo ausente (422)
    python cliente_prueba.py --entrenar --version-train v-docker --cv-folds 0

Termina con código 0 si todas las respuestas fueron las esperadas y con 1 si alguna no lo fue.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent


def llamar(url, metodo="GET", cuerpo=None, timeout=600):
    """Devuelve (código HTTP, JSON de la respuesta). Un 4xx/5xx no es una excepción: es una respuesta."""
    datos = None if cuerpo is None else json.dumps(cuerpo).encode("utf-8")
    peticion = urllib.request.Request(url, data=datos, method=metodo, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(peticion, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        texto = e.read().decode("utf-8")
        return e.code, (json.loads(texto) if texto else {})
    except urllib.error.URLError as e:
        raise SystemExit(f"No se pudo conectar con {url}: {e.reason}. ¿Está la API en ejecución?")


def mostrar(titulo, codigo, cuerpo):
    print(f"\n== {titulo} -> HTTP {codigo}")
    print(json.dumps(cuerpo, indent=2, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(description="Cliente de prueba de la API de precios de viviendas.")
    p.add_argument("--url", default="http://localhost:8000", help="dirección base de la API")
    p.add_argument("--registro", default=str(RAIZ / "ejemplos" / "registro_valido.json"),
                   help="registro JSON con las variables predictoras")
    p.add_argument("--sin-campo", metavar="CAMPO", help="quita este campo para provocar un rechazo 422")
    p.add_argument("--entrenar", action="store_true", help="también llama a POST /train")
    p.add_argument("--version-train", help="identificador de la versión que crea /train (opcional)")
    p.add_argument("--max-iter", type=int, help="iteraciones para /train (opcional)")
    p.add_argument("--cv-folds", type=int, help="folds de validación cruzada para /train (0 = sin)")
    args = p.parse_args(argv)

    base = args.url.rstrip("/")
    registro = json.loads(Path(args.registro).read_text(encoding="utf-8"))
    fallos = []

    codigo, cuerpo = llamar(f"{base}/health")
    mostrar("GET /health", codigo, cuerpo)
    if codigo != 200:
        fallos.append("/health no respondió 200")

    codigo, cuerpo = llamar(f"{base}/predict", "POST", registro)
    mostrar(f"POST /predict ({len(registro)} variables)", codigo, cuerpo)
    if codigo == 200:
        print(f"\nPredicción: {cuerpo['prediccion']:,.2f} USD con la versión {cuerpo['version']}")
    elif codigo != 503:
        fallos.append("/predict con un registro válido no respondió 200")
    else:
        fallos.append("/predict respondió 503: el modelo configurado no está disponible")

    if args.sin_campo:
        incompleto = {k: v for k, v in registro.items() if k != args.sin_campo}
        codigo, cuerpo = llamar(f"{base}/predict", "POST", incompleto)
        mostrar(f"POST /predict sin '{args.sin_campo}'", codigo, cuerpo)
        if codigo != 422:
            fallos.append(f"se esperaba 422 al faltar '{args.sin_campo}'")

    if args.entrenar:
        parametros = {k: v for k, v in {"version": args.version_train, "max_iter": args.max_iter,
                                        "cv_folds": args.cv_folds}.items() if v is not None}
        codigo, cuerpo = llamar(f"{base}/train", "POST", parametros)
        mostrar(f"POST /train {parametros or '(cuerpo vacío)'}", codigo, cuerpo)
        if codigo != 201:
            fallos.append("/train no respondió 201")

    if fallos:
        print("\nRESULTADO: " + "; ".join(fallos), file=sys.stderr)
        return 1
    print("\nRESULTADO: todas las respuestas fueron las esperadas")
    return 0


if __name__ == "__main__":
    sys.exit(main())