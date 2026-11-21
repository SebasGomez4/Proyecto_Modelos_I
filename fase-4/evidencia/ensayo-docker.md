``` bash
$ python cliente_prueba.py

== GET /health -> HTTP 200
{
  "estado": "ok",
  "mensaje": "API en ejecución",
  "modelo": {
    "version": "v1",
    "disponible": true
  },
  "versiones_disponibles": [
    "api-20261119T225453Z",
    "v1",
    "v2"
  ]
}

== POST /predict (79 variables) -> HTTP 200
{
  "prediccion": 168104.09,
  "version": "v1"
}

Predicción: 168,104.09 USD con la versión v1

RESULTADO: todas las respuestas fueron las esperadas
```

``` bash
$ python cliente_prueba.py --sin-campo "Lot Area"

== GET /health -> HTTP 200
{
  "estado": "ok",
  "mensaje": "API en ejecución",
  "modelo": {
    "version": "v1",
    "disponible": true
  },
  "versiones_disponibles": [
    "api-20261119T225453Z",
    "v1",
    "v2"
  ]
}

== POST /predict (79 variables) -> HTTP 200
{
  "prediccion": 168104.09,
  "version": "v1"
}

Predicción: 168,104.09 USD con la versión v1

== POST /predict sin 'Lot Area' -> HTTP 422
{
  "detail": "Entrada inválida: falta 1 campo obligatorio: Lot Area.",
  "errores": [
    {
      "campo": "Lot Area",
      "problema": "falta",
      "mensaje": "Falta el campo obligatorio 'Lot Area'."
    }
  ]
}

RESULTADO: todas las respuestas fueron las esperadas
```

``` bash
$ cat logs/predicciones.csv
fecha_hora,estado_http,version,prediccion
2026-11-21T19:38:25,200,v1,168104.09
2026-11-21T19:42:25,200,v1,168104.09
2026-11-21T19:42:27,422,v1,
```

``` bash
$ python monitoreo.py --detalle
Registro: logs/predicciones.csv
Total de solicitudes: 3
Solicitudes fallidas (HTTP 4xx o 5xx): 1
Porcentaje de fallos: 33,3 %

Por código HTTP:  200: 2, 422: 1
Por versión:      v1: 3
```

``` bash
$ docker ps -a
CONTAINER ID   IMAGE         COMMAND                  CREATED        STATUS                      PORTS     NAMES
a6ccd47b43d1   mysql:9.7.1   "docker-entrypoint.s…"   2 months ago   Exited (137) 2 months ago             fundamentossist-mysql-1
```