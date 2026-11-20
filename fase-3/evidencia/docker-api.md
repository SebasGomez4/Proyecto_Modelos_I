 docker build -f fase-3/Dockerfile -t modelos-api:1.0 .
[+] Building 2.0s (10/10) FINISHED                                                                                        docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                                      0.0s
 => => transferring dockerfile: 1.37kB                                                                                                    0.0s
 => [internal] load metadata for docker.io/library/python:3.12-slim                                                                       1.6s
 => [1/6] FROM docker.io/library/python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b                 0.1s
 => => resolve docker.io/library/python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b                 0.1s
 => [internal] load build context                                                                                                         0.0s
 => => transferring context: 893B                                                                                                         0.0s
 => CACHED [2/6] WORKDIR /app                                                                                                             0.0s
 => CACHED [3/6] COPY fase-3/requirements.txt .                                                                                           0.0s
 => CACHED [4/6] RUN pip install --no-cache-dir -r requirements.txt                                                                       0.0s
 => CACHED [5/6] COPY fase-2/src/ src/                                                                                                    0.0s
 => CACHED [6/6] COPY fase-3/api.py fase-3/ajustes.py fase-3/entrada.py fase-3/entrenamiento.py fase-3/servicio_modelo.py fase-3/rutas.p  0.0s
 => exporting to image                                                                                                                    0.2s
 => => exporting layers                                                                                                                   0.0s
 => => exporting manifest sha256:f4f39ca96b0164ecea42008923959fec2046df5a8da542f01ca5465edd088591                                         0.0s
 => => exporting config sha256:9296a4c7dad3318042e7493bb9a3618529e46cbb724c53ebba59a27260747f9d                                           0.0s
 => => exporting attestation manifest sha256:1ed551ef5f245ef5c2fcbe5ad60aa2657d9492df9fe327be5f217e6ed0f3dfc8                             0.0s
 => => exporting manifest list sha256:bb4f9d5a00e4e3e0d1a2431c26b4bc4311ac16c84c9ddafe06f0f951a61790a7                                    0.0s
 => => naming to docker.io/library/modelos-api:1.0                                                                                        0.0s
 => => unpacking to docker.io/library/modelos-api:1.0

docker run --rm -p 8000:8000 -e GIT_COMMIT="$(git rev-parse HEAD)" -v "$PWD/fase-2/data:/app/data:ro" -v "$PWD/fase-2/models:/app/models" modelos-api:1.0
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:37458 - "GET /health HTTP/1.1" 200 OK
INFO:     127.0.0.1:33788 - "GET /health HTTP/1.1" 200 OK


curl -s http://localhost:8000/health
{"estado":"ok","mensaje":"API en ejecución","modelo":{"version":"v1","disponible":true},"versiones_disponibles":["api-20261119T225453Z","v-docker","v1","v2"]}


python cliente_prueba.py --sin-campo "Lot Area"

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
    "v-docker",
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


python cliente_prueba.py --entrenar --version-train v-docker --cv-folds 0

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
    "v-docker",
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

== POST /train {'version': 'v-docker', 'cv_folds': 0} -> HTTP 409
{
  "detail": "La versión 'v-docker' ya existe en C:\\Users\\Sebastian\\OneDrive\\PROGRAMACION\\modelosI\\Proyecto_Modelos_I\\fase-2\\models\\v-docker. Elija otro identificador: las versiones guardadas no se sobrescriben."
}

RESULTADO: /train no respondió 201


ls ../fase-2/models
api-20261119T225453Z/  v1/  v2/  v-docker/


docker ps -a
CONTAINER ID   IMAGE                COMMAND                  CREATED        STATUS                    PORTS     NAMES
dd7d626a0ddb   postgres:16-alpine   "docker-entrypoint.s…"   7 weeks ago    Up 33 minutes (healthy)   0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp   chatbot-uae-postgres
6be67ec30890   buscar-datos         "python buscar_datos…"   3 months ago   Exited (0) 3 months ago     kind_ardinghelli
dc226e6b590a   buscar-datos         "python buscar_datos…"   3 months ago   Exited (0) 3 months ago     hardcore_sinoussi