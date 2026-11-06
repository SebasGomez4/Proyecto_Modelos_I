## 1. Construcción de la imagen
```
docker build -t modelos-fase2:1.0 .
[+] Building 163.2s (11/11) FINISHED                                             docker:desktop-linux
 => [internal] load build definition from Dockerfile                                             0.0s
 => => transferring dockerfile: 300B                                                             0.0s
 => [internal] load metadata for docker.io/library/python:3.12-slim                             17.6s
 => [internal] load .dockerignore                                                                0.0s
 => => transferring context: 155B                                                                0.0s
 => [1/6] FROM docker.io/library/python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f  52.6s
 => => resolve docker.io/library/python:3.12-slim@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3  0.0s
 => => sha256:05e0eee5572e9c9f50408a2132590133b830a668a1cf2ff2b150f19c57707e2b 248B / 248B       0.7s
 => => sha256:2dc99d2d0a39fce31e37615eb4c932e16dcc4c2b0ce9993714280081b5dbe0 12.12MB / 12.12MB  37.2s
 => => sha256:68bced9e0afa4479df17c3646e27c7c7e9ce9ba70c0e6c73b6826487b89037d9 1.29MB / 1.29MB  13.5s
 => => sha256:37fb86c2a767b88d500c6872195f88078f64765fe0d2f9cc375cc2efbd4c0a 29.86MB / 29.86MB  51.6s
 => => extracting sha256:37fb86c2a767b88d500c6872195f88078f64765fe0d2f9cc375cc2efbd4c0a41        0.5s
 => => extracting sha256:68bced9e0afa4479df17c3646e27c7c7e9ce9ba70c0e6c73b6826487b89037d9        0.1s
 => => extracting sha256:2dc99d2d0a39fce31e37615eb4c932e16dcc4c2b0ce9993714280081b5dbe085        0.2s
 => => extracting sha256:05e0eee5572e9c9f50408a2132590133b830a668a1cf2ff2b150f19c57707e2b        0.0s
 => [internal] load build context                                                                0.0s
 => => transferring context: 82.10kB                                                             0.0s
 => [2/6] WORKDIR /app                                                                           0.1s
 => [3/6] COPY requirements.txt .                                                                0.0s
 => [4/6] RUN pip install --no-cache-dir -r requirements.txt                                    82.2s
 => [5/6] COPY src/ src/                                                                         0.1s
 => [6/6] COPY src/train.py src/predict.py ./                                                    0.0s
 => exporting to image                                                                          10.4s
 => => exporting layers                                                                          8.4s
 => => exporting manifest sha256:0776377b73ecd3c2f576fd26301b7f8d994a82750ab33c73ea66de3a864bd9  0.0s
 => => exporting config sha256:f9aeb32d2f8195756b059857ba5584b8926d8ba9456f5e471376eb32a70c4823  0.0s
 => => exporting attestation manifest sha256:e20e53e16b4e1bdd44f9a809196dffda49a22b00bb703d0f7b  0.0s
 => => exporting manifest list sha256:cf467f6a53e3a8fcc25c05795d60fa6583cf79ffcc1db09757e1ad176  0.0s
 => => naming to docker.io/library/modelos-fase2:1.0                                             0.0s
 => => unpacking to docker.io/library/modelos-fase2:1.0                                          1.9s

 docker images modelos-fase2
                                                                                  i Info →   U  In Use
IMAGE               ID             DISK USAGE   CONTENT SIZE   EXTRA
modelos-fase2:1.0   cf467f6a53e3        647MB          148MB
```

## 2. Entrenamiento en el contenedor (versión v-docker)
```
docker run --rm -e GIT_COMMIT="$(git rev-parse HEAD)" \
  -v "//$(pwd)/data:/app/data:ro" \
  -v "//$(pwd)/models:/app/models" \
  modelos-fase2:1.0 python train.py --data data/train.csv --target SalePrice --output-dir models --version v-docker --seed 42
Versión:     v-docker
Datos:       1406 train / 352 test
Baseline:    MAE = 53,977  R2 = -0.047
Modelo:      MAE = 14,769  R2 = 0.919  RMSLE = 0.136
Reducción del MAE frente al baseline (test): 72.6%
Archivos en: models/v-docker  (modelo.joblib, metrics.json, metadata.json)
```

## 3. Predicción en el contenedor
```
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  modelos-fase2:1.0 python predict.py \
  --model models/v-docker/modelo.joblib \
  --input data/ejemplo_prediccion.csv \
  --output outputs/predicciones_docker.csv
Versión usada:   v-docker
Filas predichas: 5
Salida:          outputs/predicciones_docker.csv
```
## 4. Persistencia fuera del contenedor

```
docker ps -a
CONTAINER ID   IMAGE                COMMAND                  CREATED        STATUS                    PORTS                              NAMES
dd7d626a0ddb   postgres:16-alpine   "docker-entrypoint.s…"   5 weeks ago    Up 13 minutes (healthy)   0.0.0.0:5432->5432/tcp, [::]:5432->5432/tcp   chatbot-uae-postgres
6be67ec30890   buscar-datos         "python buscar_datos…"   3 months ago   Exited (0) 3 months ago                              kind_ardinghelli
dc226e6b590a   buscar-datos         "python buscar_datos…"   3 months ago   Exited (0) 3 months ago                              hardcore_sinoussi
ls models/v-docker
metadata.json  metrics.json  modelo.joblib
head -3 outputs/predicciones_docker.csv
fila,Order,PID,Id,prediccion
1,1358,903427090,0,168104.09
2,2368,527450460,1,104664.78
```
## 5. Entrada inválida (columna Lot Area ausente)
```
python -c "import pandas as pd; pd.read_csv('data/ejemplo_prediccion.csv').drop(columns=['Lot Area']).to_csv('outputs/invalido.csv', index=False)"
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  modelos-fase2:1.0 python predict.py \
  --model models/v-docker/modelo.joblib \
  --input outputs/invalido.csv \
  --output outputs/no_debe_existir.csv
Error: Faltan 1 columnas obligatorias en la entrada: Lot Area
echo "código de salida: $?"
código de salida: 1
ls outputs
invalido.csv  predicciones_docker.csv
```

## 6. Versión repetida
```
MSYS_NO_PATHCONV=1 docker run --rm -e GIT_COMMIT="$(git rev-parse HEAD)"   -v "$PWD/data:/app/data:ro"   -v "$PWD/models:/app/models"   modelos-fase2:1.0 python train.py --data data/train.csv --target SalePrice --output-dir models --version v-docker --seed 42
Error: ya existe la versión 'v-docker' en models.
```