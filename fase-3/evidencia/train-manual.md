curl -s -i -X POST http://localhost:8000/train -H "Content-Type: application/json" --data-binary @outputs/train_body.json
HTTP/1.1 201 Created
date: Thu, 19 Nov 2026 22:54:53 GMT
server: uvicorn
content-length: 553
content-type: application/json

{"version":"api-20261119T225453Z","estado":"guardada","ubicacion":"models/api-20261119T225453Z","archivos":["modelo.joblib","metrics.json","metadata.json"],"parametros":{"seed":42,"max_iter":500,"learning_rate":0.03,"cv_folds":5},"metricas_prueba":{"MAE":14937.5853,"RMSE":22722.7969,"R2":0.9167,"RMSLE":0.1378},"metricas_modelo_base_prueba":{"MAE":53976.9176,"RMSE":80587.9699,"R2":-0.0473,"RMSLE":0.4188},"version_activa":"v1","nota":"/predict sigue usando 'v1'. Para usar 'api-20261119T225453Z' inicie la API con MODEL_VERSION=api-20261119T225453Z."}


curl -s http://localhost:8000/health
{"estado":"ok","mensaje":"API en ejecución","modelo":{"version":"v1","disponible":true},"versiones_disponibles":["api-20261119T225453Z","v-docker","v1","v2"]}


curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @ejemplos/registro_valido.json    # PowerShell: curl.exe ... --data-binary "@ejemplos/registro_valido.json"
{"prediccion":168104.09,"version":"v1"}


python resumen_versiones.py --versiones v1 api-20261119T225453Z
<!-- RESUMEN-VERSIONES:INICIO versiones=v1,api-20261119T225453Z -->
**Trazabilidad de cada versión**

| Versión | Creada (UTC) | Commit del código | Python | scikit-learn | Semilla | Iteraciones | Tasa de aprendizaje | SHA-256 del modelo |
|---|---|---|---|---|---|---|---|---|
| `v1` | 2026-11-04 22:38:45 | 9951527+cambios | 3.11.9 | 1.8.0 | 42 | 300 | 0,05 | `355630f55771` |
| `api-20261119T225453Z` | 2026-11-19 22:55:55 | c5a5681 | 3.12.10 | 1.8.0 | 42 | 500 | 0,03 | `a74d8188870a` |

**Métricas de cada versión** (MAE y RMSE en USD)

| Versión | Partición (semilla; train/test) | MAE prueba | RMSE prueba | R² prueba | RMSLE prueba | MAE validación cruzada (± desv.) | MAE modelo base (prueba) | Reducción del MAE vs base |
|---|---|---|---|---|---|---|---|---|
| `v1` | 42; 1406/352 | 14.769 | 22.461 | 0,919 | 0,136 | 15.558 ± 1.224 | 53.977 | 72,6 % |
| `api-20261119T225453Z` | 42; 1406/352 | 14.938 | 22.723 | 0,917 | 0,138 | 15.675 ± 1.181 | 53.977 | 72,3 % |

- Todas las versiones usan los mismos datos y la misma partición (semilla 42, 1406/352): sus métricas son comparables.
<!-- RESUMEN-VERSIONES:FIN -->