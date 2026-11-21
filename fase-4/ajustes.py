"""Configuración de la API por variables de entorno (sin rutas absolutas ni valores fijos en el código).

| Variable        | Por defecto                       | Significado                                    |
|-----------------|-----------------------------------|------------------------------------------------|
| MODEL_VERSION   | v1                                | versión que usa /predict (¡así se cambia de versión!) |
| MODELS_DIR      | fase-2/models (o ./models)        | carpeta con las versiones guardadas            |
| DATA_PATH       | fase-2/data/train.csv (o ./data/train.csv) | datos etiquetados para /train       |
| TARGET          | SalePrice                         | variable objetivo                              |
| SEED            | 42                                | semilla de /train                              |
| CV_FOLDS        | 5                                 | folds de validación cruzada en /train (0 = sin) |
| MAX_ITER        | 300                               | iteraciones de Gradient Boosting en /train     |
| LEARNING_RATE   | 0.05                              | tasa de aprendizaje en /train                  |
"""
import os
from dataclasses import dataclass
from pathlib import Path

import rutas  # noqa: F401
from src import registry
from src.errors import ErrorAplicacion


class ErrorConfiguracion(ErrorAplicacion):
    """Una variable de entorno tiene un valor inválido."""


def _por_defecto(relativo_fase2, local):
    """fase-2/<ruta> si existe junto a fase-3 (ejecución local); si no, ./<ruta> (contenedor con montajes)."""
    candidata = rutas.RAIZ_FASE2 / relativo_fase2
    return candidata if candidata.exists() else Path(local)


def _numero(env, clave, defecto, tipo):
    texto = env.get(clave)
    if texto is None or texto.strip() == "":
        return defecto
    try:
        return tipo(texto)
    except ValueError:
        raise ErrorConfiguracion(f"{clave}={texto!r} no es un {tipo._name_} válido.") from None


@dataclass(frozen=True)
class Ajustes:
    models_dir: Path
    version: str
    data_path: Path
    target: str = "SalePrice"
    seed: int = 42
    cv_folds: int = 5
    max_iter: int = 300
    learning_rate: float = 0.05

    @classmethod
    def desde_entorno(cls, env=None):
        env = os.environ if env is None else env
        models_dir = Path(env.get("MODELS_DIR") or _por_defecto("models", "models"))
        data_path = Path(env.get("DATA_PATH") or _por_defecto("data/train.csv", "data/train.csv"))
        return cls(
            models_dir=models_dir,
            version=registry.validar_version(env.get("MODEL_VERSION") or "v1"),
            data_path=data_path,
            target=env.get("TARGET") or "SalePrice",
            seed=_numero(env, "SEED", 42, int),
            cv_folds=_numero(env, "CV_FOLDS", 5, int),
            max_iter=_numero(env, "MAX_ITER", 300, int),
            learning_rate=_numero(env, "LEARNING_RATE", 0.05, float),
        )