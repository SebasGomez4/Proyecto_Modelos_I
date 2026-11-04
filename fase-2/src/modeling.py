"""Construcción del modelo, del modelo base y de las métricas de evaluación."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.preprocessing import make_preprocessor


def construir_modelo(X, semilla=42, max_iter=300, learning_rate=0.05):
    """Pipeline completo (preprocesamiento + Gradient Boosting) que entrena sobre log1p(precio)
    y devuelve dólares. Al guardarlo se conservan las reglas aprendidas del entrenamiento."""
    pipe = Pipeline([
        ("preprocesamiento", make_preprocessor(X)),
        ("regresor", HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate,
                                                   random_state=semilla)),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def construir_baseline():
    """Modelo base: predice siempre la mediana del precio de entrenamiento (como en la Fase 1)."""
    return DummyRegressor(strategy="median")


def evaluar(y_real, y_pred):
    """MAE y RMSE en USD, R² y RMSLE (error en escala logarítmica; NO es un porcentaje)."""
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "R2": float(r2_score(y_real, y_pred)),
        "RMSLE": float(np.sqrt(mean_squared_error(np.log1p(y_real), np.log1p(np.clip(y_pred, 0, None))))),
    }


def validacion_cruzada(estimador, X, y, folds, semilla):
    """Validación cruzada sobre el conjunto de ENTRENAMIENTO. cross_validate clona el estimador y
    lo reajusta (preprocesamiento incluido) con las filas de entrenamiento de cada fold."""
    scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
    r = cross_validate(estimador, X, y, cv=KFold(folds, shuffle=True, random_state=semilla), scoring=scoring)
    salida = {"folds": folds}
    for k in scoring:
        v = r[f"test_{k}"] * (1 if k == "R2" else -1)
        salida[k] = float(v.mean())
        salida[k + "_std"] = float(v.std())
    return salida