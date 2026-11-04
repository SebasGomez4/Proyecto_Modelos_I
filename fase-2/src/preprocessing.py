"""Preprocesamiento reutilizable por train.py y predict.py (se guarda dentro del modelo).

Corrección de la Fase 1: se separa la AUSENCIA ESTRUCTURAL ("la vivienda no tiene sótano")
de la INFORMACIÓN DESCONOCIDA (dato que debería existir y falta):

- Variable vacía y su variable de referencia == 0  -> "None"        (ausencia real)
- Variable vacía y la referencia != 0 o también vacía -> "Desconocido" (dato faltante)
- Variables sin referencia disponible (Alley, Fence, Misc Feature): vacío = "None"; es una
  limitación documentada, porque no se puede distinguir de un dato perdido.

La regla usa solo valores de la propia fila, así que no aprende nada del conjunto de datos
(no hay fuga). Lo que sí aprende (medianas, categorías) se ajusta solo con entrenamiento
porque todo va dentro de un Pipeline.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# columna con posible "no aplica" -> variable numérica de referencia (None si no hay)
REGLAS_AUSENCIA = {
    "Bsmt Qual": "Total Bsmt SF", "Bsmt Cond": "Total Bsmt SF", "Bsmt Exposure": "Total Bsmt SF",
    "BsmtFin Type 1": "Total Bsmt SF", "BsmtFin Type 2": "Total Bsmt SF",
    "Garage Type": "Garage Area", "Garage Finish": "Garage Area",
    "Garage Qual": "Garage Area", "Garage Cond": "Garage Area",
    "Fireplace Qu": "Fireplaces", "Pool QC": "Pool Area", "Mas Vnr Type": "Mas Vnr Area",
    "Alley": None, "Fence": None, "Misc Feature": None,
}
# categóricas por naturaleza aunque en una muestra pequeña lleguen vacías (dtype float)
CATEGORICAS_CONOCIDAS = set(REGLAS_AUSENCIA)

NIVELES_CALIDAD = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
ORDINALES = {
    "calidad": (["Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC", "Kitchen Qual",
                 "Fireplace Qu", "Garage Qual", "Garage Cond", "Pool QC"], NIVELES_CALIDAD),
    "exposicion": (["Bsmt Exposure"], ["None", "No", "Mn", "Av", "Gd"]),
    "bsmt_fin": (["BsmtFin Type 1", "BsmtFin Type 2"], ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]),
    "garage_fin": (["Garage Finish"], ["None", "Unf", "RFn", "Fin"]),
}
CODIGOS_COMO_CATEGORIA = ["MS SubClass"]   # son códigos, no cantidades


class AusenciaEstructural(BaseEstimator, TransformerMixin):
    """Reemplaza vacíos por "None" (ausencia real) o "Desconocido" según la variable de referencia."""

    def __init__(self, reglas=None):
        self.reglas = reglas

    def fit(self, X, y=None):
        self.reglas_ = dict(REGLAS_AUSENCIA if self.reglas is None else self.reglas)
        return self

    def transform(self, X):
        X = X.copy()
        for col, ref in self.reglas_.items():
            if col not in X.columns:
                continue
            vacio = X[col].isna()
            if not vacio.any():
                continue
            if ref is None:
                ausente = pd.Series(True, index=X.index)
            elif ref in X.columns:
                ausente = pd.to_numeric(X[ref], errors="coerce") == 0   # referencia vacía -> False
            else:
                ausente = pd.Series(False, index=X.index)
            serie = X[col].astype(object)
            serie[vacio & ausente] = "None"
            serie[vacio & ~ausente] = "Desconocido"
            X[col] = serie
        return X


def definir_grupos(X):
    """Agrupa las columnas por tratamiento (usa solo nombres y tipos, no valores)."""
    columnas = list(X.columns)
    ordinales = {n: ([c for c in cols if c in columnas], niv) for n, (cols, niv) in ORDINALES.items()}
    ordinales = {n: v for n, v in ordinales.items() if v[0]}
    ord_cols = [c for cols, _ in ordinales.values() for c in cols]
    sub_cols = [c for c in CODIGOS_COMO_CATEGORIA if c in columnas]
    num_cols = [c for c in columnas
                if pd.api.types.is_numeric_dtype(X[c]) and c not in sub_cols + ord_cols
                and c not in CATEGORICAS_CONOCIDAS and c != "Garage Yr Blt"] \
        + [c for c in ["Garage Yr Blt"] if c in columnas]
    cat_cols = [c for c in columnas if c not in num_cols + ord_cols + sub_cols]
    return num_cols, ordinales, cat_cols, sub_cols


def make_preprocessor(X):
    """Pipeline de preprocesamiento sin ajustar. X solo define qué columnas hay y de qué tipo."""
    num_cols, ordinales, cat_cols, sub_cols = definir_grupos(X)
    transformers = [("num", SimpleImputer(strategy="median", add_indicator=True), num_cols)]
    for nombre, (cols, niveles) in ordinales.items():
        transformers.append((nombre, Pipeline([
            ("faltante", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            # "Desconocido" o categorías nuevas -> NaN -> mediana (con indicador)
            ("ord", OrdinalEncoder(categories=[niveles] * len(cols),
                                   handle_unknown="use_encoded_value", unknown_value=float("nan"))),
            ("imp", SimpleImputer(strategy="median", add_indicator=True))]), cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("faltante", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))
    if sub_cols:
        transformers.append(("subclass", OneHotEncoder(handle_unknown="ignore"), sub_cols))
    columnas = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    return Pipeline([("ausencia", AusenciaEstructural()), ("columnas", columnas)])