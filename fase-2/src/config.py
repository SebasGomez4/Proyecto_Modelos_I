"""Constantes del proyecto (problema de precios de viviendas, Ames Housing)."""

TARGET_DEFECTO = "SalePrice"
ID_COLUMNS_DEFECTO = ["Order", "PID", "Id"]   # identificadores: no describen la vivienda

FUENTE_DATOS = "Kaggle aecincode_houseprices (basado en Ames Housing Dataset, De Cock 2011)"
METRICA_PRINCIPAL = "MAE (USD)"
ALGORITMO = "HistGradientBoostingRegressor sobre log1p(SalePrice)"

# Alcance de generalización declarado (retroalimentación de la Fase 1)
ALCANCE = ("Estimación de precios de viviendas del mismo mercado (Ames, 2006-2010) y barrios conocidos, "
           "con partición aleatoria. No se evaluó en barrios nuevos ni en períodos futuros.")
LIMITACIONES = [
    "Datos de una sola ciudad y época.",
    "Incluye variables de la transacción (Sale Type, Sale Condition, Mo Sold, Yr Sold): solo son "
    "válidas si se conocen al momento de predecir; para una tasación previa a la venta habría que retirarlas.",
    "Sin ajuste fino de hiperparámetros; valores atípicos no tratados.",
]

MIN_FILAS = 10   # mínimo de filas para poder separar y entrenar