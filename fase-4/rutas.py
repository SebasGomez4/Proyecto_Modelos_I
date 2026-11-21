"""Hace importable el código reutilizado de la Fase 2 (fase-2/src) sin copiarlo.

Local: el repositorio tiene fase-2/ junto a fase-3/, así que se agrega fase-2 al sys.path.
Docker: la imagen copia fase-2/src a /app/src; no existe /fase-2 y no hace falta nada.
Basta con importar este módulo antes de from src import ....
"""
import sys
from pathlib import Path

RAIZ_FASE2 = Path(__file__).resolve().parents[1] / "fase-2"

if RAIZ_FASE2.is_dir() and str(RAIZ_FASE2) not in sys.path:
    sys.path.insert(0, str(RAIZ_FASE2))