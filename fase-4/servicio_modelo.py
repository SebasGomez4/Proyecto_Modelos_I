"""Acceso al modelo configurado: comprobación ligera, carga con caché y errores comprensibles.

Reutiliza src.registry (Fase 2): la API no reimplementa cómo se guardan ni cargan las versiones.
"""
import threading

import rutas  # noqa: F401
from src import registry
from src.errors import ModeloNoEncontrado


class ServicioModelo:
    def __init__(self, ajustes):
        self.ajustes = ajustes
        self._cache = {}                 # versión -> (modelo, metadata); solo se guardan cargas exitosas
        self._lock = threading.Lock()

    @property
    def version(self):
        return self.ajustes.version

    @property
    def carpeta_activa(self):
        return self.ajustes.models_dir / self.ajustes.version

    def disponible(self):
        """Comprobación barata (sin cargar el modelo): existen modelo.joblib y metadata.json."""
        c = self.carpeta_activa
        return (c / registry.ARCHIVO_MODELO).is_file() and (c / registry.ARCHIVO_METADATA).is_file()

    def versiones_disponibles(self):
        return registry.listar_versiones(self.ajustes.models_dir)

    def obtener(self):
        """(modelo, metadata) de la versión configurada. Lanza ModeloNoEncontrado / ModeloIlegible.

        Una carga fallida no se guarda en caché: si luego aparece la versión (p. ej. tras /train),
        la API se recupera sin reiniciar.
        """
        with self._lock:
            if self.version not in self._cache:
                if not self.disponible():
                    raise ModeloNoEncontrado(
                        f"La versión configurada '{self.version}' no está disponible en "
                        f"{self.ajustes.models_dir} (faltan modelo.joblib o metadata.json).")
                self._cache[self.version] = registry.cargar_version(self.carpeta_activa)
            return self._cache[self.version]