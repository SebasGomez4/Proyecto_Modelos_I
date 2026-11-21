"""Registro de las llamadas a POST /predict en un archivo CSV (Fase 4).

Una fila por llamada, incluidas las rechazadas por validación:

    fecha_hora,estado_http,version,prediccion
    2026-11-06T10:00:00,200,v1,171698.12
    2026-11-06T10:01:00,422,v1,

- fecha_hora: hora UTC, formato AAAA-MM-DDTHH:MM:SS.
- estado_http: código HTTP devuelto (cualquier 4xx o 5xx cuenta como fallida en el monitoreo).
- version: versión del modelo que validó o predijo; vacía si no se llegó a usar un modelo
  (JSON inválido, versión no disponible o método incorrecto).
- prediccion: valor devuelto; vacío en las respuestas de error.

El archivo se abre en modo de agregar (el historial se conserva entre ejecuciones) y el
encabezado se escribe solo al crearlo. Un fallo al escribir el registro nunca debe romper la API.
"""
import csv
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

CAMPOS = ["fecha_hora", "estado_http", "version", "prediccion"]
FORMATO_FECHA = "%Y-%m-%dT%H:%M:%S"


class RegistroSolicitudes:
    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self._lock = threading.Lock()          # varias solicitudes simultáneas no mezclan filas

    def registrar(self, estado_http, version=None, prediccion=None, ahora=None):
        """Agrega una fila. Devuelve True si se escribió y False si hubo un error de escritura."""
        ahora = ahora or datetime.now(timezone.utc)
        fila = {"fecha_hora": ahora.strftime(FORMATO_FECHA), "estado_http": int(estado_http),
                "version": version or "", "prediccion": "" if prediccion is None else prediccion}
        try:
            with self._lock:
                self.ruta.parent.mkdir(parents=True, exist_ok=True)
                es_nuevo = not self.ruta.exists() or self.ruta.stat().st_size == 0
                with open(self.ruta, "a", newline="", encoding="utf-8") as f:
                    escritor = csv.DictWriter(f, fieldnames=CAMPOS, lineterminator="\n")
                    if es_nuevo:
                        escritor.writeheader()                 # encabezado solo al crear el archivo
                    escritor.writerow(fila)
        except OSError as e:
            print(f"Aviso: no se pudo escribir el registro {self.ruta}: {e}", file=sys.stderr)
            return False
        return True