"""Excepciones propias de la aplicación.

Todas heredan de ErrorAplicacion: los scripts la capturan, muestran un mensaje
comprensible y terminan con código de salida distinto de cero.
"""

class ErrorAplicacion(Exception):
    """Base de los errores previsibles (entrada inválida, modelo inexistente, etc.)."""


class VersionInvalida(ErrorAplicacion):
    """El identificador de versión tiene caracteres no permitidos."""


class VersionExistente(ErrorAplicacion):
    """Ya existe una versión con ese identificador (no se sobrescribe)."""


class ModeloNoEncontrado(ErrorAplicacion):
    """No existe el archivo del modelo o sus metadatos."""


class ModeloIlegible(ErrorAplicacion):
    """El archivo del modelo o de metadatos existe pero no se pudo leer."""


class EntradaInvalida(ErrorAplicacion):
    """El CSV de entrada no existe, no se puede leer o no cumple el esquema."""


class ErrorDePrediccion(ErrorAplicacion):
    """El modelo no pudo generar predicciones válidas."""