"""Utilidades comunes de la API."""
from __future__ import annotations

from fastapi import HTTPException


def ejecutar(fn, *args, **kwargs):
    """Ejecuta una operación de base de datos traduciendo errores a HTTP.

    ValueError (identificador inválido) -> 422; cualquier error de conexión
    con MongoDB -> 503, para que la interfaz pueda informarlo con claridad.
    """
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, f"MongoDB no disponible: {type(exc).__name__}: {exc}") from exc
