"""Sesiones de conducción, eventos y estadísticas."""
from fastapi import APIRouter, HTTPException

from backend.schemas import CierreSesion, EventosLote, SesionIn
from backend.utils import ejecutar
from core import storage

router = APIRouter(tags=["Sesiones y eventos"])


@router.post("/sesiones", status_code=201)
def crear_sesion(datos: SesionIn):
    if ejecutar(storage.obtener_conductor, datos.conductor_id) is None:
        raise HTTPException(404, "Conductor no encontrado")
    ubicacion = datos.ubicacion.model_dump() if datos.ubicacion else {}
    sesion_id = ejecutar(storage.crear_sesion, datos.conductor_id, ubicacion, datos.config, datos.origen)
    return {"id": sesion_id}


@router.get("/sesiones")
def listar_sesiones(conductor_id: str | None = None, limite: int = 50):
    return ejecutar(storage.listar_sesiones, conductor_id, limite)


@router.get("/sesiones/{sesion_id}")
def obtener_sesion(sesion_id: str):
    doc = ejecutar(storage.obtener_sesion, sesion_id)
    if doc is None:
        raise HTTPException(404, "Sesión no encontrada")
    return doc


@router.put("/sesiones/{sesion_id}/cerrar")
def cerrar_sesion(sesion_id: str, datos: CierreSesion | None = None):
    if not ejecutar(storage.cerrar_sesion, sesion_id, datos.resumen if datos else None):
        raise HTTPException(404, "Sesión no encontrada")
    return {"id": sesion_id, "cerrada": True}


@router.post("/eventos/lote", status_code=201)
def eventos_lote(lote: EventosLote):
    """Recibe los eventos acumulados por la interfaz (cada ~5 s)."""
    return {"insertados": ejecutar(storage.insertar_eventos, lote.sesion_id, lote.eventos)}


@router.get("/eventos")
def eventos(sesion_id: str, incluir_muestras: bool = True):
    return ejecutar(storage.listar_eventos, sesion_id, incluir_muestras)


@router.get("/estadisticas/{sesion_id}")
def estadisticas(sesion_id: str):
    return ejecutar(storage.estadisticas, sesion_id)
