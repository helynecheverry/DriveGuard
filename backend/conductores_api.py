"""CRUD de conductores y sus contactos de emergencia."""
from fastapi import APIRouter, HTTPException

from backend.schemas import ConductorIn
from backend.utils import ejecutar
from core import storage

router = APIRouter(prefix="/conductores", tags=["Conductores"])


@router.post("", status_code=201)
def crear(datos: ConductorIn):
    conductor_id = ejecutar(storage.crear_conductor, datos.model_dump())
    return {"id": conductor_id}


@router.get("")
def listar():
    return ejecutar(storage.listar_conductores)


@router.get("/{conductor_id}")
def obtener(conductor_id: str):
    doc = ejecutar(storage.obtener_conductor, conductor_id)
    if doc is None:
        raise HTTPException(404, "Conductor no encontrado")
    return doc


@router.put("/{conductor_id}")
def actualizar(conductor_id: str, datos: ConductorIn):
    if not ejecutar(storage.actualizar_conductor, conductor_id, datos.model_dump()):
        raise HTTPException(404, "Conductor no encontrado")
    return {"id": conductor_id, "actualizado": True}


@router.delete("/{conductor_id}")
def eliminar(conductor_id: str):
    if not ejecutar(storage.eliminar_conductor, conductor_id):
        raise HTTPException(404, "Conductor no encontrado")
    return {"id": conductor_id, "eliminado": True}
