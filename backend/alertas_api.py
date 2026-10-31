"""Protocolo de emergencia: envío de alertas a contactos.

El envío NO depende de que MongoDB esté disponible: si la base de datos
falla, se usan los datos del conductor que manda la interfaz y el registro
de la alerta se informa como error de almacenamiento.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.schemas import EmergenciaIn, PruebaIn
from backend.utils import ejecutar
from core import notificaciones, storage
from core.config import ALERTA_COOLDOWN_S

router = APIRouter(prefix="/alertas", tags=["Alertas de emergencia"])


def _conductor(conductor_id: str, respaldo: dict | None) -> tuple[dict, str | None]:
    try:
        doc = storage.obtener_conductor(conductor_id)
        if doc is None:
            raise HTTPException(404, "Conductor no encontrado")
        return doc, None
    except HTTPException:
        raise
    except Exception as exc:
        if respaldo:
            return respaldo, f"MongoDB no disponible, se usaron los datos enviados por la interfaz: {exc}"
        raise HTTPException(503, f"MongoDB no disponible y no se enviaron datos de respaldo: {exc}")


def _en_cooldown(sesion_id: str | None) -> float | None:
    """Segundos restantes de espera si ya se envió una alerta reciente."""
    if not sesion_id:
        return None
    try:
        ultima = storage.ultima_alerta_enviada(sesion_id)
    except Exception:
        return None
    if not ultima:
        return None
    fecha = ultima["fecha"]
    if fecha.tzinfo is None:
        fecha = fecha.replace(tzinfo=timezone.utc)
    transcurrido = (datetime.now(timezone.utc) - fecha).total_seconds()
    return ALERTA_COOLDOWN_S - transcurrido if transcurrido < ALERTA_COOLDOWN_S else None


@router.post("/emergencia")
def emergencia(datos: EmergenciaIn):
    conductor, aviso = _conductor(datos.conductor_id, datos.conductor)
    ubicacion = datos.ubicacion.model_dump() if datos.ubicacion else None

    restante = None if datos.forzar else _en_cooldown(datos.sesion_id)
    if restante is not None:
        registro = {"sesion_id": datos.sesion_id, "conductor_id": datos.conductor_id,
                    "motivo": datos.motivo, "estado": "omitida_cooldown", "resultados": []}
        try:
            storage.registrar_alerta(registro)
        except Exception:
            pass
        return {"estado": "omitida_cooldown", "espera_s": round(restante, 1), "resultados": []}

    resultados = notificaciones.notificar_contactos(conductor, datos.motivo, datos.nivel,
                                                    ubicacion, detalle=datos.detalle)
    estado = "enviada" if any(r["ok"] for r in resultados) else "fallida"
    respuesta = {"estado": estado, "resultados": resultados}
    if aviso:
        respuesta["aviso"] = aviso
    try:
        respuesta["alerta_id"] = storage.registrar_alerta({
            "sesion_id": datos.sesion_id, "conductor_id": datos.conductor_id,
            "motivo": datos.motivo, "nivel": datos.nivel, "ubicacion": ubicacion,
            "detalle": datos.detalle, "estado": estado, "resultados": resultados})
    except Exception as exc:
        respuesta["almacenamiento_error"] = str(exc)
    return respuesta


@router.post("/prueba")
def prueba(datos: PruebaIn):
    conductor = ejecutar(storage.obtener_conductor, datos.conductor_id)
    if conductor is None:
        raise HTTPException(404, "Conductor no encontrado")
    resultados = notificaciones.notificar_contactos(conductor, "prueba", "PRUEBA",
                                                    simulado=datos.simulado)
    return {"estado": "enviada" if any(r["ok"] for r in resultados) else "fallida",
            "resultados": resultados}


@router.get("")
def listar(sesion_id: str | None = None, limite: int = 100):
    return ejecutar(storage.listar_alertas, sesion_id, limite)


@router.get("/canales")
def canales():
    return notificaciones.estado_canales()


@router.get("/telegram/chats")
def telegram_chats():
    """Ayuda a obtener el chat_id: el contacto debe escribir primero al bot."""
    try:
        return notificaciones.obtener_chat_ids_telegram()
    except Exception as exc:
        raise HTTPException(502, f"No se pudo consultar Telegram: {exc}")
