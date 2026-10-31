"""Estado compartido de la interfaz y protocolo de emergencia."""
from __future__ import annotations

import time

import streamlit as st

from core.config import ConfigRiesgo, EXPRESION_INTERVALO_S, EXPRESIONES_ACTIVAS
from frontend import api_client as api

CUENTA_REGRESIVA_S = 15
COLORES_NIVEL = {"NORMAL": "#2e9e4f", "BAJO": "#e6b800", "MEDIO": "#f07f13", "ALTO": "#d62f2f"}


def inicializar() -> None:
    """Valores por defecto de st.session_state."""
    defaults = {
        "conductor": None,
        "conductores": [],
        "config": ConfigRiesgo().a_dict(),
        "opciones": {"expresiones": EXPRESIONES_ACTIVAS, "intervalo_expresiones": EXPRESION_INTERVALO_S,
                     "sonido": True, "volumen": 0.8, "espejo": True},
        "ubicacion": {"descripcion": "", "lat": None, "lon": None},
        "sesion_id": None,
        "ultima_sesion_id": None,
        "eventos_pendientes": [],
        "t_envio_eventos": 0.0,
        "emergencia": None,
        "api_ok": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def cargar_conductores() -> list[dict]:
    try:
        st.session_state.conductores = api.listar_conductores()
        st.session_state.api_ok = True
    except api.ApiError as exc:
        st.session_state.api_ok = False
        st.session_state.error_api = str(exc)
    return st.session_state.conductores


def ubicacion_actual() -> dict:
    u = st.session_state.ubicacion
    return {"descripcion": u.get("descripcion") or "", "lat": u.get("lat"), "lon": u.get("lon")}


# --------------------------------------------------------------------------
# Protocolo de emergencia
# --------------------------------------------------------------------------
def iniciar_emergencia(motivo: str, detalle: str | None = None, inmediata: bool = False) -> None:
    """Inicia la cuenta regresiva (o el envío inmediato para SOS)."""
    if st.session_state.emergencia and not st.session_state.emergencia.get("enviada"):
        return  # ya hay una en curso
    st.session_state.emergencia = {
        "motivo": motivo, "detalle": detalle, "inicio": time.time(),
        "duracion": 0 if inmediata else CUENTA_REGRESIVA_S, "enviada": False, "resultado": None,
    }


def cancelar_emergencia() -> None:
    st.session_state.emergencia = None


def enviar_emergencia() -> dict:
    """Envía la alerta por la API; si la API no responde, envía localmente."""
    em = st.session_state.emergencia
    conductor = st.session_state.conductor
    if conductor is None:
        resultado = {"estado": "fallida", "resultados": [],
                     "error": "No hay conductor seleccionado: no se conocen contactos de emergencia."}
    else:
        payload = {"conductor_id": conductor["id"], "sesion_id": st.session_state.sesion_id,
                   "motivo": em["motivo"], "nivel": "ALTO", "ubicacion": ubicacion_actual(),
                   "detalle": em.get("detalle"), "conductor": conductor}
        try:
            resultado = api.emergencia(payload)
        except api.ApiError as exc:
            # Respaldo: la API no está disponible, se envía desde este equipo.
            from core.notificaciones import notificar_contactos
            res = notificar_contactos(conductor, em["motivo"], "ALTO", ubicacion_actual(),
                                      detalle=em.get("detalle"))
            resultado = {"estado": "enviada" if any(r["ok"] for r in res) else "fallida",
                         "resultados": res, "aviso": f"Enviado sin la API (respaldo local): {exc}"}
    em["enviada"] = True
    em["resultado"] = resultado
    return resultado
