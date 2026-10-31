"""Cliente HTTP de la API FastAPI usado por la interfaz Streamlit."""
from __future__ import annotations

import requests

from core.config import API_URL


class ApiError(Exception):
    """Error devuelto por la API o de conexión con ella."""

    def __init__(self, mensaje: str, status: int | None = None):
        super().__init__(mensaje)
        self.status = status


def _req(metodo: str, ruta: str, timeout: float = 10, **kwargs):
    try:
        r = requests.request(metodo, API_URL + ruta, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"No hay conexión con la API en {API_URL}. ¿Inició uvicorn? ({type(exc).__name__})") from exc
    if not r.ok:
        try:
            detalle = r.json().get("detail", r.text)
        except ValueError:
            detalle = r.text
        raise ApiError(f"HTTP {r.status_code}: {detalle}", r.status_code)
    return r.json()


def health() -> dict:
    return _req("GET", "/health", timeout=4)


# Conductores
def listar_conductores() -> list[dict]:
    return _req("GET", "/conductores")


def crear_conductor(datos: dict) -> str:
    return _req("POST", "/conductores", json=datos)["id"]


def actualizar_conductor(conductor_id: str, datos: dict) -> None:
    _req("PUT", f"/conductores/{conductor_id}", json=datos)


def eliminar_conductor(conductor_id: str) -> None:
    _req("DELETE", f"/conductores/{conductor_id}")


# Sesiones y eventos
def crear_sesion(conductor_id: str, ubicacion: dict | None, config: dict, origen: str = "camara") -> str:
    return _req("POST", "/sesiones", json={"conductor_id": conductor_id, "ubicacion": ubicacion,
                                           "config": config, "origen": origen})["id"]


def cerrar_sesion(sesion_id: str) -> None:
    _req("PUT", f"/sesiones/{sesion_id}/cerrar", json={})


def listar_sesiones(conductor_id: str | None = None, limite: int = 50) -> list[dict]:
    params = {"limite": limite}
    if conductor_id:
        params["conductor_id"] = conductor_id
    return _req("GET", "/sesiones", params=params)


def enviar_eventos(sesion_id: str, eventos: list[dict]) -> int:
    if not eventos:
        return 0
    return _req("POST", "/eventos/lote", json={"sesion_id": sesion_id, "eventos": eventos})["insertados"]


def eventos(sesion_id: str) -> list[dict]:
    return _req("GET", "/eventos", params={"sesion_id": sesion_id}, timeout=20)


def estadisticas(sesion_id: str) -> dict:
    return _req("GET", f"/estadisticas/{sesion_id}")


# Alertas
def emergencia(payload: dict) -> dict:
    return _req("POST", "/alertas/emergencia", json=payload, timeout=60)


def prueba_alertas(conductor_id: str, simulado: bool | None = None) -> dict:
    return _req("POST", "/alertas/prueba", json={"conductor_id": conductor_id, "simulado": simulado}, timeout=60)


def alertas(sesion_id: str | None = None) -> list[dict]:
    return _req("GET", "/alertas", params={"sesion_id": sesion_id} if sesion_id else {})


def canales() -> dict:
    return _req("GET", "/alertas/canales")


def telegram_chats() -> list[dict]:
    return _req("GET", "/alertas/telegram/chats", timeout=15)
