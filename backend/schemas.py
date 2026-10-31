"""Modelos de entrada (Pydantic v2) de la API."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_CORREO = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Contacto(BaseModel):
    nombre: str = Field(min_length=2, max_length=80)
    correo: str | None = None
    telegram_chat_id: str | None = None
    telefono: str | None = Field(default=None, description="Formato internacional, p. ej. +573001234567")

    @field_validator("correo")
    @classmethod
    def validar_correo(cls, v):
        if v in (None, ""):
            return None
        if not _CORREO.match(v):
            raise ValueError("Correo electrónico inválido")
        return v.strip().lower()

    @field_validator("telegram_chat_id", "telefono")
    @classmethod
    def vacio_a_none(cls, v):
        return None if v in (None, "") else str(v).strip()


class ConductorIn(BaseModel):
    nombre: str = Field(min_length=2, max_length=100)
    documento: str | None = None
    contactos: list[Contacto] = Field(default_factory=list)


class Ubicacion(BaseModel):
    descripcion: str = ""
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)


class SesionIn(BaseModel):
    conductor_id: str
    ubicacion: Ubicacion | None = None
    config: dict = Field(default_factory=dict)
    origen: Literal["camara", "video", "demo"] = "camara"


class CierreSesion(BaseModel):
    resumen: dict | None = None


class EventosLote(BaseModel):
    sesion_id: str
    eventos: list[dict] = Field(max_length=5000)


class EmergenciaIn(BaseModel):
    conductor_id: str
    sesion_id: str | None = None
    motivo: Literal["sin_respuesta", "sos", "impacto_simulado"]
    nivel: str = "ALTO"
    ubicacion: Ubicacion | None = None
    detalle: str | None = None
    forzar: bool = False
    # Copia del conductor enviada por la interfaz: permite avisar a los
    # contactos aunque MongoDB no esté disponible en ese momento.
    conductor: dict | None = None


class PruebaIn(BaseModel):
    conductor_id: str
    simulado: bool | None = None
