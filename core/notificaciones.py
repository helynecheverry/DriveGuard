"""Envío de alertas a los contactos de emergencia.

Canales:
- Correo electrónico con smtplib (Gmail con contraseña de aplicación).
- Telegram Bot API (gratuito, recomendado como "mensaje de texto").
- SMS con Twilio (opcional; la cuenta de prueba solo envía a números verificados).

Las credenciales se leen del archivo .env en el momento del envío. Con
NOTIFICACIONES_SIMULADAS=true no se envía nada: solo se registra el intento,
útil para practicar en clase sin credenciales.
"""
from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage

import requests

from core.config import env_bool

log = logging.getLogger("driveguard.notificaciones")

MOTIVOS_EMERGENCIA = {
    "sin_respuesta": "El conductor no responde a la alarma (posible microsueño o pérdida de conciencia).",
    "sos": "El conductor o un pasajero activó el botón SOS.",
    "impacto_simulado": "Se detectó un posible impacto (simulación con acelerómetro).",
    "prueba": "Mensaje de prueba del sistema. No hay una emergencia real.",
}


def _texto_ubicacion(ubicacion) -> str:
    if not ubicacion:
        return "No disponible"
    if isinstance(ubicacion, str):
        return ubicacion
    partes = [ubicacion.get("descripcion") or ""]
    lat, lon = ubicacion.get("lat"), ubicacion.get("lon")
    if lat is not None and lon is not None:
        partes.append(f"https://maps.google.com/?q={lat},{lon}")
    return " — ".join(p for p in partes if p) or "No disponible"


def construir_mensaje(conductor: dict, motivo: str, nivel: str = "ALTO", ubicacion=None,
                      fecha: datetime | None = None, detalle: str | None = None) -> tuple[str, str]:
    """Devuelve (asunto, cuerpo) del mensaje de emergencia."""
    fecha = fecha or datetime.now()
    nombre = conductor.get("nombre", "Conductor")
    asunto = f"[DriveGuard] {'PRUEBA' if motivo == 'prueba' else 'ALERTA'}: {nombre}"
    cuerpo = (
        f"🚨 ALERTA DRIVEGUARD LAB\n"
        f"Conductor: {nombre}\n"
        f"Fecha y hora: {fecha:%Y-%m-%d %H:%M:%S}\n"
        f"Evento: {MOTIVOS_EMERGENCIA.get(motivo, motivo)}\n"
        f"Nivel de riesgo: {nivel}\n"
        f"Ubicación: {_texto_ubicacion(ubicacion)}\n"
        + (f"Detalle: {detalle}\n" if detalle else "")
        + "\nIntente comunicarse con el conductor. Si no responde, llame a la línea de emergencias.\n"
        "Alerta automática de un prototipo académico."
    )
    return asunto, cuerpo


def estado_canales() -> dict:
    """Indica qué canales tienen credenciales configuradas."""
    return {
        "correo": bool(os.getenv("SMTP_USER") and os.getenv("SMTP_PASSWORD")),
        "telegram": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "sms_twilio": bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN")
                           and os.getenv("TWILIO_FROM")),
        "simulado": env_bool("NOTIFICACIONES_SIMULADAS", False),
    }


def _resultado(canal: str, destino: str, ok: bool, error: str | None = None) -> dict:
    return {"canal": canal, "destino": destino, "ok": ok, "error": error}


def enviar_correo(destino: str, asunto: str, cuerpo: str) -> dict:
    usuario, clave = os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD")
    if not (usuario and clave):
        return _resultado("correo", destino, False, "SMTP_USER/SMTP_PASSWORD no configurados")
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = asunto, os.getenv("SMTP_FROM", usuario), destino
    msg.set_content(cuerpo)
    try:
        with smtplib.SMTP(os.getenv("SMTP_HOST", "smtp.gmail.com"),
                          int(os.getenv("SMTP_PORT", "587")), timeout=15) as smtp:
            smtp.starttls()
            smtp.login(usuario, clave)
            smtp.send_message(msg)
        return _resultado("correo", destino, True)
    except Exception as exc:
        return _resultado("correo", destino, False, f"{type(exc).__name__}: {exc}")


def enviar_telegram(chat_id: str, texto: str) -> dict:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return _resultado("telegram", str(chat_id), False, "TELEGRAM_BOT_TOKEN no configurado")
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": texto}, timeout=10)
        datos = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.ok and datos.get("ok", False):
            return _resultado("telegram", str(chat_id), True)
        return _resultado("telegram", str(chat_id), False, datos.get("description", f"HTTP {r.status_code}"))
    except Exception as exc:
        return _resultado("telegram", str(chat_id), False, f"{type(exc).__name__}: {exc}")


def enviar_sms_twilio(numero: str, texto: str) -> dict:
    sid, token, origen = (os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"),
                          os.getenv("TWILIO_FROM"))
    if not (sid and token and origen):
        return _resultado("sms", numero, False, "Credenciales de Twilio no configuradas")
    try:
        r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                          data={"From": origen, "To": numero, "Body": texto[:1500]},
                          auth=(sid, token), timeout=15)
        if r.ok:
            return _resultado("sms", numero, True)
        return _resultado("sms", numero, False, r.json().get("message", f"HTTP {r.status_code}"))
    except Exception as exc:
        return _resultado("sms", numero, False, f"{type(exc).__name__}: {exc}")


def notificar_contactos(conductor: dict, motivo: str, nivel: str = "ALTO", ubicacion=None,
                        detalle: str | None = None, simulado: bool | None = None) -> list[dict]:
    """Envía el mensaje a todos los contactos por cada canal que tengan."""
    simulado = env_bool("NOTIFICACIONES_SIMULADAS", False) if simulado is None else simulado
    asunto, cuerpo = construir_mensaje(conductor, motivo, nivel, ubicacion, detalle=detalle)
    resultados: list[dict] = []
    contactos = conductor.get("contactos") or []
    if not contactos:
        return [_resultado("ninguno", "-", False, "El conductor no tiene contactos de emergencia")]
    for c in contactos:
        envios = []
        if c.get("correo"):
            envios.append(("correo", c["correo"], lambda c=c: enviar_correo(c["correo"], asunto, cuerpo)))
        if c.get("telegram_chat_id"):
            envios.append(("telegram", str(c["telegram_chat_id"]),
                           lambda c=c: enviar_telegram(c["telegram_chat_id"], cuerpo)))
        if c.get("telefono") and estado_canales()["sms_twilio"]:
            envios.append(("sms", c["telefono"], lambda c=c: enviar_sms_twilio(c["telefono"], cuerpo)))
        for canal, destino, enviar in envios:
            if simulado:
                res = {**_resultado(canal, destino, True), "simulado": True}
            else:
                res = enviar()
            res["contacto"] = c.get("nombre")
            resultados.append(res)
            log.info("Notificación %s a %s: %s", canal, destino, "OK" if res["ok"] else res["error"])
    return resultados or [_resultado("ninguno", "-", False, "Los contactos no tienen correo ni Telegram")]


def obtener_chat_ids_telegram() -> list[dict]:
    """Lista los chats que escribieron al bot (para obtener el chat_id)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN no configurado")
    r = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
    r.raise_for_status()
    chats = {}
    for upd in r.json().get("result", []):
        chat = (upd.get("message") or upd.get("my_chat_member") or {}).get("chat")
        if chat:
            chats[chat["id"]] = {"chat_id": chat["id"],
                                 "nombre": chat.get("first_name") or chat.get("title"),
                                 "usuario": chat.get("username")}
    return list(chats.values())
