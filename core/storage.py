"""Persistencia en MongoDB (pymongo).

Colecciones:
- conductores: nombre, documento, contactos [{nombre, correo, telegram_chat_id, telefono}]
- sesiones:    conductor_id, inicio, fin, ubicacion, config, resumen
- eventos:     sesion_id, fecha, tipo, nivel, ear, mar, perclos, ...
               (tipo "muestra" = métricas periódicas cada 5 s)
- alertas:     sesion_id, conductor_id, motivo, resultados de envío, estado

No se guardan imágenes ni fotogramas (privacidad), igual que en SentimentLab V3.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId

from core.config import MONGO_DB, MONGO_URI

_client = None


def set_client(client) -> None:
    """Permite inyectar otro cliente (p. ej. mongomock en las pruebas)."""
    global _client
    _client = client


def get_db():
    """Devuelve la base de datos; falla rápido (3 s) si MongoDB no responde."""
    global _client
    if _client is None:
        from pymongo import MongoClient
        cliente = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        cliente.admin.command("ping")
        _client = cliente
        _crear_indices(cliente[MONGO_DB])
    return _client[MONGO_DB]


def ping() -> bool:
    try:
        get_db().command("ping")
        return True
    except Exception:
        global _client
        _client = None
        return False


def _crear_indices(db) -> None:
    db.eventos.create_index([("sesion_id", 1), ("fecha", 1)])
    db.sesiones.create_index([("conductor_id", 1), ("inicio", -1)])
    db.alertas.create_index([("sesion_id", 1), ("fecha", -1)])


def ahora() -> datetime:
    return datetime.now(timezone.utc)


def _oid(valor: str) -> ObjectId:
    try:
        return ObjectId(str(valor))
    except (InvalidId, TypeError) as exc:
        raise ValueError(f"Identificador inválido: {valor}") from exc


def _limpiar(doc: dict | None) -> dict | None:
    """Convierte ObjectId y fechas a texto para poder devolver JSON."""
    if doc is None:
        return None
    out = {}
    for k, v in doc.items():
        if k == "_id":
            out["id"] = str(v)
        elif isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).isoformat()
        elif isinstance(v, dict):
            out[k] = _limpiar(v)
        else:
            out[k] = v
    return out


def _fecha(valor) -> datetime:
    if isinstance(valor, datetime):
        return valor
    if isinstance(valor, str):
        try:
            return datetime.fromisoformat(valor.replace("Z", "+00:00"))
        except ValueError:
            pass
    return ahora()


# ------------------------------------------------------------------ conductores
def crear_conductor(datos: dict) -> str:
    doc = {**datos, "creado": ahora()}
    return str(get_db().conductores.insert_one(doc).inserted_id)


def listar_conductores() -> list[dict]:
    return [_limpiar(d) for d in get_db().conductores.find().sort("nombre", 1)]


def obtener_conductor(conductor_id: str) -> dict | None:
    return _limpiar(get_db().conductores.find_one({"_id": _oid(conductor_id)}))


def actualizar_conductor(conductor_id: str, datos: dict) -> bool:
    r = get_db().conductores.update_one({"_id": _oid(conductor_id)},
                                        {"$set": {**datos, "actualizado": ahora()}})
    return r.matched_count == 1


def eliminar_conductor(conductor_id: str) -> bool:
    return get_db().conductores.delete_one({"_id": _oid(conductor_id)}).deleted_count == 1


# --------------------------------------------------------------------- sesiones
def crear_sesion(conductor_id: str, ubicacion: dict | None = None, config: dict | None = None,
                 origen: str = "camara") -> str:
    doc = {"conductor_id": str(conductor_id), "inicio": ahora(), "fin": None,
           "ubicacion": ubicacion or {}, "config": config or {}, "origen": origen, "resumen": {}}
    return str(get_db().sesiones.insert_one(doc).inserted_id)


def cerrar_sesion(sesion_id: str, resumen: dict | None = None) -> bool:
    resumen = resumen or estadisticas(sesion_id)
    r = get_db().sesiones.update_one({"_id": _oid(sesion_id)},
                                     {"$set": {"fin": ahora(), "resumen": resumen}})
    return r.matched_count == 1


def obtener_sesion(sesion_id: str) -> dict | None:
    return _limpiar(get_db().sesiones.find_one({"_id": _oid(sesion_id)}))


def listar_sesiones(conductor_id: str | None = None, limite: int = 50) -> list[dict]:
    filtro = {"conductor_id": str(conductor_id)} if conductor_id else {}
    cursor = get_db().sesiones.find(filtro).sort("inicio", -1).limit(min(limite, 500))
    return [_limpiar(d) for d in cursor]


# ---------------------------------------------------------------------- eventos
def insertar_eventos(sesion_id: str, eventos: list[dict]) -> int:
    if not eventos:
        return 0
    _oid(sesion_id)
    docs = []
    for e in eventos:
        doc = {k: v for k, v in e.items() if k not in ("_id", "id")}
        doc["sesion_id"] = str(sesion_id)
        doc["fecha"] = _fecha(e.get("fecha"))
        docs.append(doc)
    return len(get_db().eventos.insert_many(docs).inserted_ids)


def listar_eventos(sesion_id: str, incluir_muestras: bool = True, limite: int = 20000) -> list[dict]:
    filtro: dict = {"sesion_id": str(sesion_id)}
    if not incluir_muestras:
        filtro["tipo"] = {"$ne": "muestra"}
    cursor = get_db().eventos.find(filtro).sort("fecha", 1).limit(limite)
    return [_limpiar(d) for d in cursor]


# ---------------------------------------------------------------------- alertas
def registrar_alerta(doc: dict) -> str:
    datos = {**doc, "fecha": ahora()}
    return str(get_db().alertas.insert_one(datos).inserted_id)


def ultima_alerta_enviada(sesion_id: str) -> dict | None:
    doc = get_db().alertas.find_one({"sesion_id": str(sesion_id), "estado": "enviada"},
                                    sort=[("fecha", -1)])
    return doc


def listar_alertas(sesion_id: str | None = None, limite: int = 100) -> list[dict]:
    filtro = {"sesion_id": str(sesion_id)} if sesion_id else {}
    return [_limpiar(d) for d in get_db().alertas.find(filtro).sort("fecha", -1).limit(limite)]


# ---------------------------------------------------------------- estadísticas
def estadisticas(sesion_id: str) -> dict:
    """Resumen de una sesión calculado a partir de sus eventos."""
    eventos = listar_eventos(sesion_id)
    alertas = get_db().alertas.count_documents({"sesion_id": str(sesion_id)})
    muestras = [e for e in eventos if e.get("tipo") == "muestra"]
    riesgos = [e for e in eventos if e.get("tipo") not in ("muestra", "recuperacion")]
    ears = [e["ear"] for e in muestras if e.get("ear") is not None]
    perclos = [e["perclos"] for e in muestras if e.get("perclos") is not None]
    duracion = 0.0
    if len(eventos) >= 2:
        duracion = (_fecha(eventos[-1]["fecha"]) - _fecha(eventos[0]["fecha"])).total_seconds()
    return {
        "eventos_riesgo": len(riesgos),
        "por_tipo": dict(Counter(e.get("tipo") for e in riesgos)),
        "por_nivel": dict(Counter(e.get("nivel") for e in riesgos)),
        "ear_promedio": round(sum(ears) / len(ears), 4) if ears else None,
        "perclos_max": round(max(perclos), 4) if perclos else None,
        "duracion_s": round(duracion, 1),
        "alertas": alertas,
    }
