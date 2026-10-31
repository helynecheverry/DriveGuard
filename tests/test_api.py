"""Pruebas de la API con una base MongoDB simulada (mongomock)."""
import mongomock
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from core import storage


@pytest.fixture()
def cliente(monkeypatch):
    storage.set_client(mongomock.MongoClient(tz_aware=True))
    monkeypatch.setenv("NOTIFICACIONES_SIMULADAS", "true")
    yield TestClient(app)
    storage.set_client(None)


@pytest.fixture()
def conductor_id(cliente):
    r = cliente.post("/conductores", json={"nombre": "Ana Pérez", "documento": "123",
                                           "contactos": [{"nombre": "Luis", "correo": "luis@example.com",
                                                          "telegram_chat_id": "999"}]})
    assert r.status_code == 201
    return r.json()["id"]


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["mongodb"] is True


def test_crud_conductor(cliente, conductor_id):
    assert cliente.get(f"/conductores/{conductor_id}").json()["nombre"] == "Ana Pérez"
    assert len(cliente.get("/conductores").json()) == 1
    r = cliente.put(f"/conductores/{conductor_id}", json={"nombre": "Ana P.", "contactos": []})
    assert r.status_code == 200
    assert cliente.get(f"/conductores/{conductor_id}").json()["nombre"] == "Ana P."
    assert cliente.delete(f"/conductores/{conductor_id}").status_code == 200
    assert cliente.get(f"/conductores/{conductor_id}").status_code == 404


def test_validaciones(cliente):
    assert cliente.post("/conductores", json={"nombre": "A"}).status_code == 422
    r = cliente.post("/conductores", json={"nombre": "Ana", "contactos": [{"nombre": "Luis", "correo": "no-es-correo"}]})
    assert r.status_code == 422
    assert cliente.get("/conductores/xyz").status_code == 422


def test_sesion_eventos_y_estadisticas(cliente, conductor_id):
    sid = cliente.post("/sesiones", json={"conductor_id": conductor_id,
                                          "ubicacion": {"descripcion": "Pasto"}}).json()["id"]
    eventos = [{"tipo": "muestra", "nivel": "NORMAL", "ear": 0.30, "perclos": 0.05,
                "fecha": "2026-10-26T10:00:00+00:00"},
               {"tipo": "microsueno", "nivel": "ALTO", "ear": 0.08, "perclos": 0.2,
                "fecha": "2026-10-26T10:00:20+00:00"},
               {"tipo": "muestra", "nivel": "ALTO", "ear": 0.10, "perclos": 0.25,
                "fecha": "2026-10-26T10:01:00+00:00"}]
    assert cliente.post("/eventos/lote", json={"sesion_id": sid, "eventos": eventos}).json()["insertados"] == 3
    stats = cliente.get(f"/estadisticas/{sid}").json()
    assert stats["eventos_riesgo"] == 1 and stats["por_tipo"] == {"microsueno": 1}
    assert stats["duracion_s"] == 60.0 and stats["perclos_max"] == 0.25
    assert cliente.put(f"/sesiones/{sid}/cerrar", json={}).status_code == 200
    sesiones = cliente.get("/sesiones", params={"conductor_id": conductor_id}).json()
    assert sesiones[0]["fin"] is not None and sesiones[0]["resumen"]["eventos_riesgo"] == 1


def test_sesion_conductor_inexistente(cliente):
    r = cliente.post("/sesiones", json={"conductor_id": "65f000000000000000000000"})
    assert r.status_code == 404


def test_emergencia_y_cooldown(cliente, conductor_id):
    sid = cliente.post("/sesiones", json={"conductor_id": conductor_id}).json()["id"]
    cuerpo = {"conductor_id": conductor_id, "sesion_id": sid, "motivo": "sin_respuesta"}
    r = cliente.post("/alertas/emergencia", json=cuerpo).json()
    assert r["estado"] == "enviada" and len(r["resultados"]) == 2   # correo + telegram
    r2 = cliente.post("/alertas/emergencia", json=cuerpo).json()
    assert r2["estado"] == "omitida_cooldown"
    r3 = cliente.post("/alertas/emergencia", json={**cuerpo, "forzar": True}).json()
    assert r3["estado"] == "enviada"
    assert len(cliente.get("/alertas", params={"sesion_id": sid}).json()) == 3


def test_emergencia_sin_mongodb_usa_respaldo(cliente, monkeypatch):
    def caida(*a, **k):
        raise ConnectionError("MongoDB caído")
    monkeypatch.setattr(storage, "obtener_conductor", caida)
    monkeypatch.setattr(storage, "registrar_alerta", caida)
    respaldo = {"nombre": "Ana", "contactos": [{"nombre": "Luis", "correo": "luis@example.com"}]}
    r = cliente.post("/alertas/emergencia", json={"conductor_id": "65f000000000000000000000",
                                                  "motivo": "sos", "conductor": respaldo}).json()
    assert r["estado"] == "enviada" and "aviso" in r and "almacenamiento_error" in r


def test_prueba_notificaciones(cliente, conductor_id):
    r = cliente.post("/alertas/prueba", json={"conductor_id": conductor_id, "simulado": True}).json()
    assert r["estado"] == "enviada"
