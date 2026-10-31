"""API REST de DriveGuard Lab.

Ejecutar desde la raíz del proyecto:
    python -m uvicorn backend.main:app --reload --port 8000
Documentación interactiva: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI

from backend.alertas_api import router as alertas_router
from backend.conductores_api import router as conductores_router
from backend.sesiones_api import router as sesiones_router
from core import notificaciones, storage

app = FastAPI(
    title="DriveGuard Lab API",
    version="1.0",
    description="Monitoreo de somnolencia y riesgo del conductor (prototipo académico).",
)


@app.get("/health", tags=["Sistema"])
def health():
    return {"status": "ok", "mongodb": storage.ping(), "canales": notificaciones.estado_canales()}


app.include_router(conductores_router)
app.include_router(sesiones_router)
app.include_router(alertas_router)
