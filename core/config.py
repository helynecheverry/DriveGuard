"""Configuración central de DriveGuard Lab.

Reúne rutas del proyecto, variables de entorno (.env) y los umbrales por
defecto del motor de riesgo. Al importarse crea las carpetas de trabajo
(data/, models/, assets/...). Esto corrige un problema del laboratorio 02 de
SentimentLab V3, donde la celda de cámara fallaba porque data/ no existía.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CAPTURAS_DIR = DATA_DIR / "capturas"
VIDEOS_DIR = DATA_DIR / "videos"
EJEMPLOS_DIR = DATA_DIR / "ejemplos"
OJOS_DIR = DATA_DIR / "ojos"           # dataset de ojos (MRL o capturas propias)
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"


def asegurar_carpetas() -> None:
    """Crea las carpetas de trabajo si no existen (idempotente)."""
    for carpeta in (DATA_DIR, CAPTURAS_DIR, VIDEOS_DIR, EJEMPLOS_DIR, OJOS_DIR,
                    MODELS_DIR, ASSETS_DIR):
        carpeta.mkdir(parents=True, exist_ok=True)


asegurar_carpetas()


def env_bool(nombre: str, defecto: bool = False) -> bool:
    """Lee una variable de entorno booleana (true/1/si/yes)."""
    valor = os.getenv(nombre)
    if valor is None:
        return defecto
    return valor.strip().lower() in {"1", "true", "si", "sí", "yes", "y"}


def env_float(nombre: str, defecto: float) -> float:
    """Lee una variable de entorno numérica con valor por defecto."""
    try:
        return float(os.getenv(nombre, defecto))
    except (TypeError, ValueError):
        return defecto


# --------------------------------------------------------------------------
# Variables generales
# --------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "driveguard_lab")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

LANDMARKER_PATH = Path(os.getenv("LANDMARKER_PATH", MODELS_DIR / "face_landmarker.task"))
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")

FACE_MODEL_ID = os.getenv("FACE_MODEL_ID", "trpakov/vit-face-expression")
FACE_DEVICE = os.getenv("FACE_DEVICE", "cpu")
EXPRESIONES_ACTIVAS = env_bool("EXPRESIONES_ACTIVAS", True)
# Segundos entre dos clasificaciones ViT. En CPU conviene 2-3 s; subirlo
# no afecta la detección de somnolencia, que corre con MediaPipe en cada frame.
EXPRESION_INTERVALO_S = env_float("EXPRESION_INTERVALO_S", 2.0)

OJOS_METODO = os.getenv("OJOS_METODO", "ear").lower()      # "ear" o "cnn"
CNN_OJOS_PATH = Path(os.getenv("CNN_OJOS_PATH", MODELS_DIR / "cnn_ojos.pt"))

ALARMA_BACKEND = os.getenv("ALARMA_BACKEND", "auto").lower()  # auto|pygame|winsound|ninguno

NOTIFICACIONES_SIMULADAS = env_bool("NOTIFICACIONES_SIMULADAS", False)
ALERTA_COOLDOWN_S = env_float("ALERTA_COOLDOWN_S", 120)


# --------------------------------------------------------------------------
# Umbrales del motor de riesgo
# --------------------------------------------------------------------------
@dataclass
class ConfigRiesgo:
    """Parámetros del motor de riesgo. Todos son editables desde la interfaz."""

    calibracion_s: float = 10.0          # duración de la calibración inicial
    factor_ear: float = 0.75             # umbral = factor * EAR base
    ear_umbral_defecto: float = 0.21     # se usa si no hay calibración válida
    ear_umbral_manual: float | None = None  # None = automático; valor fijo si se ajusta en la interfaz
    suavizado_frames: int = 3            # promedio móvil del EAR

    microsueno_s: float = 1.5            # ojos cerrados continuos -> ALTO
    perclos_ventana_s: float = 60.0
    perclos_umbral: float = 0.30         # 30 % del tiempo con ojos cerrados

    mar_umbral: float = 0.60             # boca abierta (bostezo)
    bostezo_min_s: float = 2.0
    bostezo_ventana_s: float = 300.0
    bostezos_limite: int = 3

    cabeceo_grados: float = 15.0         # caída de la cabeza respecto a la base
    cabeceo_min_s: float = 0.4
    cabeceo_ventana_s: float = 60.0
    cabeceos_limite: int = 3
    cabeza_caida_s: float = 3.0          # cabeza caída continua -> ALTO

    giro_grados: float = 30.0            # cabeza girada (yaw) -> distracción
    distraccion_s: float = 3.0           # giro o rostro ausente

    expresion_ventana_s: float = 30.0
    expresion_umbral: float = 0.70       # fracción de expresiones negativas

    histeresis_s: float = 2.0            # tiempo antes de bajar de nivel
    sin_respuesta_s: float = 10.0        # ALTO sin respuesta -> emergencia
    silencio_tras_ok_s: float = 5.0      # pausa tras "Estoy bien"

    @classmethod
    def desde_dict(cls, datos: dict | None) -> "ConfigRiesgo":
        """Crea la configuración ignorando claves desconocidas."""
        validos = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (datos or {}).items() if k in validos})

    def a_dict(self) -> dict:
        return asdict(self)
