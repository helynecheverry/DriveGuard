"""Puntos faciales con MediaPipe FaceLandmarker y métricas de somnolencia.

Por qué MediaPipe y no Haar Cascade: el detector Haar de SentimentLab V3 solo
entrega el rectángulo de la cara. Para medir somnolencia necesitamos la forma
de los ojos y de la boca, y FaceLandmarker (modelo preentrenado de Google,
funciona en CPU) entrega 478 puntos 3D, coeficientes de expresión
(blendshapes) y la matriz de orientación de la cabeza.

Se usa la API "Tasks" (mediapipe.tasks.python.vision), no la API antigua
mp.solutions, que está obsoleta.

Métricas:
- EAR (Eye Aspect Ratio, Soukupová y Čech, 2016):
      EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)
  Ojo abierto ~0.25-0.35, cerrado < 0.15.
- MAR (Mouth Aspect Ratio): apertura vertical media / ancho de la boca.
  Boca cerrada ~0.0-0.1, hablando ~0.2-0.4, bostezo > 0.6.
- Pitch / yaw / roll: ángulos de la cabeza en grados.
"""
from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.config import LANDMARKER_PATH, LANDMARKER_URL

# Índices de la malla facial de MediaPipe (468/478 puntos).
# Orden: p1 (comisura externa), p2, p3 (párpado superior), p4 (comisura
# interna), p5, p6 (párpado inferior).
OJO_DERECHO = [33, 160, 158, 133, 153, 144]
OJO_IZQUIERDO = [362, 385, 387, 263, 373, 380]
# Boca interna: comisuras y tres pares verticales.
BOCA_COMISURAS = (78, 308)
BOCA_VERTICALES = [(81, 178), (13, 14), (311, 402)]
CONTORNO_OJO_DER = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
CONTORNO_OJO_IZQ = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
CONTORNO_BOCA = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


# --------------------------------------------------------------------------
# Métricas puras (sin MediaPipe): se pueden probar con puntos conocidos
# --------------------------------------------------------------------------
def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, float)[:2] - np.asarray(b, float)[:2]))


def calcular_ear(ojo: np.ndarray) -> float:
    """EAR de un ojo a partir de 6 puntos (p1..p6) en píxeles."""
    ojo = np.asarray(ojo, dtype=float)
    if ojo.shape[0] != 6:
        raise ValueError("Se requieren 6 puntos para calcular el EAR")
    horizontal = _dist(ojo[0], ojo[3])
    if horizontal < 1e-6:
        return 0.0
    return (_dist(ojo[1], ojo[5]) + _dist(ojo[2], ojo[4])) / (2.0 * horizontal)


def calcular_mar(comisuras: np.ndarray, verticales: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """MAR: promedio de aperturas verticales dividido por el ancho de la boca."""
    ancho = _dist(comisuras[0], comisuras[1])
    if ancho < 1e-6 or not verticales:
        return 0.0
    return sum(_dist(a, b) for a, b in verticales) / (len(verticales) * ancho)


def angulos_desde_matriz(matriz: np.ndarray) -> tuple[float, float, float]:
    """Convierte la matriz de transformación 4x4 en (pitch, yaw, roll) en grados."""
    r = np.asarray(matriz, dtype=float)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll


def diferencia_angular(a: float, b: float) -> float:
    """Diferencia a-b normalizada al rango [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def metricas_desde_puntos(puntos: np.ndarray) -> dict:
    """Calcula EAR (por ojo y promedio) y MAR a partir de la malla en píxeles."""
    ear_der = calcular_ear(puntos[OJO_DERECHO])
    ear_izq = calcular_ear(puntos[OJO_IZQUIERDO])
    mar = calcular_mar(puntos[list(BOCA_COMISURAS)],
                       [(puntos[a], puntos[b]) for a, b in BOCA_VERTICALES])
    return {"ear_der": ear_der, "ear_izq": ear_izq, "ear": (ear_der + ear_izq) / 2, "mar": mar}


# --------------------------------------------------------------------------
# Modelo MediaPipe
# --------------------------------------------------------------------------
def descargar_modelo(destino: Path = LANDMARKER_PATH, url: str = LANDMARKER_URL) -> Path:
    """Descarga face_landmarker.task (≈3,6 MB) si no existe."""
    destino = Path(destino)
    if destino.exists() and destino.stat().st_size > 100_000:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo FaceLandmarker en {destino} ...")
    tmp = destino.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(destino)
    print("Modelo descargado.")
    return destino


@dataclass
class ResultadoRostro:
    """Resultado de procesar un fotograma con FaceLandmarker."""

    puntos: np.ndarray                  # (N, 3) en píxeles (z relativo)
    ear: float
    ear_izq: float
    ear_der: float
    mar: float
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    blendshapes: dict = field(default_factory=dict)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, ancho, alto


class DetectorRostro:
    """Envoltura de MediaPipe FaceLandmarker.

    modo="video": para secuencias (cámara o archivo); usa marcas de tiempo.
    modo="imagen": para fotografías sueltas (notebook).
    """

    def __init__(self, modo: str = "video", model_path: Path | str = LANDMARKER_PATH,
                 descargar: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = Path(model_path)
        if not model_path.exists():
            if not descargar:
                raise FileNotFoundError(
                    f"No existe {model_path}. Ejecute: python scripts/descargar_modelos.py")
            descargar_modelo(model_path)

        self._mp = mp
        self.modo = modo
        running = vision.RunningMode.VIDEO if modo == "video" else vision.RunningMode.IMAGE
        opciones = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opciones)
        self._ultimo_ts = -1

    def procesar(self, frame_bgr: np.ndarray, t_ms: int | None = None) -> ResultadoRostro | None:
        """Procesa un fotograma BGR. Devuelve None si no hay rostro."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Fotograma vacío")
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        imagen = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if self.modo == "video":
            ts = int(t_ms if t_ms is not None else time.monotonic() * 1000)
            ts = max(ts, self._ultimo_ts + 1)       # MediaPipe exige tiempos crecientes
            self._ultimo_ts = ts
            res = self._landmarker.detect_for_video(imagen, ts)
        else:
            res = self._landmarker.detect(imagen)

        if not res.face_landmarks:
            return None

        puntos = np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]],
                          dtype=np.float32)
        metricas = metricas_desde_puntos(puntos)

        pitch = yaw = roll = None
        if res.facial_transformation_matrixes:
            pitch, yaw, roll = angulos_desde_matriz(res.facial_transformation_matrixes[0])

        blend = {}
        if res.face_blendshapes:
            blend = {c.category_name: round(float(c.score), 4) for c in res.face_blendshapes[0]}

        x0, y0 = np.clip(puntos[:, :2].min(axis=0), 0, [w - 1, h - 1]).astype(int)
        x1, y1 = np.clip(puntos[:, :2].max(axis=0), 0, [w - 1, h - 1]).astype(int)
        return ResultadoRostro(puntos=puntos, pitch=pitch, yaw=yaw, roll=roll,
                               blendshapes=blend, bbox=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                               **metricas)

    def cerrar(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()


def recorte_ojo(frame_bgr: np.ndarray, puntos: np.ndarray, indices: list[int],
                margen: float = 0.6, tam: int = 64) -> np.ndarray | None:
    """Recorte cuadrado en escala de grises de un ojo (para la CNN opcional)."""
    pts = puntos[indices, :2]
    cx, cy = pts.mean(axis=0)
    lado = (pts[:, 0].max() - pts[:, 0].min()) * (1 + margen)
    if lado < 8:
        return None
    h, w = frame_bgr.shape[:2]
    x0, y0 = int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2))
    x1, y1 = int(min(w, cx + lado / 2)), int(min(h, cy + lado / 2))
    recorte = frame_bgr[y0:y1, x0:x1]
    if recorte.size == 0:
        return None
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gris, (tam, tam), interpolation=cv2.INTER_AREA)
