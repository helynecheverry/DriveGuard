"""Expresiones faciales aparentes (reutilizado de SentimentLab V3).

Se conserva el detector Haar y el clasificador ViT `trpakov/vit-face-expression`
del laboratorio 02. Novedad en DriveGuard Lab: `AnalizadorExpresiones`, que
ejecuta el ViT en un hilo propio y como máximo una vez cada
`EXPRESION_INTERVALO_S` segundos. Así la clasificación (lenta en CPU) nunca
frena la detección de somnolencia, que corre con MediaPipe en cada fotograma.
Si en clase va lento, suba el intervalo (p. ej. 4 s) o desactive las
expresiones: la detección de somnolencia no se ve afectada.

Las etiquetas describen expresiones visibles, NO emociones internas ni
diagnósticos.
"""
from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from core.config import EXPRESION_INTERVALO_S, FACE_DEVICE, FACE_MODEL_ID

log = logging.getLogger("driveguard.faces")

LABELS = {"angry": "enojo aparente", "disgust": "disgusto aparente", "fear": "miedo aparente",
          "happy": "alegría aparente", "sad": "tristeza aparente",
          "surprise": "sorpresa aparente", "neutral": "expresión neutral"}
# Expresiones que el motor de riesgo cuenta como "negativas"
NEGATIVAS = {"angry", "disgust", "fear", "sad"}


@lru_cache(maxsize=1)
def detector():
    """Detector Haar de OpenCV (igual que en V3)."""
    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    d = cv2.CascadeClassifier(xml)
    if d.empty():
        raise RuntimeError("No se pudo cargar el detector Haar de OpenCV")
    return d


@lru_cache(maxsize=1)
def classifier():
    """Pipeline de transformers con el modelo ViT (se descarga la primera vez)."""
    from transformers import pipeline
    import torch
    usar_gpu = torch.cuda.is_available() and FACE_DEVICE != "cpu"
    return pipeline("image-classification", model=FACE_MODEL_ID, device=0 if usar_gpu else -1)


def clasificar_recorte(recorte_bgr: np.ndarray, top_k: int = 7) -> list[dict]:
    """Clasifica un recorte de rostro y devuelve las predicciones ordenadas."""
    rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
    preds = classifier()(Image.fromarray(rgb), top_k=top_k)
    return [{"etiqueta_modelo": p["label"].lower(),
             "etiqueta": LABELS.get(p["label"].lower(), p["label"]),
             "score": round(float(p["score"]), 4)} for p in preds]


def detect_faces(frame_bgr: np.ndarray, max_faces: int = 8):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    boxes = detector().detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(45, 45))
    return sorted([tuple(map(int, b)) for b in boxes], key=lambda b: b[2] * b[3], reverse=True)[:max_faces]


def recortar(frame_bgr: np.ndarray, bbox, margen: float = 0.10) -> np.ndarray:
    """Recorta el rostro con un margen moderado."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox
    pad = int(min(bw, bh) * margen)
    return frame_bgr[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]


def analyze_frame(frame_bgr: np.ndarray, threshold: float = 0.40, infer: bool = True) -> dict:
    """Compatibilidad con V3: Haar + ViT sobre un fotograma completo."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Fotograma vacío")
    resultados = []
    for bbox in detect_faces(frame_bgr):
        item = {"bbox": list(bbox), "etiqueta": "sin_clasificar", "confianza": None, "predicciones": []}
        if infer:
            preds = clasificar_recorte(recortar(frame_bgr, bbox))
            item["predicciones"] = preds
            if preds:
                item["confianza"] = preds[0]["score"]
                item["etiqueta"] = preds[0]["etiqueta"] if preds[0]["score"] >= threshold else "incierto"
        resultados.append(item)
    return {"modalidad": "rostros", "rostros_detectados": len(resultados), "rostros": resultados,
            "aviso": "Estimaciones de expresiones faciales visibles, no emociones reales."}


class AnalizadorExpresiones:
    """Clasifica expresiones en segundo plano sin bloquear el video.

    Uso: en cada fotograma se llama `enviar(recorte)`; el hilo toma solo el
    recorte más reciente y lo clasifica si ya pasó el intervalo. `ultimo`
    contiene la última predicción y `nuevo_resultado()` indica si hay una
    predicción que el motor de riesgo aún no ha consumido.
    """

    def __init__(self, intervalo_s: float = EXPRESION_INTERVALO_S, umbral: float = 0.40):
        self.intervalo_s = intervalo_s
        self.umbral = umbral
        self.ultimo: dict | None = None
        self.error: str | None = None
        self.tiempo_inferencia_s: float | None = None
        self._pendiente: np.ndarray | None = None
        self._t_ultimo = 0.0
        self._nuevo = False
        self._lock = threading.Lock()
        self._evento = threading.Event()
        self._fin = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, name="expresiones", daemon=True)
        self._hilo.start()

    def listo_para_nuevo(self) -> bool:
        return time.monotonic() - self._t_ultimo >= self.intervalo_s

    def enviar(self, recorte_bgr: np.ndarray | None) -> None:
        if recorte_bgr is None or recorte_bgr.size == 0 or not self.listo_para_nuevo():
            return
        with self._lock:
            self._pendiente = recorte_bgr.copy()
        self._evento.set()

    def nuevo_resultado(self) -> dict | None:
        """Devuelve la última predicción una sola vez (o None)."""
        with self._lock:
            if not self._nuevo:
                return None
            self._nuevo = False
            return self.ultimo

    def detener(self) -> None:
        self._fin.set()
        self._evento.set()

    def _bucle(self) -> None:
        while not self._fin.is_set():
            self._evento.wait(0.5)
            self._evento.clear()
            with self._lock:
                recorte, self._pendiente = self._pendiente, None
            if recorte is None:
                continue
            self._t_ultimo = time.monotonic()
            try:
                inicio = time.perf_counter()
                preds = clasificar_recorte(recorte)
                self.tiempo_inferencia_s = time.perf_counter() - inicio
                top = preds[0] if preds else None
                resultado = {
                    "etiqueta": top["etiqueta"] if top and top["score"] >= self.umbral else "incierto",
                    "etiqueta_modelo": top["etiqueta_modelo"] if top else None,
                    "score": top["score"] if top else None,
                    "negativa": bool(top and top["score"] >= self.umbral
                                     and top["etiqueta_modelo"] in NEGATIVAS),
                }
                with self._lock:
                    self.ultimo, self._nuevo, self.error = resultado, True, None
            except Exception as exc:  # modelo sin descargar, sin internet, etc.
                self.error = str(exc)
                log.error("Error clasificando expresión: %s", exc)
