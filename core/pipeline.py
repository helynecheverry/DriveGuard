"""Pipeline completo por fotograma, compartido por la cámara en vivo,
el análisis de videos de prueba y los notebooks.

    fotograma -> MediaPipe (EAR, MAR, pose) -> [CNN ojos] -> [ViT en hilo aparte]
              -> MotorRiesgo -> Alarma local -> dibujo -> eventos para MongoDB
"""
from __future__ import annotations

import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from core.config import (CNN_OJOS_PATH, EXPRESION_INTERVALO_S, EXPRESIONES_ACTIVAS, OJOS_METODO,
                         ConfigRiesgo)
from core.landmarks import (CONTORNO_BOCA, CONTORNO_OJO_DER, CONTORNO_OJO_IZQ, OJO_DERECHO,
                            OJO_IZQUIERDO, DetectorRostro, ResultadoRostro, recorte_ojo)
from core.riesgo import EstadoRiesgo, Medicion, MotorRiesgo

COLORES = {"NORMAL": (80, 190, 60), "BAJO": (0, 215, 255), "MEDIO": (0, 140, 255), "ALTO": (40, 40, 230)}
TEXTO_MOTIVO = {"microsueno": "MICROSUENO", "cabeza_caida": "CABEZA CAIDA",
                "cansancio_perclos": "CANSANCIO (PERCLOS)", "bostezos": "BOSTEZOS",
                "cabeceos": "CABECEOS", "distraccion": "DISTRACCION",
                "rostro_ausente": "ROSTRO AUSENTE", "expresion_negativa": "EXPRESION NEGATIVA"}


def _ascii(texto: str) -> str:
    """OpenCV no dibuja tildes ni eñes: se eliminan para el texto sobre el video."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


class ProcesadorConductor:
    """Procesa fotogramas y mantiene el estado de riesgo de una sesión.

    Es seguro llamar `snapshot()`, `extraer_eventos()` y `reconocer()` desde el
    hilo de Streamlit mientras `procesar()` corre en el hilo de video.
    """

    def __init__(self, config: ConfigRiesgo | None = None, expresiones: bool = EXPRESIONES_ACTIVAS,
                 intervalo_expresiones_s: float = EXPRESION_INTERVALO_S, alarma=True,
                 metodo_ojos: str = OJOS_METODO, intervalo_muestra_s: float = 5.0,
                 detector: DetectorRostro | None = None):
        self.detector = detector or DetectorRostro(modo="video")
        self.motor = MotorRiesgo(config)
        self.intervalo_muestra_s = intervalo_muestra_s

        self.expr = None
        if expresiones:
            from core.faces import AnalizadorExpresiones
            self.expr = AnalizadorExpresiones(intervalo_s=intervalo_expresiones_s)

        if alarma is True:
            from core.alarma import Alarma
            self.alarma = Alarma()
        else:
            self.alarma = alarma or None   # instancia propia o None

        self.cnn = None
        self.aviso_cnn = None
        if metodo_ojos == "cnn":
            try:
                from core.ojos_cnn import ClasificadorOjos
                self.cnn = ClasificadorOjos(CNN_OJOS_PATH)
            except Exception as exc:
                self.aviso_cnn = f"CNN no disponible, se usa EAR: {exc}"

        self._lock = threading.Lock()
        self._eventos: list[dict] = []
        self._ultimo_nivel = "NORMAL"
        self._t_muestra = -1e9
        self._t_prev: float | None = None
        self.fps = 0.0
        self.expresion: dict | None = None
        self.estado = EstadoRiesgo(t=0.0)
        self._ojos: dict = {}

    # ------------------------------------------------------------ hilo video
    def procesar(self, frame_bgr: np.ndarray, t: float | None = None) -> tuple[np.ndarray, EstadoRiesgo]:
        """Analiza un fotograma BGR y devuelve (fotograma anotado, estado)."""
        t = time.monotonic() if t is None else t
        rostro = self.detector.procesar(frame_bgr, int(t * 1000))

        if rostro is not None:
            med = Medicion(t=t, rostro=True, ear=rostro.ear, mar=rostro.mar,
                           pitch=rostro.pitch, yaw=rostro.yaw)
            if self.expr is not None:
                from core.faces import recortar
                self.expr.enviar(recortar(frame_bgr, rostro.bbox))
                nuevo = self.expr.nuevo_resultado()
                if nuevo is not None:
                    self.expresion = nuevo
                    med.expresion_negativa = nuevo["negativa"]
            if self.cnn is not None:
                recortes = [r for r in (recorte_ojo(frame_bgr, rostro.puntos, OJO_DERECHO),
                                        recorte_ojo(frame_bgr, rostro.puntos, OJO_IZQUIERDO)) if r is not None]
                if recortes:
                    med.ojos_cerrados = bool(self.cnn.prob_cerrado(recortes).mean() > 0.5)
        else:
            med = Medicion(t=t, rostro=False)

        with self._lock:
            self._ojos = {} if rostro is None else {
                "derecho": {"puntos": rostro.puntos[OJO_DERECHO, :2].tolist(), "ear": float(rostro.ear_der)},
                "izquierdo": {"puntos": rostro.puntos[OJO_IZQUIERDO, :2].tolist(), "ear": float(rostro.ear_izq)},
            }
            estado = self.motor.actualizar(med)
            self.estado = estado
            if self._t_prev is not None and t > self._t_prev:
                inst = 1.0 / (t - self._t_prev)
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
            self._t_prev = t
            self._registrar_eventos(estado)

        if self.alarma is not None:
            self.alarma.actualizar(estado.nivel, estado.silenciado)
        return dibujar(frame_bgr, rostro, estado, self.fps, self.expresion), estado

    def _registrar_eventos(self, e: EstadoRiesgo) -> None:
        if e.calibrando:
            return
        base = {"t_s": round(e.t, 3), "fecha": datetime.now(timezone.utc).isoformat(),
                "nivel": e.nivel, "motivos": list(e.motivos), "ear": _r(e.ear),
                "ear_umbral": _r(e.ear_umbral), "mar": _r(e.mar),
                "perclos": _r(e.perclos), "bostezos": e.bostezos, "cabeceos": e.cabeceos,
                "rostro": e.rostro,
                "expresion": (self.expresion or {}).get("etiqueta")}
        if e.nivel != self._ultimo_nivel:
            tipo = e.motivo_principal if e.nivel != "NORMAL" else "recuperacion"
            self._eventos.append({**base, "tipo": tipo or "cambio_nivel",
                                  "nivel_anterior": self._ultimo_nivel})
            self._ultimo_nivel = e.nivel
        if e.t - self._t_muestra >= self.intervalo_muestra_s:
            self._eventos.append({**base, "tipo": "muestra"})
            self._t_muestra = e.t

    # ---------------------------------------------------------- hilo Streamlit
    def snapshot(self) -> dict:
        with self._lock:
            datos = self.estado.a_dict()
            datos["fps"] = round(self.fps, 1)
            # Copias independientes para que la interfaz no modifique el estado del hilo de video.
            datos["ojos"] = {k: {"puntos": [p[:] for p in v["puntos"]], "ear": v["ear"]}
                             for k, v in self._ojos.items()}
        datos["expresion"] = self.expresion
        datos["alarma_backend"] = self.alarma.backend if self.alarma else "desactivada"
        datos["error_expresiones"] = self.expr.error if self.expr else None
        datos["aviso_cnn"] = self.aviso_cnn
        return datos

    def extraer_eventos(self) -> list[dict]:
        with self._lock:
            eventos, self._eventos = self._eventos, []
        return eventos

    def reconocer(self) -> None:
        """Botón "Estoy bien"."""
        with self._lock:
            self.motor.reconocer(self._t_prev if self._t_prev is not None else time.monotonic())
        if self.alarma is not None:
            self.alarma.actualizar("NORMAL", True)

    def recalibrar(self) -> None:
        with self._lock:
            self.motor.recalibrar()

    def actualizar_config(self, config: ConfigRiesgo, intervalo_expresiones_s: float | None = None,
                          volumen: float | None = None, sonido: bool | None = None) -> None:
        with self._lock:
            self.motor.actualizar_config(config)
        if self.expr is not None and intervalo_expresiones_s:
            self.expr.intervalo_s = intervalo_expresiones_s
        if self.alarma is not None:
            self.alarma.configurar(volumen=volumen, activa=sonido)

    def cerrar(self) -> None:
        if self.alarma is not None:
            self.alarma.detener()
        if self.expr is not None:
            self.expr.detener()
        self.detector.cerrar()


def _r(x):
    return None if x is None else round(float(x), 4)


# --------------------------------------------------------------------------
# Dibujo sobre el video
# --------------------------------------------------------------------------
def dibujar(frame_bgr: np.ndarray, rostro: ResultadoRostro | None, e: EstadoRiesgo,
            fps: float = 0.0, expresion: dict | None = None) -> np.ndarray:
    """Dibuja ojos, boca, barra de nivel y métricas sobre una copia del fotograma."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    color = COLORES.get(e.nivel, (255, 255, 255))
    fuente = cv2.FONT_HERSHEY_SIMPLEX

    if rostro is not None:
        for indices in (CONTORNO_OJO_DER, CONTORNO_OJO_IZQ, CONTORNO_BOCA):
            pts = rostro.puntos[indices, :2].astype(np.int32)
            cv2.polylines(out, [pts], True, color, 1, cv2.LINE_AA)
        for indices in (OJO_DERECHO, OJO_IZQUIERDO):
            for numero, punto in enumerate(rostro.puntos[indices, :2].astype(int), 1):
                x, y = map(int, punto)
                cv2.circle(out, (x, y), 2, (255, 230, 60), -1, cv2.LINE_AA)
                if e.calibrando:
                    cv2.putText(out, str(numero), (x + 3, y - 3), fuente, 0.3,
                                (255, 255, 255), 1, cv2.LINE_AA)

    # Barra superior con el nivel
    cv2.rectangle(out, (0, 0), (w, 34), color, -1)
    if e.calibrando:
        titulo = f"CALIBRANDO {int(e.progreso_calibracion * 100)}% - mire al frente con ojos abiertos"
    else:
        motivo = TEXTO_MOTIVO.get(e.motivo_principal or "", "")
        titulo = f"RIESGO {e.nivel}" + (f" - {motivo}" if motivo else "")
        if e.silenciado:
            titulo += " (silenciado)"
    cv2.putText(out, _ascii(titulo), (10, 24), fuente, 0.62, (0, 0, 0) if e.nivel in ("BAJO", "MEDIO") else (255, 255, 255), 2, cv2.LINE_AA)
    if e.calibrando:
        cv2.rectangle(out, (0, 34), (int(w * e.progreso_calibracion), 40), (255, 255, 255), -1)

    # Panel de métricas
    lineas = [f"EAR {e.ear:.3f} / umbral {e.ear_umbral:.3f}" if e.ear is not None else "EAR --",
              f"MAR {e.mar:.2f}" if e.mar is not None else "MAR --",
              f"PERCLOS {e.perclos * 100:.0f}%  Bostezos {e.bostezos}  Cabeceos {e.cabeceos}",
              f"Ojos cerrados {e.tiempo_ojos_cerrados:.1f}s  FPS {fps:.0f}"]
    if expresion:
        lineas.append(f"Expresion: {expresion.get('etiqueta')}")
    y = 60
    for linea in lineas:
        cv2.putText(out, _ascii(linea), (11, y + 1), fuente, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, _ascii(linea), (10, y), fuente, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    if rostro is None and not e.calibrando:
        cv2.putText(out, "ROSTRO NO DETECTADO", (10, h - 20), fuente, 0.8, (40, 40, 230), 2, cv2.LINE_AA)
    if e.emergencia and int(time.monotonic() * 2) % 2 == 0:
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
        cv2.putText(out, "CONDUCTOR NO RESPONDE", (10, h - 50), fuente, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------
# Análisis de un video grabado (sin cámara en vivo)
# --------------------------------------------------------------------------
def analizar_video(ruta: str | Path, config: ConfigRiesgo | None = None, expresiones: bool = False,
                   salida: str | Path | None = None, max_segundos: float | None = None,
                   progreso=None):
    """Recorre un video y devuelve un DataFrame con métricas por fotograma.

    Útil en clase para trabajar sin conducir: se graba un video de 1-2 minutos
    simulando parpadeos, microsueños y bostezos y se analiza aquí.
    `salida` (opcional) guarda el video anotado en .mp4.
    """
    import pandas as pd

    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video {ruta}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    proc = ProcesadorConductor(config=config, expresiones=expresiones, alarma=False)
    escritor = None
    filas, i = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            if max_segundos and t > max_segundos:
                break
            anotado, e = proc.procesar(frame, t=t)
            if salida is not None:
                if escritor is None:
                    alto, ancho = anotado.shape[:2]
                    escritor = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ancho, alto))
                escritor.write(anotado)
            filas.append({"t": t, "nivel": e.nivel, "ear": e.ear, "ear_umbral": e.ear_umbral,
                          "mar": e.mar, "perclos": e.perclos, "ojos_cerrados": e.ojos_cerrados,
                          "tiempo_ojos_cerrados": e.tiempo_ojos_cerrados, "bostezos": e.bostezos,
                          "cabeceos": e.cabeceos, "rostro": e.rostro, "calibrando": e.calibrando,
                          "emergencia": e.emergencia, "motivos": ",".join(e.motivos)})
            i += 1
            if progreso and total:
                progreso(min(1.0, i / total))
    finally:
        cap.release()
        if escritor is not None:
            escritor.release()
        proc.cerrar()
    return pd.DataFrame(filas)
